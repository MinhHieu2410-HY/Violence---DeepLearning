# Demo một file: ảnh, video và webcam trên máy chạy Streamlit.
# Python 3.11 hoặc 3.12. Cài trong môi trường của bạn:
# python -m pip install streamlit ultralytics opencv-python numpy Pillow imageio-ffmpeg
# Chạy: python -m streamlit run demo.py
# Không dùng WebRTC, aiortc hoặc PyAV.
# Video kết quả: MP4/H.264 phát trên web, không âm thanh.
# FFmpeg: dùng ffmpeg.exe trong PATH hoặc cài imageio-ffmpeg (kèm binary).
# Webcam được OpenCV mở trên máy chạy Python, không lấy từ trình duyệt từ xa.

# ===== CHỈNH ĐƯỜNG DẪN TRỌNG SỐ TẠI ĐÂY =====
# Ví dụ Windows: BEST_MODEL_PATH = r"L:\Learning\Violence\best.pt"
# Dùng file YOLO đã huấn luyện có lớp violence, không phải link chia sẻ Google Drive.
BEST_MODEL_PATH = r"L:\Learning\Violence\Run\ft_02\yolov8n_ft\weights\best.pt"

from dataclasses import dataclass
from pathlib import Path
import csv
import io
import tempfile
import threading
import time
import os
import shutil
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Settings:
    # Nguong chon tu ket qua quet tren valid: conf 0.05, iou 0.55, imgsz 640, agnostic_nms False
    confidence: float = 0.05
    imgsz: int = 640
    device: str = 'cpu'
    nms_iou: float = 0.55
    agnostic: bool = False


def violence_class(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    matches = [int(i) for i, name in items
               if str(name).lower().replace(' ', '').replace('-', '').replace('_', '')
               in {'violence', 'violent', 'baoluc'}]
    if len(matches) != 1:
        raise ValueError(f'Checkpoint phải có đúng một lớp violence. Các lớp hiện tại: {names}')
    return matches[0]


def select_boxes(data, class_id, confidence):
    a = np.asarray(data, dtype=float).reshape(-1, 6)
    return a[(a[:, 5] == class_id) & (a[:, 4] >= confidence)].copy()


def draw_violence(frame, boxes):
    out = frame.copy()
    for x1, y1, x2, y2, score, _ in boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (45, 45, 240), 2)
        label = f'BAO LUC {score:.0%}'
        y = max(22, y1)
        cv2.rectangle(out, (x1, y - 22), (min(out.shape[1], x1 + 170), y), (45, 45, 220), -1)
        cv2.putText(out, label, (x1 + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
    if len(boxes):
        cv2.rectangle(out, (0,0), (out.shape[1]-1, out.shape[0]-1), (45,45,240), 5)
    return out


class Detector:
    def __init__(self, weights):
        if not Path(weights).is_file():
            raise FileNotFoundError(f'Không tìm thấy trọng số: {weights}. Sửa BEST_MODEL_PATH ở đầu demo.py.')
        self.model = YOLO(str(weights))
        self.class_id = violence_class(self.model.names)
        self.lock = threading.Lock()

    def predict_frame(self, frame, settings):
        started = time.perf_counter()
        # Same low candidate threshold / NMS settings as the sweep notebook (agnostic_nms=False).
        # Filter only AFTER inference: negative-class detections are never drawn.
        with self.lock:
            r = self.model.predict(frame, conf=0.001, iou=settings.nms_iou, imgsz=settings.imgsz,
                                   agnostic_nms=settings.agnostic,
                                   device=settings.device, verbose=False, max_det=300)[0]
            data = r.boxes.data.cpu().numpy() if r.boxes is not None else np.empty((0,6))
        boxes = select_boxes(data, self.class_id, settings.confidence)
        violence = data[data[:, 5] == self.class_id]
        score = float(violence[:, 4].max()) if len(violence) else 0.0
        return draw_violence(frame, boxes), boxes, time.perf_counter() - started, score

    def infer(self, frame, settings):
        out, boxes, elapsed, _ = self.predict_frame(frame, settings)
        return out, boxes, elapsed


def csv_bytes(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=['frame', 'time_seconds', 'violence_boxes', 'max_confidence', 'threshold', 'alert'])
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode('utf-8-sig')


def ffmpeg_executable():
    binary = shutil.which('ffmpeg')
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError('Video MP4 cần FFmpeg. Cài: python -m pip install imageio-ffmpeg')


def run_ffmpeg(binary, args, cancel, log_path):
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with open(log_path, 'wb') as log:
        proc = subprocess.Popen([binary, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', *args],
                                stdout=subprocess.DEVNULL, stderr=log, creationflags=flags)
        try:
            while proc.poll() is None:
                if cancel.wait(0.1):
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise InterruptedError('Đã dừng xử lý.')
            if proc.returncode:
                raise RuntimeError('FFmpeg không xử lý được video: ' + Path(log_path).read_text(errors='replace')[-1500:])
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class VideoJob:
    def __init__(self, data, suffix, detector, settings):
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.state = dict(status='running', stage='Chuẩn bị video', frames=0, total=0,
                          alerts=0, error='', video=None, csv=None, elapsed=0.0, fps=0.0)
        self.thread = threading.Thread(target=self._run, args=(data, suffix, detector, settings), daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.state.copy()

    def update(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def _run(self, data, suffix, detector, settings):
        started = time.perf_counter()
        cap, writer = None, None
        try:
            binary = ffmpeg_executable()
            with tempfile.TemporaryDirectory(prefix='violence_') as folder:
                src = Path(folder)/('input'+suffix)
                normalized = Path(folder)/'frames.mkv'
                raw = Path(folder)/'annotated.avi'
                dst = Path(folder)/'result.mp4'
                log = Path(folder)/'ffmpeg.log'
                src.write_bytes(data)
                del data
                probe = cv2.VideoCapture(str(src))
                try:
                    if not probe.isOpened():
                        raise ValueError('Không đọc được video đầu vào.')
                    fps = probe.get(cv2.CAP_PROP_FPS)
                finally:
                    probe.release()
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError('Không xác định được FPS nguồn. Hãy chuyển đầu vào sang MP4 có FPS cố định.')
                # Honor timestamps/rotation, normalize VFR to the source average FPS.
                # Lossless normalization avoids degraded input to the detector.
                self.update(stage='Chuẩn hóa thời gian và hướng video', fps=fps)
                run_ffmpeg(binary, ['-i', str(src), '-map', '0:v:0', '-an',
                                   '-vf', f'fps={fps:.9f}', '-c:v', 'ffv1', str(normalized)], self.cancel, log)
                cap = cv2.VideoCapture(str(normalized))
                try:
                    if not cap.isOpened():
                        raise ValueError('Không đọc được video sau chuẩn hóa.')
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.update(total=max(0,total), stage='Suy luận từng frame')
                    rows, count, alerts = [], 0, 0
                    try:
                        while not self.cancel.is_set():
                            ok, frame = cap.read()
                            if not ok:
                                break
                            h, w = frame.shape[:2]
                            if writer is None:
                                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'FFV1'),
                                                         fps, (w + w%2, h + h%2))
                                if not writer.isOpened():
                                    raise ValueError('Không tạo được video trung gian.')
                            # No temporal sampling; no reuse of boxes from another frame.
                            out, boxes, _, score = detector.predict_frame(frame, settings)
                            alerts += bool(len(boxes))
                            rows.append(dict(frame=count, time_seconds=round(count/fps,4),
                                             violence_boxes=len(boxes), max_confidence=round(score,6),
                                             threshold=settings.confidence, alert=int(bool(len(boxes)))))
                            out = cv2.copyMakeBorder(out, 0, h%2, 0, w%2, cv2.BORDER_CONSTANT)
                            writer.write(out)
                            count += 1
                            self.update(frames=count, alerts=alerts, elapsed=time.perf_counter()-started)
                    finally:
                        if writer is not None:
                            writer.release()
                            writer = None
                finally:
                    cap.release()
                    cap = None
                if self.cancel.is_set():
                    raise InterruptedError('Đã dừng xử lý.')
                if count == 0:
                    raise ValueError('Video không có khung hình đọc được.')
                if total > 0 and count < total - 2:
                    raise ValueError(f'Video bị lỗi giải mã: {count}/{total} frame.')
                self.update(stage='Mã hóa MP4 hoàn chỉnh cho trình duyệt')
                run_ffmpeg(binary, ['-i', str(raw), '-an', '-c:v', 'libx264',
                                   '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                                   '-movflags', '+faststart', str(dst)], self.cancel, log)
                self.update(status='done', stage='Hoàn tất', video=dst.read_bytes(),
                            csv=csv_bytes(rows), elapsed=time.perf_counter()-started)
        except InterruptedError:
            self.update(status='cancelled')
        except Exception as exc:
            self.update(status='error', error=str(exc))
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()


class LocalCamera:
    """Session-owned capture; release on Stop, mode switch and object disposal."""
    def __init__(self, index):
        self.cap = None
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
        if self.cap is None:
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.close()
            raise RuntimeError('Không mở được webcam. Đóng Zoom/Teams/Camera hoặc thử ID camera khác.')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frames = 0

    def read(self):
        ok, frame = self.cap.read() if self.cap is not None else (False, None)
        if not ok:
            raise RuntimeError('Không đọc được khung hình từ webcam.')
        self.frames += 1
        return frame

    def close(self):
        cap, self.cap = self.cap, None
        if cap is not None:
            cap.release()

    def __del__(self):
        self.close()


from pathlib import Path
import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
import torch



def main():
    st.set_page_config(page_title='Violence Monitor', page_icon='🔴', layout='wide')
    st.markdown('''<style>
    .block-container {padding-top:2rem; max-width:1300px}
    [data-testid="stMetric"] {background:#172033; border:1px solid #2c3952; border-radius:12px; padding:14px}
    .hero {padding:24px 28px; border:1px solid #334155; border-radius:18px; background:linear-gradient(110deg,#172033,#292034); margin-bottom:22px}
    .hero h1 {margin:0; font-size:2.2rem}.hero p {color:#cbd5e1; margin:8px 0 0}
    </style><div class="hero"><h1><span style="color:#ef4444">●</span> Violence Monitor</h1><p>Nhận diện bạo lực · Ảnh, video & camera trực tiếp</p></div>''', unsafe_allow_html=True)

    @st.cache_resource
    def load_detector(weights_path, modified_ns):
        return Detector(weights_path)

    with st.sidebar:
        st.title('Điều khiển')
        mode = st.radio('Nguồn đầu vào', ['Ảnh', 'Video', 'Camera realtime'])
        preset = st.selectbox('Độ nhạy', ['Theo kết quả quét · ngưỡng 0,05', 'Demo · ngưỡng 0,25', 'Tùy chỉnh'])
        if preset == 'Tùy chỉnh':
            confidence = st.slider('Ngưỡng tin cậy', 0.01, 0.95, 0.05, 0.01)
        else:
            confidence = 0.05 if 'quét' in preset else 0.25
        st.caption('Giảm ngưỡng để phát hiện nhiều hơn; tăng ngưỡng để giảm cảnh báo nhầm.')
        size = st.select_slider('Kích thước suy luận', [320, 416, 640, 960, 1280], value=640)
        nms_iou = st.slider('IoU gộp khung trùng', 0.3, 0.7, 0.55, 0.05)
        options = ['CPU'] + (['GPU NVIDIA'] if torch.cuda.is_available() else [])
        device_name = st.selectbox('Thiết bị', options, index=len(options)-1)
        st.divider()
        st.caption('YOLOv8n • checkpoint đã huấn luyện của bài tập lớn')
        st.caption('Chỉ hiển thị khung và cảnh báo bạo lực.')
        if size != 640 or abs(nms_iou - 0.55) > 1e-9 or (preset != 'Tùy chỉnh' and 'quét' not in preset):
            st.caption('Khác cấu hình đã quét (ngưỡng 0,05 · IoU 0,55 · kích thước 640): kết quả có thể khác số liệu trong báo cáo.')

    settings = Settings(confidence, size, '0' if device_name == 'GPU NVIDIA' else 'cpu', nms_iou, False)
    try:
        weights = Path(os.environ.get('BEST_MODEL_PATH', BEST_MODEL_PATH)).expanduser()
        if not weights.is_absolute():
            weights = Path(__file__).resolve().parent / weights
        detector = load_detector(str(weights), weights.stat().st_mtime_ns if weights.is_file() else 0)
    except Exception as exc:
        st.error(f'Không tải được mô hình: {exc}')
        st.stop()

    if mode != 'Camera realtime' and st.session_state.get('camera') is not None:
        st.session_state.camera.close()
        st.session_state.camera = None

    if mode != 'Video' and 'video_job' in st.session_state:
        job = st.session_state.video_job
        if job.snapshot()['status'] == 'running':
            job.cancel.set()

    if mode == 'Ảnh':
        st.subheader('Phân tích ảnh')
        file = st.file_uploader('Chọn ảnh JPG, PNG hoặc WEBP', type=['jpg','jpeg','png','webp'])
        if file:
            try:
                pil = ImageOps.exif_transpose(Image.open(file)).convert('RGB')
                if pil.width * pil.height > 25_000_000:
                    st.error('Ảnh quá lớn. Vui lòng giảm xuống dưới 25 megapixel.')
                    st.stop()
                original = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                with st.spinner('Đang phân tích ảnh…'):
                    result, boxes, elapsed = detector.infer(original, settings)
                a,b = st.columns(2)
                a.image(pil, caption='Ảnh đầu vào', width='stretch')
                b.image(result, channels='BGR', caption='Kết quả', width='stretch')
                if len(boxes):
                    st.error(f'⚠️ PHÁT HIỆN BẠO LỰC · {len(boxes)} vùng · điểm cao nhất {boxes[:,4].max():.1%}')
                st.caption(f'Đã xử lý ảnh trong {elapsed:.2f} giây.')
                ok, encoded = cv2.imencode('.jpg', result)
                if ok:
                    st.download_button('Tải ảnh kết quả', encoded.tobytes(), 'ket_qua_bao_luc.jpg', 'image/jpeg')
            except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
                st.error(f'Không xử lý được ảnh: {exc}')
        else:
            st.info('Tải một ảnh lên để bắt đầu.')

    elif mode == 'Video':
        st.subheader('Phân tích video')
        st.caption('Xử lý toàn bộ video, sau đó phát kết quả MP4 hoàn chỉnh bên dưới. Không có âm thanh.')
        with st.expander('Cài bộ mã hóa MP4 nếu máy chưa có FFmpeg'):
            st.code('python -m pip install imageio-ffmpeg', language='bash')
        file = st.file_uploader('Chọn video (tối đa 200 MB)', type=['mp4','mov','avi','mkv','webm'])
        existing = st.session_state.get('video_job')
        running = existing is not None and existing.snapshot()['status'] == 'running'
        if st.button('Bắt đầu phân tích', type='primary', disabled=file is None or running):
            if file.size > 200*1024*1024:
                st.error('Video vượt quá 200 MB.')
            else:
                st.session_state.video_job = VideoJob(file.getvalue(), Path(file.name).suffix, detector, settings)
                st.session_state.video_source = file.name
                st.session_state.video_settings = f'Ngưỡng {confidence:.2f} · kích thước {size} · NMS IoU {nms_iou:.2f} · {weights.name}'
                st.rerun()

        @st.fragment(run_every=0.5 if running else None)
        def show_video_job():
            job = st.session_state.get('video_job')
            if job is None:
                return
            s = job.snapshot()
            if s['status'] != 'running' and st.session_state.get('terminal_job') != id(job):
                st.session_state.terminal_job = id(job)
                st.rerun()
            st.caption(f"Kết quả của: {st.session_state.video_source} · {st.session_state.video_settings}")
            a,b,c = st.columns(3)
            a.metric('Khung hình đã xử lý', s['frames'])
            b.metric('Khung hình có cảnh báo', s['alerts'])
            c.metric('Thời gian xử lý', f"{s['elapsed']:.1f} s")
            if s['status'] == 'running':
                progress = min(s['frames']/s['total'], 1.0) if s['total'] else 0.0
                st.progress(progress, text=s['stage'])
                if st.button('Dừng xử lý', key='stop_video'):
                    job.cancel.set()
            elif s['status'] == 'done':
                st.success('Đã xử lý xong video. Bấm Play để xem kết quả hoàn chỉnh.')
                st.video(s['video'], format='video/mp4')
                st.caption(f"{s['frames']} frame · {s['fps']:.3f} FPS · {s['frames']/s['fps']:.2f} giây")
                a,b = st.columns(2)
                a.download_button('Tải video kết quả', s['video'], 'ket_qua_bao_luc.mp4', 'video/mp4')
                b.download_button('Tải điểm dự đoán từng frame', s['csv'], 'nhat_ky_bao_luc.csv', 'text/csv')
            elif s['status'] == 'cancelled':
                st.info('Đã dừng. Bấm Bắt đầu phân tích để chạy lại.')
            else:
                st.error(f"Xử lý thất bại: {s['error']}")
        show_video_job()
        st.caption('Video có FPS biến thiên được chuẩn hóa theo thời gian nguồn. Mỗi frame sau chuẩn hóa đều được suy luận; không lấy mẫu cách frame. CSV ghi cả điểm dưới ngưỡng để kiểm tra bỏ sót.')

    else:
        st.subheader('Camera realtime')
        st.caption('Webcam được mở trực tiếp trên máy chạy Streamlit. Bấm Bắt đầu / Dừng camera.')
        camera = st.session_state.get('camera')
        index = st.number_input('ID camera (thường là 0)', min_value=0, max_value=10,
                                value=0, step=1, disabled=camera is not None)
        a, b = st.columns(2)
        if a.button('Bắt đầu camera', type='primary', disabled=camera is not None):
            try:
                st.session_state.camera = LocalCamera(int(index))
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if b.button('Dừng camera', disabled=camera is None):
            camera.close()
            st.session_state.camera = None
            st.rerun()

        @st.fragment(run_every=0.1)
        def show_camera():
            cam = st.session_state.get('camera')
            if cam is None:
                return
            try:
                frame = cam.read()
                result, boxes, elapsed = detector.infer(frame, settings)
                st.image(result, channels='BGR', width='stretch')
                if len(boxes):
                    st.error(f'⚠️ PHÁT HIỆN BẠO LỰC · {len(boxes)} vùng · {boxes[:,4].max():.1%}')
                st.caption(f'Khung hình: {cam.frames} · Suy luận: {1/max(elapsed,1e-6):.1f} FPS')
            except Exception as exc:
                cam.close()
                st.session_state.camera = None
                st.error(f'Camera đã dừng: {exc}')
        show_camera()
        st.caption('Nếu webcam không mở: đóng ứng dụng khác đang dùng camera, kiểm tra quyền camera Windows và thử ID 1.')

    st.divider()
    st.caption('Demo học thuật • Video và camera được phân tích theo từng khung hình.')


if __name__ == "__main__":
    main()
