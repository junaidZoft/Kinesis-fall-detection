# app.py
import streamlit as st
import cv2
import threading
import time
import queue
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# === CONFIG ===
RTSP_URL_DEFAULT = "rtsp://vmnavas:Zoft%402025@192.168.68.50:554/stream1"
MODEL_PATH_DEFAULT = "fall.pt"  # put your fall.pt in same folder or give full path
IMGSZ = 640            # inference size (adjustable)
CONF_THRES = 0.35      # confidence threshold
NMS_IOU = 0.45         # NMS IoU
QUEUE_SIZE = 4         # small queue to avoid memory growth
DISPLAY_FPS = True     # overlay FPS
# ==============

st.set_page_config(page_title="Fall Detection Stream", layout="wide")

st.title("Real-time Fall Detection (YOLO) — Streamlit")
col1, col2 = st.columns([3,1])

with col2:
    st.markdown("### Settings")
    rtsp_url = st.text_input("RTSP URL", RTSP_URL_DEFAULT)
    model_path = st.text_input("YOLO weights (best.pt)", MODEL_PATH_DEFAULT)
    imgsz = st.number_input("Inference image size (px)", min_value=224, max_value=1280, value=IMGSZ, step=32)
    conf_thres = st.slider("Confidence threshold", 0.05, 0.99, CONF_THRES, 0.01)
    use_cuda = st.checkbox("Use CUDA (if available)", value=torch.cuda.is_available())
    start_stop = st.button("Start/Restart Stream")

# Display area
frame_placeholder = col1.empty()
log = st.sidebar.empty()

# Thread-safe queues
frame_q = queue.Queue(maxsize=QUEUE_SIZE)
out_q = queue.Queue(maxsize=QUEUE_SIZE)

# Global flags
stop_event = threading.Event()
model_lock = threading.Lock()

@st.cache_resource(ttl=3600)
def load_model(weights_path: str, device: str, imgsz:int):
    """
    Load ultralytics/YOLOv8 style model (torch) or generic TorchScript.
    This uses torch.hub or a direct torch.load depending on weight type.
    We'll prefer ultralytics if installed. If not present, fallback to torch.
    """
    # Try to use ultralytics (recommended) - it provides simple .predict or __call__ with numpy
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        model.fuse()  # fuse conv+bn for speed if supported
        # set device and half if possible
        model.to(device)
        if device.startswith("cuda"):
            try:
                model.model.half()  # attempt half precision (may fail for some models)
            except Exception:
                pass
        return ("ultralytics", model)
    except Exception:
        # fallback to torch.load state dict and custom inference (less ideal)
        st.warning("ultralytics package not available; fallback may be slower. Install `ultralytics` for best performance.")
        # Try torch.hub YOLOv5/YOLOv8 style? Attempt loading with torch.hub
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
            model.to(device)
            if device.startswith("cuda"):
                model.half()
            return ("yolov5_hub", model)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            raise

def video_capture_thread(rtsp_url, cap_q, stop_event):
    """Capture frames from RTSP and put into cap_q."""
    # Use OpenCV with FFMPEG backend
    # Try to set buffer parameters for low-latency
    pipeline = rtsp_url
    cap = cv2.VideoCapture(pipeline, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        st.error("Failed to open RTSP stream. Check URL and encoding.")
        return

    # reduce latency options when possible
    # Note: these backend properties may not work everywhere; they're best-effort
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # small wait & retry
            time.sleep(0.05)
            continue
        # Push frame: if queue full, drop the oldest to keep up with live stream
        try:
            if cap_q.full():
                try:
                    _ = cap_q.get_nowait()
                except queue.Empty:
                    pass
            cap_q.put_nowait(frame)
        except queue.Full:
            pass
    cap.release()

def inference_thread(cap_q, out_q, stop_event, model_info, device, imgsz, conf_thres, nms_iou):
    """
    Runs inference on frames from cap_q, puts annotated frames to out_q.
    model_info: tuple("ultralytics", model_obj) or ("yolov5_hub", model)
    """
    backend_type, model = model_info
    last_time = time.time()
    frames_processed = 0
    while not stop_event.is_set():
        try:
            frame = cap_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # Prepare for model
        img = cv2.resize(frame, (imgsz, imgsz))
        # convert BGR->RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference depending on backend
        preds = None
        try:
            if backend_type == "ultralytics":
                # ultralytics YOLO: returns Results object
                # use model.predict to control options
                results = model.predict(source=[img_rgb], imgsz=imgsz, conf=conf_thres, iou=nms_iou, device=device, half=(device.startswith("cuda")))
                # results is a list; get first
                results0 = results[0]
                boxes = results0.boxes  # Boxes object
                annotated = frame.copy()  # draw on original frame for correct coordinates
                # Map classes to labels
                h, w = frame.shape[:2]
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()  # (x1,y1,x2,y2) in model input scale
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = "FALL" if cls == 0 else "NO-FALL"
                    color = (0,0,255) if cls == 0 else (0,255,0)
                    # scale coordinates to original frame size (use float for precision)
                    x1 = int(round(xyxy[0] * w / imgsz))
                    y1 = int(round(xyxy[1] * h / imgsz))
                    x2 = int(round(xyxy[2] * w / imgsz))
                    y2 = int(round(xyxy[3] * h / imgsz))
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(y1-8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                annotated_frame = annotated
            else:
                # yolov5 hub fallback
                # model expects BGR images usually
                res = model(img)  # will handle resizing
                annotated = np.squeeze(res.render())  # rendered BGR
                annotated_frame = cv2.resize(annotated, (frame.shape[1], frame.shape[0]))
        except Exception as e:
            # In-case of an inference error, show raw frame
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"Inference error: {e}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

        # overlay FPS
        frames_processed += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frames_processed / (now - last_time)
            frames_processed = 0
            last_time = now
            fps_text = f"{fps:.1f} FPS"
        else:
            fps_text = ""

        if DISPLAY_FPS and fps_text:
            cv2.putText(annotated_frame, fps_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

        # send out; if queue full drop oldest
        try:
            if out_q.full():
                try:
                    _ = out_q.get_nowait()
                except queue.Empty:
                    pass
            out_q.put_nowait(annotated_frame)
        except queue.Full:
            pass

# Manage threads and model lifecycle
def start_pipeline(rtsp_url, model_path, imgsz, conf_thres, use_cuda):
    stop_event.clear()
    device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
    # load model once
    with model_lock:
        model_info = load_model(model_path, device=device, imgsz=imgsz)
    # start capture thread
    cap_thread = threading.Thread(target=video_capture_thread, args=(rtsp_url, frame_q, stop_event), daemon=True)
    inf_thread = threading.Thread(target=inference_thread, args=(frame_q, out_q, stop_event, model_info, device, imgsz, conf_thres, NMS_IOU), daemon=True)
    cap_thread.start()
    inf_thread.start()
    return cap_thread, inf_thread

# Control start/stop
if 'threads' not in st.session_state:
    st.session_state['threads'] = None

if start_stop:
    # stop if running and restart
    if st.session_state['threads'] is not None:
        stop_event.set()
        time.sleep(0.3)
        stop_event.clear()
    try:
        threads = start_pipeline(rtsp_url, model_path, imgsz, conf_thres, use_cuda)
        st.session_state['threads'] = threads
        log.text(f"Started pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        st.error(f"Failed to start pipeline: {e}")
        st.session_state['threads'] = None

# Main display loop (runs in Streamlit app)
try:
    while True:
        if st.session_state.get('threads') is None:
            frame_placeholder.image(np.zeros((480,640,3), dtype=np.uint8), channels="BGR", caption="No stream running")
            time.sleep(0.5)
            break
        try:
            frame = out_q.get(timeout=1.0)
            # convert BGR to RGB for Streamlit if necessary; we can use channels="BGR"
            frame_placeholder.image(frame, channels="BGR")
        except queue.Empty:
            # no frame yet
            time.sleep(0.05)
            continue
except KeyboardInterrupt:
    pass
finally:
    # cleanup on app end
    stop_event.set()
    time.sleep(0.2)
