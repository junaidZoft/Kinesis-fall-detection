import streamlit as st
import cv2
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import deque
import os
from dotenv import load_dotenv
import boto3
from ultralytics import YOLO
import platform

# === OPTIMIZED CONFIG ===
MODEL_PATH_DEFAULT = "models/fall/fall.pt"
IMGSZ = 416  # Reduced for speed
CONF_THRES = 0.50  # Higher threshold = fewer false positives
FIRE_CONF_THRES = 0.40
WEAPON_CONF_THRES = 0.40
NMS_IOU = 0.50
PERSON_MODEL_PATH = "models/base/yolov8n.pt"
FIRE_MODEL_PATH = "models/fire/fire.pt"
WEAPON_MODEL_PATH = "models/weapon/weapon.pt"
SNAPSHOT_DIR = Path("snapshots")

# Optimization settings
FRAME_SKIP = 2  # Process every Nth frame
RESIZE_WIDTH = 640  # Resize for processing
TEMPORAL_WINDOW = 5  # Frames to confirm detection
FALL_CONFIRMATION_FRAMES = 3  # Must detect in N frames
FIRE_CONFIRMATION_FRAMES = 4
WEAPON_CONFIRMATION_FRAMES = 4
BATCH_SIZE = 1  # Batch processing
MAX_QUEUE_SIZE = 2  # Minimal buffering
# ========================

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

STREAM_NAME = os.getenv("STREAM_NAME")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
S3_PREFIX = os.getenv("AWS_S3_PREFIX", "detections/")

def get_kinesis_hls_url():
    """Get HLS streaming session URL from AWS Kinesis Video Streams."""
    client = boto3.client(
        "kinesisvideo",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    endpoint = client.get_data_endpoint(
        StreamName=STREAM_NAME,
        APIName="GET_HLS_STREAMING_SESSION_URL"
    )["DataEndpoint"]
    kvam = boto3.client(
        "kinesis-video-archived-media",
        endpoint_url=endpoint,
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    resp = kvam.get_hls_streaming_session_url(
        StreamName=STREAM_NAME,
        PlaybackMode="LIVE"
    )
    return resp["HLSStreamingSessionURL"]

def init_directories():
    """Create necessary directories"""
    directories = [
        Path("models/base"), Path("models/fall"), Path("models/fire"), Path("models/weapon"),
        SNAPSHOT_DIR / "fall", SNAPSHOT_DIR / "fire", SNAPSHOT_DIR / "weapon"
    ]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)

def upload_to_s3(file_path: Path, detection_type: str):
    """Upload file to S3 bucket"""
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION,
                                 aws_access_key_id=AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        folder_key = f"{S3_PREFIX}{detection_type}/"
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=folder_key)
        except:
            s3_client.put_object(Bucket=S3_BUCKET, Key=folder_key)
        s3_key = f"{folder_key}{file_path.name}"
        s3_client.upload_file(str(file_path), S3_BUCKET, s3_key)
    except Exception as e:
        print(f"S3 upload failed: {e}")

def beep():
    """Play alert sound"""
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 200)
        else:
            os.system('play -nq -t alsa synth 0.2 sine 1000 2>/dev/null')
    except:
        pass

class TemporalFilter:
    """Reduce false positives by requiring detection consistency"""
    def __init__(self, window_size, confirmation_frames):
        self.window_size = window_size
        self.confirmation_frames = confirmation_frames
        self.detections = deque(maxlen=window_size)
    
    def add_detection(self, detected):
        self.detections.append(detected)
        return sum(self.detections) >= self.confirmation_frames
    
    def reset(self):
        self.detections.clear()

def optimized_capture_process(hls_url, frame_queue, stop_event, fps_counter):
    """Optimized frame capture with minimal buffering"""
    cap = cv2.VideoCapture(hls_url)
    if not cap.isOpened():
        print("Failed to open HLS stream")
        return
    
    # Optimize capture settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    last_fps_time = time.time()
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        
        # Skip frames for speed
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # Resize immediately to reduce processing load
        h, w = frame.shape[:2]
        if w > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / w
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        
        # Drop old frames if queue is full (always use latest)
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                break
        
        try:
            frame_queue.put_nowait(frame)
        except:
            pass
        
        # FPS calculation
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            fps_counter.value = int(frame_count / (current_time - last_fps_time))
            frame_count = 0
            last_fps_time = current_time
    
    cap.release()

def optimized_inference_process(frame_queue, result_queue, stop_event, device_str, conf_thres, fire_conf, weapon_conf):
    """Optimized multi-model inference with temporal filtering"""
    # Load models
    device = torch.device(device_str)
    fall_model = YOLO(MODEL_PATH_DEFAULT).to(device)
    person_model = YOLO(PERSON_MODEL_PATH).to(device)
    fire_model = YOLO(FIRE_MODEL_PATH).to(device)
    weapon_model = YOLO(WEAPON_MODEL_PATH).to(device)
    
    # Enable optimizations
    for model in [fall_model, person_model, fire_model, weapon_model]:
        model.fuse()
        if device_str.startswith("cuda"):
            try:
                model.model.half()
            except:
                pass
    
    # Temporal filters for false positive reduction
    fall_filter = TemporalFilter(TEMPORAL_WINDOW, FALL_CONFIRMATION_FRAMES)
    fire_filter = TemporalFilter(TEMPORAL_WINDOW, FIRE_CONFIRMATION_FRAMES)
    weapon_filter = TemporalFilter(TEMPORAL_WINDOW, WEAPON_CONFIRMATION_FRAMES)
    
    last_fall_save = 0
    last_fire_save = 0
    last_weapon_save = 0
    save_cooldown = 2.0  # Seconds between saves
    
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except:
            continue
        
        h, w = frame.shape[:2]
        annotated = frame.copy()
        current_time = time.time()
        
        # Stage 1: Person Detection (fast)
        person_results = person_model.predict(frame, imgsz=IMGSZ, conf=0.4, iou=0.5, 
                                               device=device, half=(device_str.startswith("cuda")), verbose=False)
        
        person_boxes = []
        for box in person_results[0].boxes:
            if int(box.cls[0]) == 0:  # Person class
                person_boxes.append(box.xyxy[0].cpu().numpy())
        
        # Stage 2: Fall Detection on person crops
        fall_detected_frame = False
        for xyxy in person_boxes:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            
            # Fall classification
            fall_results = fall_model.predict(person_crop, imgsz=IMGSZ, conf=conf_thres, 
                                               iou=NMS_IOU, device=device, 
                                               half=(device_str.startswith("cuda")), verbose=False)
            
            is_fall = False
            fall_conf = 0
            for fbox in fall_results[0].boxes:
                if int(fbox.cls[0]) == 0:  # Fall class
                    fall_conf = float(fbox.conf[0])
                    if fall_conf >= conf_thres:
                        is_fall = True
                        break
            
            if is_fall:
                fall_detected_frame = True
            
            # Draw with confidence
            color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"FALL {fall_conf:.2f}" if is_fall else "OK"
            cv2.putText(annotated, label, (x1, max(y1-10, 0)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Temporal filtering for fall
        fall_confirmed = fall_filter.add_detection(fall_detected_frame)
        if fall_confirmed and (current_time - last_fall_save) > save_cooldown:
            save_dir = SNAPSHOT_DIR / "fall"
            save_dir.mkdir(exist_ok=True, parents=True)
            filename = save_dir / f"fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(filename), annotated)
            upload_to_s3(filename, "fall")
            beep()
            last_fall_save = current_time
            fall_filter.reset()
        
        # Stage 3: Fire Detection (parallel)
        fire_results = fire_model.predict(frame, imgsz=IMGSZ, conf=fire_conf, 
                                           device=device, half=(device_str.startswith("cuda")), verbose=False)
        fire_detected_frame = False
        for box in fire_results[0].boxes:
            cls_name = fire_model.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            if cls_name in ["fire", "smoke"] and conf >= fire_conf:
                fire_detected_frame = True
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                color = (0, 0, 255) if cls_name == "fire" else (128, 128, 128)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated, f"{cls_name.upper()} {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        fire_confirmed = fire_filter.add_detection(fire_detected_frame)
        if fire_confirmed and (current_time - last_fire_save) > save_cooldown:
            save_dir = SNAPSHOT_DIR / "fire"
            save_dir.mkdir(exist_ok=True, parents=True)
            filename = save_dir / f"fire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(filename), annotated)
            upload_to_s3(filename, "fire")
            last_fire_save = current_time
            fire_filter.reset()
        
        # Stage 4: Weapon Detection
        weapon_results = weapon_model.predict(frame, imgsz=IMGSZ, conf=weapon_conf,
                                               device=device, half=(device_str.startswith("cuda")), verbose=False)
        weapon_detected_frame = False
        for box in weapon_results[0].boxes:
            cls_name = weapon_model.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            if cls_name == "weapon" and conf >= weapon_conf:
                weapon_detected_frame = True
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(annotated, f"WEAPON {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        weapon_confirmed = weapon_filter.add_detection(weapon_detected_frame)
        if weapon_confirmed and (current_time - last_weapon_save) > save_cooldown:
            save_dir = SNAPSHOT_DIR / "weapon"
            save_dir.mkdir(exist_ok=True, parents=True)
            filename = save_dir / f"weapon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(filename), annotated)
            upload_to_s3(filename, "weapon")
            last_weapon_save = current_time
            weapon_filter.reset()
        
        # Send result (drop old frames)
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except:
                break
        
        try:
            result_queue.put_nowait(annotated)
        except:
            pass

# Streamlit UI
st.set_page_config(page_title="Optimized Multi-Event Detection", layout="wide", page_icon="⚡")
st.markdown("""
    <style>
        .event-badge {font-size: 1.2em; font-weight: bold; padding: 4px 10px; border-radius: 8px;}
        .fall-badge {background: #ffcccc; color: #b71c1c;}
        .fire-badge {background: #ffe0b2; color: #e65100;}
        .weapon-badge {background: #e1bee7; color: #4a148c;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Optimized Real-time Detection (Low Latency)")
st.markdown("**Optimizations:** Frame skipping, temporal filtering, minimal buffering, batch processing")

col1, col2 = st.columns([4, 1])

with col2:
    st.markdown("#### Settings")
    try:
        hls_url = get_kinesis_hls_url()
        st.success("✅ HLS URL obtained")
    except Exception as e:
        st.error(f"❌ HLS Error: {e}")
        hls_url = None
    
    conf_thres = st.slider("Fall Confidence", 0.3, 0.9, 0.50, 0.05)
    fire_conf = st.slider("Fire Confidence", 0.3, 0.9, 0.40, 0.05)
    weapon_conf = st.slider("Weapon Confidence", 0.3, 0.9, 0.40, 0.05)
    use_cuda = st.checkbox("GPU Acceleration", value=torch.cuda.is_available())
    start_btn = st.button("🚀 Start Optimized Stream")
    
    st.markdown("---")
    st.markdown("#### Performance")
    fps_display = st.empty()
    
    st.markdown("---")
    st.markdown("#### Recent Detections")
    for label, folder in [("FALL", "fall"), ("FIRE", "fire"), ("WEAPON", "weapon")]:
        st.markdown(f'<span class="event-badge {folder}-badge">{label}</span>', unsafe_allow_html=True)
        folder_path = SNAPSHOT_DIR / folder
        if folder_path.exists():
            files = sorted(folder_path.glob("*.jpg"), reverse=True)[:2]
            for img in files:
                st.image(str(img), width=140)
        st.markdown("---")

frame_display = col1.empty()

if 'processes' not in st.session_state:
    st.session_state['processes'] = None
    st.session_state['stop_event'] = None

if start_btn:
    # Stop existing
    if st.session_state['stop_event']:
        st.session_state['stop_event'].set()
        time.sleep(0.5)
    
    # Initialize
    init_directories()
    
    # Multiprocessing setup
    mp.set_start_method('spawn', force=True)
    frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    result_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    stop_event = Event()
    fps_counter = Value('i', 0)
    
    device_str = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
    
    # Start processes
    capture_proc = Process(target=optimized_capture_process, 
                          args=(hls_url, frame_queue, stop_event, fps_counter), daemon=True)
    inference_proc = Process(target=optimized_inference_process,
                            args=(frame_queue, result_queue, stop_event, device_str, 
                                  conf_thres, fire_conf, weapon_conf), daemon=True)
    
    capture_proc.start()
    inference_proc.start()
    
    st.session_state['processes'] = (capture_proc, inference_proc, frame_queue, result_queue, fps_counter)
    st.session_state['stop_event'] = stop_event
    st.success("✅ Optimized pipeline started!")

# Display loop
try:
    if st.session_state['processes']:
        _, _, _, result_queue, fps_counter = st.session_state['processes']
        
        while True:
            try:
                frame = result_queue.get(timeout=1.0)
                frame_display.image(frame, channels="BGR", use_column_width=True)
                fps_display.metric("FPS", fps_counter.value)
            except:
                time.sleep(0.01)
                continue
    else:
        frame_display.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")
        
except KeyboardInterrupt:
    if st.session_state['stop_event']:
        st.session_state['stop_event'].set()
