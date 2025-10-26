import streamlit as st
import cv2
import threading
import time
import queue
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import boto3
import platform

# === CONFIG ===
MODEL_PATH_DEFAULT = "models/fall/fall.pt"  # Update path
IMGSZ = 640
CONF_THRES = 0.35
FIRE_CONF_THRES = 0.25  # Default fire detection threshold
WEAPON_CONF_THRES = 0.25  # Default weapon detection threshold
NMS_IOU = 0.45
QUEUE_SIZE = 4
DISPLAY_FPS = True
PERSON_MODEL_PATH = "models/base/yolov8n.pt"  # Update path
FIRE_MODEL_PATH = "models/fire/fire.pt"  # Update path
WEAPON_MODEL_PATH = "models/weapon/weapon.pt"  # Update path
SNAPSHOT_DIR = Path("snapshots")  # Add this config
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
S3_PREFIX = os.getenv("AWS_S3_PREFIX", "detections/")
# ==============

# Load .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Get Kinesis stream info from env
STREAM_NAME = os.getenv("STREAM_NAME")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

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

st.set_page_config(page_title="Multi-Event Detection Stream", layout="wide", page_icon="🛡️")
st.markdown("""
    <style>
        .event-badge {font-size: 1.2em; font-weight: bold; padding: 4px 10px; border-radius: 8px;}
        .fall-badge {background: #ffcccc; color: #b71c1c;}
        .fire-badge {background: #ffe0b2; color: #e65100;}
        .weapon-badge {background: #e1bee7; color: #4a148c;}
        .status-ok {color: #388e3c; font-weight: bold;}
        .status-alert {color: #d32f2f; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Real-time Fall, Fire & Weapon Detection")
st.markdown("Monitor live video for critical events. Snapshots and status update in real time.")

col1, col2 = st.columns([4, 1])

with col2:
    st.markdown("#### Stream Settings")
    # No RTSP input; show HLS URL
    try:
        hls_url = get_kinesis_hls_url()
        st.success("HLS URL obtained.")
        st.code(hls_url, language="text")
    except Exception as e:
        st.error(f"Failed to get HLS URL: {e}")
        hls_url = None
    model_path = st.text_input("YOLO weights", MODEL_PATH_DEFAULT)
    imgsz = st.number_input("Image size (px)", min_value=224, max_value=1280, value=IMGSZ, step=32)
    conf_thres = st.slider("Fall Detection Threshold", 0.05, 0.99, CONF_THRES, 0.01)
    fire_conf = st.slider("Fire Detection Threshold", 0.05, 0.99, FIRE_CONF_THRES, 0.01)
    weapon_conf = st.slider("Weapon Detection Threshold", 0.05, 0.99, WEAPON_CONF_THRES, 0.01)
    use_cuda = st.checkbox("Use CUDA (if available)", value=torch.cuda.is_available())
    start_stop = st.button("▶️ Start/Restart Stream")
    st.markdown("---")

    st.markdown("#### Recent Event Snapshots")
    snapshot_dirs = {
        "FALL": ("snapshots/fall", "fall-badge"),
        "FIRE": ("snapshots/fire", "fire-badge"),
        "WEAPON": ("snapshots/weapon", "weapon-badge")
    }
    for label, (folder, badge_class) in snapshot_dirs.items():
        st.markdown(f'<span class="event-badge {badge_class}">{label}</span>', unsafe_allow_html=True)
        folder_path = Path(folder)
        if folder_path.exists():
            files = sorted(folder_path.glob("*.jpg"), reverse=True)[:3]
            for img_file in files:
                st.image(str(img_file), caption=img_file.name, width=140)
        else:
            st.write("No snapshots yet.")
        st.markdown("---")

frame_placeholder = col1.empty()

with st.sidebar:
    st.markdown("### Detection Status")
    status_fall = "🟢 OK" if not Path("snapshots/fall").glob("*.jpg") else "🔴 Event"
    status_fire = "🟢 OK" if not Path("snapshots/fire").glob("*.jpg") else "🔴 Event"
    status_weapon = "🟢 OK" if not Path("snapshots/weapon").glob("*.jpg") else "🔴 Event"
    st.markdown(f"**Fall:** <span class='status-ok'>{status_fall}</span>", unsafe_allow_html=True)
    st.markdown(f"**Fire:** <span class='status-ok'>{status_fire}</span>", unsafe_allow_html=True)
    st.markdown(f"**Weapon:** <span class='status-ok'>{status_weapon}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### How it works")
    st.info("Start the stream. When an event is detected, a snapshot is saved and shown in the sidebar.")

log = st.sidebar.empty()

# Sidebar status
with st.sidebar:
    st.markdown("### Detection Status")
    st.markdown("- **Fall**: Saves to `snapshots/fall/`")
    st.markdown("- **Fire**: Saves to `snapshots/fire/`")
    st.markdown("- **Weapon**: Saves to `snapshots/weapon/`")
    st.markdown("---")
    st.markdown("### System Info")
    st.write(f"CUDA Available: `{torch.cuda.is_available()}`")
    st.write(f"Device Count: `{torch.cuda.device_count()}`")
    st.write(f"Current Device: `{torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}`")
    st.markdown("---")
    st.markdown("### Instructions")
    st.info("Start the stream to begin detection. Snapshots will appear here when events are detected.")

frame_q = queue.Queue(maxsize=QUEUE_SIZE)
out_q = queue.Queue(maxsize=QUEUE_SIZE)
stop_event = threading.Event()
model_lock = threading.Lock()

# Add dependency checks at the start
try:
    import tqdm
except ImportError:
    st.error("Missing required package 'tqdm'. Please install it using: pip install tqdm")
    st.stop()

try:
    from ultralytics import YOLO
except ImportError:
    st.error("Missing required package 'ultralytics'. Please install it using: pip install ultralytics")
    st.stop()

@st.cache_resource(ttl=3600)
def load_model(weights_path: str, device: str, imgsz: int):
    """Modified model loading with better error handling"""
    try:
        if not Path(weights_path).exists():
            st.error(f"Model file not found: {weights_path}")
            raise FileNotFoundError(f"Model file not found: {weights_path}")
            
        model = YOLO(weights_path)
        model.fuse()
        model.to(device)
        if device.startswith("cuda"):
            try:
                model.model.half()
            except Exception:
                pass
        return ("ultralytics", model)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        raise

def load_person_model(device: str):
    from ultralytics import YOLO
    model = YOLO(PERSON_MODEL_PATH)
    model.fuse()
    model.to(device)
    if device.startswith("cuda"):
        try:
            model.model.half()
        except Exception:
            pass
    return model

def load_fire_model(device: str):
    from ultralytics import YOLO
    model = YOLO(FIRE_MODEL_PATH)
    model.fuse()
    model.to(device)
    if device.startswith("cuda"):
        try:
            model.model.half()
        except Exception:
            pass
    return model

def load_weapon_model(device: str):
    from ultralytics import YOLO
    model = YOLO(WEAPON_MODEL_PATH)
    model.fuse()
    model.to(device)
    if device.startswith("cuda"):
        try:
            model.model.half()
        except Exception:
            pass
    return model

def video_capture_thread(hls_url, cap_q, stop_event):
    """Capture frames from HLS and put into cap_q."""
    if not hls_url:
        return
    cap = cv2.VideoCapture(hls_url)
    if not cap.isOpened():
        st.error("Failed to open HLS stream. Check Kinesis and stream status.")
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
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

def beep():
    """Play a beep sound (cross-platform)."""
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 300)  # frequency, duration(ms)
        else:
            import os
            os.system('play -nq -t alsa synth 0.3 sine 1000')
    except Exception:
        pass

def init_s3_folders():
    """Initialize S3 folders for detections"""
    try:
        s3_client = boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Create empty objects to represent folders
        folders = [
            f"{S3_PREFIX}fall/",
            f"{S3_PREFIX}fire/",
            f"{S3_PREFIX}weapon/"
        ]
        
        for folder in folders:
            s3_client.put_object(Bucket=S3_BUCKET, Key=folder)
            print(f"Initialized S3 folder: s3://{S3_BUCKET}/{folder}")
            
    except Exception as e:
        print(f"Failed to initialize S3 folders: {e}")

def upload_to_s3(file_path: Path, detection_type: str):
    """Upload file to S3 bucket with folder verification"""
    try:
        s3_client = boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Verify folder exists
        folder_key = f"{S3_PREFIX}{detection_type}/"
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=folder_key)
        except:
            # Create folder if doesn't exist
            s3_client.put_object(Bucket=S3_BUCKET, Key=folder_key)
            print(f"Created missing folder: s3://{S3_BUCKET}/{folder_key}")
        
        # Upload file
        s3_key = f"{folder_key}{file_path.name}"
        s3_client.upload_file(str(file_path), S3_BUCKET, s3_key)
        print(f"Uploaded {file_path.name} to s3://{S3_BUCKET}/{s3_key}")
        
    except Exception as e:
        print(f"Failed to upload to S3: {e}")

def save_fall_frame(frame):
    """Save frame locally and to S3"""
    fall_dir = SNAPSHOT_DIR / "fall"
    fall_dir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = fall_dir / f"fall_{ts}.jpg"
    cv2.imwrite(str(filename), frame)
    upload_to_s3(filename, "fall")

def save_fire_frame(frame):
    fire_dir = SNAPSHOT_DIR / "fire"
    fire_dir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = fire_dir / f"fire_{ts}.jpg"
    cv2.imwrite(str(filename), frame)
    upload_to_s3(filename, "fire")

def save_weapon_frame(frame):
    weapon_dir = SNAPSHOT_DIR / "weapon"
    weapon_dir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = weapon_dir / f"weapon_{ts}.jpg"
    cv2.imwrite(str(filename), frame)
    upload_to_s3(filename, "weapon")

def inference_thread(cap_q, out_q, stop_event, model_info, device, imgsz, conf_thres, fire_conf, weapon_conf, nms_iou):
    backend_type, fall_model = model_info
    person_model = load_person_model(device)
    fire_model = load_fire_model(device)
    weapon_model = load_weapon_model(device)
    last_time = time.time()
    frames_processed = 0
    while not stop_event.is_set():
        try:
            frame = cap_q.get(timeout=0.5)
        except queue.Empty:
            continue

        h, w = frame.shape[:2]
        annotated = frame.copy()
        fall_count = 0
        nofall_count = 0
        fall_detected_this_frame = False  # Track if any fall detected

        # --- Stage 1: Person Detection ---
        try:
            results = person_model.predict(source=[frame], imgsz=imgsz, conf=0.3, iou=0.5, device=device, half=(device.startswith("cuda")))
            boxes = results[0].boxes
            person_boxes = []
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    xyxy = box.xyxy[0].cpu().numpy()
                    person_boxes.append(xyxy)
        except Exception as e:
            cv2.putText(annotated, f"Person detection error: {e}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            annotated_frame = annotated
            try:
                if out_q.full():
                    _ = out_q.get_nowait()
                out_q.put_nowait(annotated_frame)
            except queue.Full:
                pass
            continue

        # --- Stage 2: Fall/No-Fall Classification ---
        for i, xyxy in enumerate(person_boxes):
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            crop_resized = cv2.resize(person_crop, (imgsz, imgsz))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            try:
                if backend_type == "ultralytics":
                    fall_results = fall_model.predict(source=[crop_rgb], imgsz=imgsz, conf=conf_thres, iou=nms_iou, device=device, half=(device.startswith("cuda")))
                    fall_boxes = fall_results[0].boxes
                    is_fall = False
                    fall_conf = 0
                    for fbox in fall_boxes:
                        fcls = int(fbox.cls[0])
                        fconf = float(fbox.conf[0])
                        if fcls == 0 and fconf >= conf_thres:
                            is_fall = True
                            fall_conf = fconf
                            break
                    label = "FALL" if is_fall else "NO-FALL"
                    color = (0,0,255) if is_fall else (0,255,0)
                    conf_disp = fall_conf if is_fall else 0.0
                    if is_fall:
                        fall_count += 1
                        fall_detected_this_frame = True
                    else:
                        nofall_count += 1
                else:
                    res = fall_model(crop_resized)
                    pred = res.pred[0]
                    is_fall = False
                    conf_disp = 0.0
                    for det in pred:
                        fcls = int(det[5])
                        fconf = float(det[4])
                        if fcls == 0 and fconf >= conf_thres:
                            is_fall = True
                            conf_disp = fconf
                            break
                    label = "FALL" if is_fall else "NO-FALL"
                    color = (0,0,255) if is_fall else (0,255,0)
                    if is_fall:
                        fall_count += 1
                        fall_detected_this_frame = True
                    else:
                        nofall_count += 1
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                cv2.putText(annotated, f"{label} {conf_disp:.2f}", (x1, max(y1-8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception as e:
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,255), 2)
                cv2.putText(annotated, f"Error", (x1, max(y1-8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        # --- Beep and save frame if fall detected ---
        if fall_detected_this_frame:
            beep()
            save_fall_frame(annotated)

        # --- Fire Detection ---
        fire_detected = False
        try:
            fire_results = fire_model.predict(source=[frame], imgsz=imgsz, conf=fire_conf, iou=0.5, device=device, half=(device.startswith("cuda")))
            for result in fire_results:
                for box in result.boxes:
                    cls_name = fire_model.names[int(box.cls[0])].lower()
                    conf = float(box.conf[0])
                    if cls_name in ["fire", "smoke"] and conf >= 0.25:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                        color = (0, 0, 255) if cls_name == "fire" else (128, 128, 128)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(annotated, f"{cls_name.upper()} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        fire_detected = True
            if fire_detected:
                save_fire_frame(annotated)
        except Exception as e:
            pass

        # --- Weapon Detection ---
        weapon_detected = False
        try:
            weapon_results = weapon_model.predict(source=[frame], imgsz=imgsz, conf=weapon_conf, iou=0.5, device=device, half=(device.startswith("cuda")))
            for result in weapon_results:
                for box in result.boxes:
                    cls_name = weapon_model.names[int(box.cls[0])].lower()
                    conf = float(box.conf[0])
                    if cls_name == "weapon" and conf >= 0.25:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(annotated, f"WEAPON {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        weapon_detected = True
            if weapon_detected:
                save_weapon_frame(annotated)
        except Exception as e:
            pass

        summary_text = f"FALL: {fall_count}  NO-FALL: {nofall_count}"
        cv2.putText(annotated, summary_text, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        annotated_frame = annotated

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

        try:
            if out_q.full():
                _ = out_q.get_nowait()
            out_q.put_nowait(annotated_frame)
        except queue.Full:
            pass

def start_pipeline(hls_url, model_path, imgsz, conf_thres, fire_conf, weapon_conf, use_cuda):
    """Modified pipeline start with better error handling"""
    try:
        stop_event.clear()
        cuda_available = torch.cuda.is_available()
        device = "cuda:0" if use_cuda and cuda_available else "cpu"
        st.sidebar.markdown(f"**Device in use:** `{device}`")
        
        # Verify model files exist
        model_files = [
            (MODEL_PATH_DEFAULT, "Fall Detection"),
            (PERSON_MODEL_PATH, "Person Detection"),
            (FIRE_MODEL_PATH, "Fire Detection"),
            (WEAPON_MODEL_PATH, "Weapon Detection")
        ]
        
        for path, name in model_files:
            if not Path(path).exists():
                st.error(f"Missing {name} model: {path}")
                return None
        
        with model_lock:
            model_info = load_model(model_path, device=device, imgsz=imgsz)
        
        cap_thread = threading.Thread(target=video_capture_thread, args=(hls_url, frame_q, stop_event), daemon=True)
        inf_thread = threading.Thread(
            target=inference_thread,
            args=(frame_q, out_q, stop_event, model_info, device, imgsz, conf_thres, fire_conf, weapon_conf, NMS_IOU),
            daemon=True
        )
        
        cap_thread.start()
        inf_thread.start()
        return cap_thread, inf_thread
    
    except Exception as e:
        st.error(f"Failed to start pipeline: {str(e)}")
        return None

# Initialize required directories
def init_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        Path("models/base"),
        Path("models/fall"),
        Path("models/fire"),
        Path("models/weapon"),
        SNAPSHOT_DIR / "fall",
        SNAPSHOT_DIR / "fire",
        SNAPSHOT_DIR / "weapon"
    ]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Initialized directory: {dir_path}")

# Create directories at startup
init_directories()
init_s3_folders()  # Add this line

if 'threads' not in st.session_state:
    st.session_state['threads'] = None

if start_stop:
    if st.session_state['threads'] is not None:
        stop_event.set()
        time.sleep(0.3)
        stop_event.clear()
    try:
        threads = start_pipeline(hls_url, model_path, imgsz, conf_thres, fire_conf, weapon_conf, use_cuda)
        st.session_state['threads'] = threads
        log.text(f"Started pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        st.error(f"Failed to start pipeline: {e}")
        st.session_state['threads'] = None

try:
    while True:
        if st.session_state.get('threads') is None:
            frame_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR", caption="No stream running")
            time.sleep(0.5)
            break
        try:
            frame = out_q.get(timeout=1.0)
            frame_placeholder.image(frame, channels="BGR")
        except queue.Empty:
            time.sleep(0.05)
            continue
except KeyboardInterrupt:
    pass
finally:
    stop_event.set()
    time.sleep(0.2)