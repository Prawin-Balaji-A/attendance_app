"""
main.py — FastAPI backend for AI Attendance System.

Pipeline (all MIT/Apache-2.0 — commercial safe):
  PiCamera2 (BGR888) → YuNet (MIT) → SORT (MIT) → SFace (MIT) → SQLite

Architecture:
  Thread 1 — Camera capture: always runs at 25 FPS, never blocked by AI
  Thread 2 — AI pipeline:    runs independently, updates annotations async
  Main      — FastAPI:       serves MJPEG stream + REST endpoints
"""

import os
import io
import time
import asyncio
import threading
import numpy as np
import cv2
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database
from pipeline.engine import AttendanceEngine

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Global State ────────────────────────────────────────────────────────────
class AppState:
    camera_mode   = 'normal'
    is_scanning   = True

    # Written by camera thread — raw frame (BGR, correct colors)
    current_frame = None

    # Written by AI thread — annotated frame for MJPEG stream
    # Falls back to current_frame when AI hasn't run yet
    annotated_frame = None

    frame_lock     = threading.Lock()
    annotated_lock = threading.Lock()

    live_results = {
        'faces_detected': 0,
        'known_count':    0,
        'unknown_count':  0,
        'camera_mode':    'normal',
        'last_updated':   '',
        'results':        [],
    }

state  = AppState()
engine = None


# ── Placeholder frame ────────────────────────────────────────────────────────
def _make_placeholder(msg="Camera initializing..."):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (30, 20, 50)
    cv2.putText(img, msg,
                (int(img.shape[1]/2) - 200, img.shape[0]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
    cv2.putText(img, datetime.now().strftime("%H:%M:%S"),
                (int(img.shape[1]/2) - 60, img.shape[0]//2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    return img


# ── Camera open ──────────────────────────────────────────────────────────────
def _open_camera():
    """Returns (type, cam, rotation) or (None, None, 0)."""

    # --- PiCamera2 first ---
    try:
        from picamera2 import Picamera2
        cam = Picamera2()

        # libcamera automatically applies orientation from the tuning file,
        # so we do not need to manually rotate the frame.
        print("[Camera] IMX708 detected — using native orientation")

        # Use BGR888 directly so OpenCV receives native BGR (fixes blue tint)
        config = cam.create_preview_configuration(
            main={"size": (1280, 720), "format": "BGR888"},
        )
        cam.configure(config)
        cam.start()
        time.sleep(1.5)

        test = cam.capture_array()
        if test is not None and len(test.shape) == 3:
            print("[Camera] PiCamera2 ready at 1280x720 BGR888")
            return ("picamera2", cam)

        cam.stop(); cam.close()
    except Exception as e:
        print(f"[Camera] PiCamera2 failed: {e}")

    # --- OpenCV fallback ---
    for idx in range(4):
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS,          30)
                    print(f"[Camera] OpenCV camera index {idx} opened")
                    return ("opencv", cap)
            cap.release()
        except Exception as e:
            print(f"[Camera] OpenCV index {idx} failed: {e}")

    print("[Camera] No camera found — using placeholder")
    return (None, None)


def _read_frame(cam_type, cam):
    """Read one BGR frame natively."""
    frame = None

    if cam_type == "picamera2":
        try:
            # capture_array is returning RGB despite BGR config. Swap it for OpenCV!
            rgb = cam.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[Camera] capture error: {e}")
            return None
    elif cam_type == "opencv":
        try:
            ret, f = cam.read()
            frame = f if ret else None
        except Exception:
            return None

    if frame is None:
        return None

    # Ensure 3 channels
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]

    # Increase brightness and contrast to improve lighting
    # alpha = contrast (1.0-3.0), beta = brightness (0-100)
    frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=60)

    return frame


# ── Thread 1: Camera capture (ALWAYS fast, never blocked by AI) ───────────────
def camera_capture_thread():
    """Captures frames as fast as possible. Never runs AI."""
    # Set placeholder immediately
    with state.frame_lock:
        state.current_frame = _make_placeholder("Camera initializing...")

    cam_type, cam = _open_camera()

    if cam is None:
        while True:
            ph = _make_placeholder("No camera detected")
            with state.frame_lock:
                state.current_frame = ph
            time.sleep(0.5)
        return

    fail_count = 0
    while True:
        t0 = time.time()
        frame = _read_frame(cam_type, cam)

        if frame is None:
            fail_count += 1
            if fail_count > 50:
                print("[Camera] Too many failures, reinitializing...")
                fail_count = 0
                try:
                    if cam_type == "opencv":   cam.release()
                    elif cam_type == "picamera2": cam.stop()
                except Exception: pass
                time.sleep(2)
                cam_type, cam = _open_camera()
                if cam is None:
                    with state.frame_lock:
                        state.current_frame = _make_placeholder("Camera lost")
            time.sleep(0.05)
            continue

        fail_count = 0
        with state.frame_lock:
            state.current_frame = frame

        # Throttle to 30 FPS max
        elapsed = time.time() - t0
        time.sleep(max(0.0, 0.033 - elapsed))


# ── Thread 2: AI processing (runs independently, won't block stream) ──────────
def ai_processing_thread():
    """Reads current_frame, runs pipeline, updates annotated_frame."""
    global engine

    print("[AI Thread] Waiting for engine...")
    while engine is None:
        time.sleep(0.5)

    print("[AI Thread] Running.")

    while True:
        t0 = time.time()

        with state.frame_lock:
            frame = state.current_frame.copy() if state.current_frame is not None else None

        if frame is None or not state.is_scanning:
            # No frame or scanning paused — just pass through raw frame
            if frame is not None:
                with state.annotated_lock:
                    state.annotated_frame = frame
            time.sleep(0.1)
            continue

        try:
            db_enc   = database.get_encodings()
            users_db = database.get_users()

            annotated, results = engine.process_frame(
                frame,
                db_enc,
                users_db,
                log_attendance_fn  = database.log_attendance,
                already_logged_fn  = database.already_logged_today,
            )

            known   = sum(1 for r in results if r["known"])
            unknown = sum(1 for r in results if not r["known"])

            state.live_results = {
                'faces_detected': len(results),
                'known_count':    known,
                'unknown_count':  unknown,
                'camera_mode':    state.camera_mode,
                'last_updated':   datetime.now().strftime("%H:%M:%S"),
                'results':        results,
            }

            with state.annotated_lock:
                state.annotated_frame = annotated

        except Exception as e:
            print(f"[AI Thread] Error: {e}")
            import traceback; traceback.print_exc()
            with state.annotated_lock:
                state.annotated_frame = frame

        # Run at ~8 FPS (AI is expensive)
        elapsed = time.time() - t0
        time.sleep(max(0.0, 0.12 - elapsed))


# ── MJPEG stream ─────────────────────────────────────────────────────────────
def _mjpeg_generator():
    """
    Always yields frames at ~25 FPS.
    Shows annotated frame when AI is ready, else raw camera frame.
    Full resolution — no cropping.
    """
    while True:
        t0 = time.time()

        # Prefer annotated frame, fall back to raw
        with state.annotated_lock:
            frame = state.annotated_frame

        if frame is None:
            with state.frame_lock:
                frame = state.current_frame

        if frame is None:
            frame = _make_placeholder()

        if state.camera_mode == 'backlight':
            frame = _enhance_backlight(frame)

        ret, buf = cv2.imencode('.jpg', frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + buf.tobytes()
                + b'\r\n'
            )

        elapsed = time.time() - t0
        time.sleep(max(0.0, 0.04 - elapsed))  # 25 FPS cap


def _enhance_backlight(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ── FastAPI lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("[Main] Initializing database...")
    database.init_db()

    print("[Main] Loading AI pipeline...")
    try:
        engine = AttendanceEngine(MODELS_DIR)
        
        # Train ML classifier with existing database on startup
        encodings = database.get_encodings()
        if encodings:
            engine.recognizer.train_classifier(encodings)
            
        print("[Main] AI pipeline ready.")
    except Exception as e:
        print(f"[Main] WARNING: Engine load failed: {e}")
        engine = None

    # Thread 1: camera capture (fast, never blocked)
    t_cam = threading.Thread(target=camera_capture_thread,
                             daemon=True, name="CameraCapture")
    t_cam.start()

    # Thread 2: AI processing (async, won't block stream)
    t_ai = threading.Thread(target=ai_processing_thread,
                            daemon=True, name="AIProcessing")
    t_ai.start()

    print("[Main] Server ready at http://0.0.0.0:8000")
    yield
    print("[Main] Shutting down.")
    if engine:
        engine.close()


app = FastAPI(title="Attendance Backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                  allow_methods=["*"], allow_headers=["*"],
                  allow_credentials=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "time":         datetime.now().isoformat(),
        "camera_ready": state.current_frame is not None,
        "engine_ready": engine is not None,
    }


@app.get("/debug-face-feed")
async def debug_face_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/camera-frame")
async def camera_frame():
    with state.annotated_lock:
        frame = state.annotated_frame
    if frame is None:
        with state.frame_lock:
            frame = state.current_frame
    if frame is None:
        frame = _make_placeholder()
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        return JSONResponse({"success": False, "message": "Encode failed"}, 500)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


def _extract_augmented_embeddings(engine, frame):
    """
    Data Augmentation: Multiply a single frame into multiple variations
    (brightness, flip, slight rotations) to massively improve SVM/KNN training.
    """
    embs = []
    
    # 1. Original
    emb = engine.extract_embedding_from_image(frame)
    if emb is not None: embs.append(emb.tolist())
    else: return []  # If no face in original, skip variations
    
    # 2. Brighten
    bright = cv2.convertScaleAbs(frame, alpha=1.0, beta=30)
    emb = engine.extract_embedding_from_image(bright)
    if emb is not None: embs.append(emb.tolist())
    
    # 3. Darken
    dark = cv2.convertScaleAbs(frame, alpha=1.0, beta=-30)
    emb = engine.extract_embedding_from_image(dark)
    if emb is not None: embs.append(emb.tolist())
    
    # 4. Horizontal Flip (perfect for generalizing side profiles)
    flipped = cv2.flip(frame, 1)
    emb = engine.extract_embedding_from_image(flipped)
    if emb is not None: embs.append(emb.tolist())
    
    # 5. Rotate +7 degrees
    h, w = frame.shape[:2]
    M1 = cv2.getRotationMatrix2D((w/2, h/2), 7, 1.0)
    rot1 = cv2.warpAffine(frame, M1, (w, h))
    emb = engine.extract_embedding_from_image(rot1)
    if emb is not None: embs.append(emb.tolist())
    
    # 6. Rotate -7 degrees
    M2 = cv2.getRotationMatrix2D((w/2, h/2), -7, 1.0)
    rot2 = cv2.warpAffine(frame, M2, (w, h))
    emb = engine.extract_embedding_from_image(rot2)
    if emb is not None: embs.append(emb.tolist())
    
    return embs


@app.post("/register-image")
async def register_image(
    name: str = Form(...), user_id: str = Form(...), group: str = Form(...),
    image: UploadFile = File(...),
):
    if engine is None:
        return {"success": False, "message": "AI engine not ready"}
    raw = await image.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"success": False, "message": "Invalid image file"}
    embs = _extract_augmented_embeddings(engine, img)
    if not embs:
        return {"success": False, "message": "No face detected. Ensure face is clearly visible."}
    encodings = database.get_encodings()
    encodings.setdefault(user_id, []).extend(embs)
    database.save_encodings(encodings)
    database.add_user(user_id.strip(), name.strip(), group.strip())
    
    # Trigger model retrain
    engine.recognizer.train_classifier(encodings)
    
    return {"success": True,
            "message": f"{len(embs)} face augmented angles saved for {name}."}


@app.post("/register-video")
async def register_video(
    name: str = Form(...), user_id: str = Form(...), group: str = Form(...),
    video: UploadFile = File(...),
):
    if engine is None:
        return {"success": False, "message": "AI engine not ready"}
    import tempfile
    raw = await video.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(raw); tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    sample_every = max(1, int(fps * 0.5))
    collected, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % sample_every == 0:
            embs = _extract_augmented_embeddings(engine, frame)
            collected.extend(embs)
        idx += 1
    cap.release(); os.unlink(tmp_path)
    if not collected:
        return {"success": False, "message": "No face detected in video."}
    encodings = database.get_encodings()
    encodings.setdefault(user_id, []).extend(collected)
    database.save_encodings(encodings)
    database.add_user(user_id.strip(), name.strip(), group.strip())
    
    # Trigger model retrain
    engine.recognizer.train_classifier(encodings)
    
    return {"success": True,
            "message": f"Registered {name} with {len(collected)} augmented face angles."}


@app.post("/register-live")
async def register_live(
    name: str = Form(...), user_id: str = Form(...), group: str = Form(...),
):
    if engine is None:
        return {"success": False, "message": "AI engine not ready"}
        
    start = time.time()
    raw_frames = []
    last_cap = 0.0
    
    # 1. Capture Phase: 10 seconds, no AI processing to prevent lag
    while time.time() - start < 10.0:
        now = time.time()
        # Grab a frame every 0.20s (up to 50 frames total)
        if now - last_cap >= 0.20:
            with state.frame_lock:
                frame = state.current_frame.copy() if state.current_frame is not None else None
            if frame is not None:
                raw_frames.append(frame)
            last_cap = now
        await asyncio.sleep(0.05)
        
    if not raw_frames:
        return {"success": False,
                "message": "No frames captured. Camera might be disconnected."}

    # 2. Processing Phase: Extract embeddings from the 50 distinct real frames
    collected = []
    for frame in raw_frames:
        emb = engine.extract_embedding_from_image(frame)
        if emb is not None:
            collected.append(emb.tolist())
        # Yield to let the live stream update
        await asyncio.sleep(0.01)

    if not collected:
        return {"success": False,
                "message": "No face detected in any frame. Stand in front of camera and try again."}
                
    encodings = database.get_encodings()
    encodings.setdefault(user_id, []).extend(collected)
    database.save_encodings(encodings)
    database.add_user(user_id.strip(), name.strip(), group.strip())
    
    # Trigger model retrain
    engine.recognizer.train_classifier(encodings)
    
    return {"success": True,
            "message": f"SUCCESS! {name} registered with {len(collected)} unique real face profiles."}


@app.get("/users")
async def get_users():
    return {"success": True, "users": database.get_users()}


@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    ok = database.delete_user(user_id)
    if ok:
        return {"success": True, "message": "User deleted."}
    return JSONResponse({"success": False, "message": "User not found."}, 404)


@app.get("/groups")
async def get_groups():
    return {"success": True, "groups": database.get_groups()}


@app.get("/groups/{group_name}")
async def get_group_details(group_name: str):
    return {"success": True, "group": database.get_group_details(group_name)}


@app.get("/attendance")
async def get_attendance():
    return {"success": True, "attendance": database.get_attendance()}


@app.post("/start-live-scan")
async def start_live_scan():
    state.is_scanning = True
    return {"success": True, "message": "Live scan running."}


@app.post("/stop-live-scan")
async def stop_live_scan():
    state.is_scanning = False
    return {"success": True, "message": "Live scan paused."}


class CameraMode(BaseModel):
    mode: str


@app.post("/set-camera-mode")
async def set_camera_mode(body: CameraMode):
    state.camera_mode = body.mode
    state.live_results['camera_mode'] = body.mode
    return {"success": True, "message": f"Mode set to {body.mode}."}


@app.get("/live-results")
async def get_live_results():
    return {"success": True, **state.live_results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
