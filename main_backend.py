import os
import cv2
import csv
import time
import json
import pickle
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel

from picamera2 import Picamera2


app = FastAPI(title="Face Attendance Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

USERS_JSON = DATA_DIR / "users.json"
ATTENDANCE_FILE = DATA_DIR / "attendance.csv"
DB_FILE = DATA_DIR / "face_embeddings.pkl"
OLD_DB_FILE = BASE_DIR / "database.pkl"

MODELS_DIR = Path("/home/admin/Desktop/ai_cctv_code/models")
DETECTOR_MODEL = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
STREAM_WIDTH = 960
STREAM_HEIGHT = 540

DETECTION_THRESHOLD = 0.45
NMS_THRESHOLD = 0.25
TOP_K = 10000

COSINE_THRESHOLD = 0.55
DUPLICATE_FACE_THRESHOLD = 0.58

MAX_EMBEDDINGS_PER_USER = 10
LIVE_SCAN_INTERVAL = 1.0

camera_lock = threading.Lock()
ai_lock = threading.Lock()
db_lock = threading.Lock()
latest_lock = threading.Lock()
attendance_lock = threading.Lock()

camera = None
live_thread = None
live_running = True
current_camera_mode = "normal"

latest_results = {
    "success": True,
    "message": "Auto live scan starting",
    "faces_detected": 0,
    "known_count": 0,
    "unknown_count": 0,
    "camera_mode": current_camera_mode,
    "last_updated": "",
    "results": [],
}


class CameraModeRequest(BaseModel):
    mode: str


CAMERA_MODES = {
    "normal": {
        "AwbMode": 0,
        "ColourGains": (1.6, 1.4),
        "Sharpness": 2.0,
        "Contrast": 1.1,
        "Brightness": 0.0,
        "ExposureValue": 0.0,
    },
    "sunlight": {
        "AwbMode": 0,
        "ColourGains": (1.5, 1.3),
        "Sharpness": 1.8,
        "Contrast": 0.8,
        "Brightness": 0.2,
        "ExposureValue": -0.3,
    },
    "shade": {
        "AwbMode": 0,
        "ColourGains": (1.7, 1.5),
        "Sharpness": 2.0,
        "Contrast": 1.0,
        "Brightness": 0.3,
        "ExposureValue": 0.5,
    },
}


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def init_files():
    if not USERS_JSON.exists():
        with open(USERS_JSON, "w") as f:
            json.dump([], f, indent=4)

    if not ATTENDANCE_FILE.exists():
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "UserID", "Group", "Date", "Time", "Timestamp"])

    if not DB_FILE.exists():
        with open(DB_FILE, "wb") as f:
            pickle.dump({}, f)


init_files()


def load_users():
    try:
        with open(USERS_JSON, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_users(users):
    with open(USERS_JSON, "w") as f:
        json.dump(users, f, indent=4)


def parse_old_user_key(key):
    parts = str(key).split("|")
    return {
        "name": parts[0] if len(parts) > 0 else str(key),
        "user_id": parts[1] if len(parts) > 1 else str(key),
        "group": parts[2] if len(parts) > 2 else "Default",
    }


def load_db():
    db = {}

    if DB_FILE.exists():
        try:
            with open(DB_FILE, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    db.update(data)
        except Exception:
            pass

    if OLD_DB_FILE.exists():
        try:
            with open(OLD_DB_FILE, "rb") as f:
                old_data = pickle.load(f)

            if isinstance(old_data, dict):
                for key, value in old_data.items():
                    key_str = str(key)

                    if "|" in key_str:
                        info = parse_old_user_key(key_str)
                        uid = info["user_id"]
                        name = info["name"]
                        group = info["group"]
                    else:
                        uid = key_str
                        name = key_str
                        group = "Default"

                    if uid not in db:
                        features = value if isinstance(value, list) else [value]
                        db[uid] = {
                            "name": name,
                            "group": group,
                            "embeddings": features,
                        }
        except Exception:
            pass

    return db


def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)


def sync_users_file_with_model():
    with db_lock:
        users = load_users()
        db = load_db()
        existing_ids = {str(u.get("user_id")) for u in users}

        for user_id, data in db.items():
            uid = str(user_id)

            if uid not in existing_ids:
                users.append({
                    "name": data.get("name", uid),
                    "user_id": uid,
                    "group": data.get("group", "Default"),
                    "created_at": "Already trained",
                })

        save_users(users)
        return users


def read_attendance():
    if not ATTENDANCE_FILE.exists():
        return []

    try:
        with open(ATTENDANCE_FILE, "r") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def mark_attendance(name, user_id, group):
    print("ATTENDANCE CHECK:", user_id)
    name = str(name).strip()
    user_id = str(user_id).strip()
    group = str(group).strip()

    if not user_id or name.lower() == "unknown":
        return {"marked": False, "message": "Unknown not marked"}

    today = datetime.now().strftime("%Y-%m-%d")

    with attendance_lock:
        rows = read_attendance()

        for row in rows:
            if (
                str(row.get("UserID")).strip() == user_id
                and str(row.get("Date")).strip() == today
            ):
                print("ALREADY MARKED:", user_id)
                return {"marked": False, "message": "Already marked today"}

        now = datetime.now()

        with open(ATTENDANCE_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                name,
                user_id,
                group,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                now.strftime("%Y-%m-%d %H:%M:%S"),
            ])

    return {"marked": True, "message": "Attendance marked"}


def get_camera_controls():
    mode = CAMERA_MODES.get(current_camera_mode, CAMERA_MODES["normal"])

    return {
        "FrameRate": 30.0,
        "AwbMode": mode["AwbMode"],
        "ColourGains": mode["ColourGains"],
        "Sharpness": mode["Sharpness"],
        "Contrast": mode["Contrast"],
        "Brightness": mode["Brightness"],
        "ExposureValue": mode["ExposureValue"],
        "NoiseReductionMode": 2,
        "AeEnable": True,
    }


def get_camera():
    global camera

    with camera_lock:
        if camera is None:
            camera = Picamera2()

            config = camera.create_video_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                    "format": "RGB888",
                },
                controls=get_camera_controls(),
                buffer_count=4,
            )

            camera.configure(config)
            camera.start()
            time.sleep(1.0)

        return camera


def apply_camera_mode(mode):
    global current_camera_mode, camera

    if mode not in CAMERA_MODES:
        return False

    current_camera_mode = mode

    with camera_lock:
        if camera is not None:
            camera.set_controls(get_camera_controls())

    return True


def capture_frame():
    cam = get_camera()

    with camera_lock:
        frame = cam.capture_array()

    if frame is None:
        return None

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


def frame_to_jpeg(frame, quality=60):
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )

    if not ok:
        return None

    return buffer.tobytes()


class FaceAI:
    def __init__(self):
        if not os.path.exists(DETECTOR_MODEL):
            raise FileNotFoundError(f"Detector model not found: {DETECTOR_MODEL}")

        if not os.path.exists(RECOGNIZER_MODEL):
            raise FileNotFoundError(f"Recognizer model not found: {RECOGNIZER_MODEL}")

        self.detector = cv2.FaceDetectorYN_create(
            DETECTOR_MODEL,
            "",
            (CAMERA_WIDTH, CAMERA_HEIGHT),
            DETECTION_THRESHOLD,
            NMS_THRESHOLD,
            TOP_K,
        )

        self.recognizer = cv2.FaceRecognizerSF_create(RECOGNIZER_MODEL, "")

    def update_input_size(self, width, height):
        self.detector.setInputSize((int(width), int(height)))

    def detect(self, frame):
        h, w = frame.shape[:2]

        with ai_lock:
            self.update_input_size(w, h)
            _, faces = self.detector.detect(frame)

        return faces if faces is not None else []

    def extract_feature(self, frame, face):
        with ai_lock:
            aligned_face = self.recognizer.alignCrop(frame, face)
            feature = self.recognizer.feature(aligned_face)

        return feature

    def match(self, feature1, feature2):
        with ai_lock:
            return self.recognizer.match(
                normalize_feature(feature1),
                normalize_feature(feature2),
                cv2.FaceRecognizerSF_FR_COSINE,
            )


face_ai = FaceAI()


def normalize_feature(feature):
    arr = np.array(feature).astype("float32")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    return arr


def recognize_feature(feature, duplicate_check=False):
    with db_lock:
        db = load_db()

    if not db:
        return {
            "known": False,
            "name": "Unknown",
            "user_id": "",
            "group": "",
            "score": 0.0,
        }

    threshold = DUPLICATE_FACE_THRESHOLD if duplicate_check else COSINE_THRESHOLD

    best_score = 0.0
    best_user_id = None

    for user_id, data in db.items():
        embeddings = data.get("embeddings", [])

        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        for saved_feature in embeddings:
            try:
                score = face_ai.match(feature, saved_feature)

                if float(score) > best_score:
                    best_score = float(score)
                    best_user_id = str(user_id)
            except Exception:
                continue

    if best_user_id is not None and best_score >= threshold:
        data = db[best_user_id]

        return {
            "known": True,
            "name": str(data.get("name", best_user_id)),
            "user_id": str(best_user_id),
            "group": str(data.get("group", "Default")),
            "score": round(float(best_score), 3),
        }

    return {
        "known": False,
        "name": "Unknown",
        "user_id": "",
        "group": "",
        "score": round(float(best_score), 3),
    }


def image_bytes_to_frame(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    return frame


def extract_feature_from_uploaded_image(image_bytes):
    frame = image_bytes_to_frame(image_bytes)

    if frame is None:
        return None, "Invalid image file"

    h, w = frame.shape[:2]

    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, (1280, int(h * scale)))

    faces = face_ai.detect(frame)

    if len(faces) == 0:
        return None, "No face detected. Use clear face image"

    biggest = max(faces, key=lambda f: float(f[2]) * float(f[3]))
    feature = face_ai.extract_feature(frame, biggest)

    return feature, "Face feature extracted"


def process_frame(frame, mark=True):
    faces = face_ai.detect(frame)
    results = []

    for face in faces:
        x, y, w, h = face[:4].astype(int)

        feature = face_ai.extract_feature(frame, face)
        rec = recognize_feature(feature)

        attendance_result = {
            "marked": False,
            "message": "Unknown face - not marked",
        }

        status = "unknown"

        if rec["known"] is True and rec["user_id"]:
            if mark:
                attendance_result = mark_attendance(
                    rec["name"],
                    rec["user_id"],
                    rec["group"],
                )
                status = "marked" if attendance_result["marked"] else "already_marked"
            else:
                status = "known"

        results.append({
            "name": rec["name"] if rec["known"] else "Unknown",
            "user_id": rec["user_id"] if rec["known"] else "",
            "group": rec["group"] if rec["known"] else "",
            "known": bool(rec["known"]),
            "status": status,
            "marked": bool(attendance_result["marked"]),
            "message": attendance_result["message"],
            "score": float(rec["score"]),
            "face_confidence": round(float(face[-1]), 3),
            "box": [int(x), int(y), int(w), int(h)],
        })

    return faces, results


def resize_for_stream(frame):
    return cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_AREA)


def draw_latest_boxes(frame):
    display_frame = frame.copy()

    with latest_lock:
        results = latest_results.get("results", [])

    scale_x = frame.shape[1] / float(CAMERA_WIDTH)
    scale_y = frame.shape[0] / float(CAMERA_HEIGHT)

    for result in results:
        x, y, w, h = result.get("box", [0, 0, 0, 0])

        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        if result.get("known") is True:
            color = (0, 255, 0)
            text = f"{result.get('name')} ({result.get('score')})"
        else:
            color = (0, 0, 255)
            text = "UNKNOWN"

        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
        cv2.rectangle(display_frame, (x, max(0, y - 30)), (x + w, y), color, -1)

        cv2.putText(
            display_frame,
            text,
            (x + 5, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    cv2.putText(
        display_frame,
        f"Faces: {len(results)} | Mode: {current_camera_mode}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
    )

    return display_frame


def mjpeg_generator(debug=False):
    while True:
        try:
            frame = capture_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            frame = resize_for_stream(frame)

            if debug:
                frame = draw_latest_boxes(frame)

            jpg = frame_to_jpeg(frame, quality=55)

            if jpg is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )

            time.sleep(0.04)

        except Exception as e:
            print("MJPEG error:", e)
            time.sleep(0.5)


def live_scan_loop():
    global latest_results

    while live_running:
        try:
            frame = capture_frame()

            if frame is None:
                time.sleep(1)
                continue

            _, results = process_frame(frame, mark=True)

            known_count = len([r for r in results if r["known"]])
            unknown_count = len([r for r in results if not r["known"]])

            new_latest = make_json_safe({
                "success": True,
                "message": "Auto live scan running",
                "faces_detected": len(results),
                "known_count": known_count,
                "unknown_count": unknown_count,
                "camera_mode": current_camera_mode,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
            })

            with latest_lock:
                latest_results = new_latest

            time.sleep(LIVE_SCAN_INTERVAL)

        except Exception as e:
            with latest_lock:
                latest_results = {
                    "success": False,
                    "message": str(e),
                    "faces_detected": 0,
                    "known_count": 0,
                    "unknown_count": 0,
                    "camera_mode": current_camera_mode,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "results": [],
                }

            time.sleep(1)


@app.on_event("startup")
def startup_event():
    global live_thread, live_running

    live_running = True

    if live_thread is None or not live_thread.is_alive():
        live_thread = threading.Thread(target=live_scan_loop, daemon=True)
        live_thread.start()


@app.get("/")
def home():
    return make_json_safe({
        "success": True,
        "message": "Face Attendance Backend Running",
        "auto_detection": True,
        "camera_mode": current_camera_mode,
        "registered_users": len(sync_users_file_with_model()),
    })


@app.get("/video-feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(debug=False),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/debug-face-feed")
def debug_face_feed():
    return StreamingResponse(
        mjpeg_generator(debug=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/camera-frame")
def camera_frame():
    frame = capture_frame()

    if frame is None:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Failed to capture frame"},
        )

    frame = resize_for_stream(frame)
    jpg = frame_to_jpeg(frame, quality=70)

    if jpg is None:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Failed to encode frame"},
        )

    return Response(content=jpg, media_type="image/jpeg")


@app.get("/test-face")
def test_face():
    frame = capture_frame()

    if frame is None:
        return {"success": False, "message": "Failed to capture frame"}

    faces, results = process_frame(frame, mark=False)

    return make_json_safe({
        "success": True,
        "faces_detected": int(len(faces)),
        "known_count": int(len([r for r in results if r["known"]])),
        "unknown_count": int(len([r for r in results if not r["known"]])),
        "camera_mode": current_camera_mode,
        "results": results,
    })


@app.post("/register-image")
async def register_user_from_image(
    name: str = Form(...),
    user_id: str = Form(...),
    group: str = Form(...),
    image: UploadFile = File(...),
):
    name = name.strip()
    user_id = user_id.strip()
    group = group.strip()

    if not name or not user_id or not group:
        return {"success": False, "message": "Name, User ID and Group required"}

    try:
        image_bytes = await image.read()

        if len(image_bytes) > 5 * 1024 * 1024:
            return {
                "success": False,
                "message": "Image too large. Upload image below 5MB",
            }

        feature, msg = extract_feature_from_uploaded_image(image_bytes)

        if feature is None:
            return {"success": False, "message": msg}

        duplicate = recognize_feature(feature, duplicate_check=True)

        with db_lock:
            db = load_db()
            users = load_users()

            user_exists = user_id in db

            if duplicate["known"] is True:
                duplicate_user_id = str(duplicate["user_id"])

                if duplicate_user_id != user_id:
                    return {
                        "success": False,
                        "message": f"This face is already registered as {duplicate['name']} ({duplicate_user_id})",
                    }

            if user_exists:
                embeddings = db[user_id].get("embeddings", [])

                if not isinstance(embeddings, list):
                    embeddings = [embeddings]

                if len(embeddings) >= MAX_EMBEDDINGS_PER_USER:
                    return {
                        "success": False,
                        "message": f"Maximum {MAX_EMBEDDINGS_PER_USER} face images already added for this user",
                    }

                embeddings.append(feature)

                db[user_id]["name"] = name
                db[user_id]["group"] = group
                db[user_id]["embeddings"] = embeddings

                save_db(db)

                for u in users:
                    if str(u.get("user_id")) == user_id:
                        u["name"] = name
                        u["group"] = group
                        break

                save_users(users)

                return {
                    "success": True,
                    "message": f"Extra face angle added for {name}. Total images: {len(embeddings)}",
                }

            db[user_id] = {
                "name": name,
                "group": group,
                "embeddings": [feature],
            }

            save_db(db)

            exists_in_users = any(str(u.get("user_id")) == user_id for u in users)

            if not exists_in_users:
                users.append({
                    "name": name,
                    "user_id": user_id,
                    "group": group,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

            save_users(users)

        return {
            "success": True,
            "message": f"{name} registered successfully",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Registration failed: {str(e)}",
        }


@app.post("/camera-mode")
def set_camera_mode(request: CameraModeRequest):
    mode = request.mode.strip().lower()

    if mode not in CAMERA_MODES:
        return {
            "success": False,
            "message": "Invalid mode. Use normal, sunlight, or shade",
            "available_modes": list(CAMERA_MODES.keys()),
        }

    ok = apply_camera_mode(mode)

    return {
        "success": ok,
        "message": f"Camera mode changed to {mode}",
        "camera_mode": current_camera_mode,
    }


@app.get("/camera-mode")
def get_camera_mode():
    return {
        "success": True,
        "camera_mode": current_camera_mode,
        "available_modes": list(CAMERA_MODES.keys()),
    }


@app.post("/start-live-scan")
def start_live_scan():
    return {"success": True, "message": "Auto live scan already running"}


@app.post("/stop-live-scan")
def stop_live_scan():
    return {
        "success": True,
        "message": "Auto detection enabled. Backend keeps scanning.",
    }


@app.get("/live-results")
def get_live_results():
    with latest_lock:
        return make_json_safe(latest_results)


@app.get("/users")
def get_users():
    users = sync_users_file_with_model()
    return make_json_safe({
        "success": True,
        "count": len(users),
        "users": users,
    })


@app.delete("/users/{user_id}")
def delete_user(user_id: str):

    user_id = str(user_id).strip()

    with db_lock:

        users = load_users()

        users = [
            u for u in users
            if str(u.get("user_id", "")).strip() != user_id
        ]

        save_users(users)

        db = load_db()

        if user_id in db:
            del db[user_id]

        save_db(db)

    return {
        "success": True,
        "message": f"User {user_id} deleted successfully",
    }


@app.get("/attendance")
def get_attendance():
    rows = read_attendance()
    rows.reverse()

    return make_json_safe({
        "success": True,
        "count": len(rows),
        "attendance": rows,
    })


@app.get("/groups")
def get_groups():
    users = sync_users_file_with_model()
    attendance = read_attendance()
    today = datetime.now().strftime("%Y-%m-%d")

    present_ids = {
        str(row.get("UserID"))
        for row in attendance
        if row.get("Date") == today and str(row.get("UserID")).strip() != ""
    }

    group_map = {}

    for user in users:
        group = str(user.get("group") or "Default")
        user_id = str(user.get("user_id"))

        if group not in group_map:
            group_map[group] = {
                "groupName": group,
                "totalMembers": 0,
                "presentCount": 0,
                "absentCount": 0,
                "presentUsers": [],
                "absentUsers": [],
            }

        group_map[group]["totalMembers"] += 1

        if user_id in present_ids:
            group_map[group]["presentCount"] += 1
            group_map[group]["presentUsers"].append(user)
        else:
            group_map[group]["absentUsers"].append(user)

    for group in group_map.values():
        group["absentCount"] = group["totalMembers"] - group["presentCount"]

    return make_json_safe({
        "success": True,
        "count": len(group_map),
        "groups": list(group_map.values()),
    })


@app.get("/groups/{group_name}")
def get_group_details(group_name: str):
    groups = get_groups()["groups"]

    for group in groups:
        if group["groupName"] == group_name:
            return make_json_safe({
                "success": True,
                "group": group,
                "groupName": group_name,
                "members": group["presentUsers"] + group["absentUsers"],
            })

    return {
        "success": False,
        "message": "Group not found",
        "groupName": group_name,
        "members": [],
    }
