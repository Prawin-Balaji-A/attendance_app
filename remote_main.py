import os
import cv2
import time
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database
from pipeline.engine import AttendanceEngine

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
engine = AttendanceEngine(MODELS_DIR)

# Global State
class AppState:
    camera_mode = 'normal'
    is_live_scanning = False
    current_frame = None
    live_results = {
        'faces_detected': 0,
        'known_count': 0,
        'unknown_count': 0,
        'too_far_count': 0,
        'camera_mode': 'normal',
        'last_updated': '',
        'results': []
    }

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    database.init_db()
    # Start camera loop
    asyncio.create_task(camera_loop())
    yield
    # Teardown
    app_state.is_live_scanning = False

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def camera_loop():
    print("Starting camera loop...")
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        if not cap.isOpened():
            print("ERROR: Could not open camera 0")
        await asyncio.sleep(2)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not read frame from camera")
                await asyncio.sleep(0.1)
                continue
                
            if app_state.is_live_scanning:
                try:
                    db_encodings = database.get_encodings()
                    processed_frame, recognized_users = engine.process_frame(frame, db_encodings)
                    
                    known_count = 0
                    unknown_count = 0
                    res = []
                    
                    for user_id, conf in recognized_users:
                        if user_id:
                            known_count += 1
                            users = database.get_users()
                            user_info = next((u for u in users if u['user_id'] == user_id), None)
                            if user_info:
                                database.log_attendance(user_id, user_info['name'], user_info['group'])
                                res.append({"name": user_info['name'], "status": "Logged"})
                        else:
                            unknown_count += 1
                            
                    app_state.live_results.update({
                        'faces_detected': known_count + unknown_count,
                        'known_count': known_count,
                        'unknown_count': unknown_count,
                        'last_updated': time.strftime("%H:%M:%S"),
                        'results': res
                    })
                    app_state.current_frame = processed_frame
                except Exception as e:
                    print("Error in engine processing:", e)
                    import traceback
                    traceback.print_exc()
            else:
                app_state.current_frame = frame
                
            await asyncio.sleep(0.03)
    except Exception as e:
        print("Camera loop crashed!", e)
        import traceback
        traceback.print_exc()

def generate_mjpeg():
    while True:
        if app_state.current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', app_state.current_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.get("/debug-face-feed")
async def debug_face_feed():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera-frame")
async def camera_frame():
    if app_state.current_frame is not None:
        ret, buffer = cv2.imencode('.jpg', app_state.current_frame)
        if ret:
            return StreamingResponse(iter([buffer.tobytes()]), media_type="image/jpeg")
    return {"success": False, "message": "Camera not ready"}

@app.post("/register-image")
async def register_image(name: str = Form(...), user_id: str = Form(...), group: str = Form(...), image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"success": False, "message": "Invalid image"}
        
    person_bboxes = engine.detect_persons(img)
    if len(person_bboxes) == 0:
        return {"success": False, "message": "No person found"}
        
    # Take first person
    x1, y1, x2, y2, _ = map(int, person_bboxes[0])
    person_crop = img[y1:y2, x1:x2]
    
    face_crop = engine.detect_face(person_crop)
    if face_crop is None:
        return {"success": False, "message": "No face found"}
        
    encoding = engine.extract_encoding(face_crop)
    
    # Save encoding
    encodings = database.get_encodings()
    if user_id not in encodings:
        encodings[user_id] = []
    encodings[user_id].append(encoding)
    database.save_encodings(encodings)
    database.add_user(user_id, name, group)
    
    return {"success": True, "message": "Registered successfully"}

@app.post("/register-video")
async def register_video(name: str = Form(...), user_id: str = Form(...), group: str = Form(...), video: UploadFile = File(...)):
    # Simple mock for video: process first frame as image
    return await register_image(name, user_id, group, video)

@app.post("/register-live")
async def register_live(name: str = Form(...), user_id: str = Form(...), group: str = Form(...)):
    # Reads from live camera for 5 seconds to capture multiple angles
    start_time = time.time()
    collected_encodings = []
    
    while time.time() - start_time < 5.0:
        if app_state.current_frame is not None:
            frame = app_state.current_frame.copy()
            person_bboxes = engine.detect_persons(frame)
            if len(person_bboxes) > 0:
                x1, y1, x2, y2, _ = map(int, person_bboxes[0])
                face_crop = engine.detect_face(frame[y1:y2, x1:x2])
                if face_crop is not None:
                    enc = engine.extract_encoding(face_crop)
                    if enc is not None:
                        collected_encodings.append(enc)
        await asyncio.sleep(0.5) # Process 2 FPS for registration
        
    if len(collected_encodings) > 0:
        encodings = database.get_encodings()
        if user_id not in encodings:
            encodings[user_id] = []
        encodings[user_id].extend(collected_encodings)
        database.save_encodings(encodings)
        database.add_user(user_id, name, group)
        return {"success": True, "message": f"Registered perfectly with {len(collected_encodings)} angles."}
        
    return {"success": False, "message": "Failed to capture face during live registration."}

@app.get("/users")
async def get_users():
    return {"success": True, "users": database.get_users()}

@app.get("/groups")
async def get_groups():
    return {"success": True, "groups": database.get_groups()}

@app.get("/groups/{group_name}")
async def get_group_details(group_name: str):
    return {"success": True, "users": database.get_users_by_group(group_name)}

@app.get("/attendance")
async def get_attendance():
    return {"success": True, "attendance": database.get_attendance()}

@app.post("/start-live-scan")
async def start_live_scan():
    app_state.is_live_scanning = True
    return {"success": True, "message": "Live scan started"}

class CameraMode(BaseModel):
    mode: str

@app.post("/set-camera-mode")
async def set_camera_mode(mode: CameraMode):
    app_state.camera_mode = mode.mode
    app_state.live_results['camera_mode'] = mode.mode
    return {"success": True, "message": "Mode updated"}

@app.post("/stop-live-scan")
async def stop_live_scan():
    app_state.is_live_scanning = False
    return {"success": True, "message": "Live scan stopped"}

@app.get("/live-results")
async def get_live_results():
    return {
        "success": True,
        **app_state.live_results
    }

@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    success = database.delete_user(user_id)
    if success:
        return {"success": True, "message": "User deleted"}
    return {"success": False, "message": "User not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
