import cv2
import time
import threading
from datetime import datetime


class LiveScanner:
    def __init__(
        self,
        camera_backend,
        camera_source,
        camera_width,
        camera_height,
        ai,
        db,
        is_already_marked_today,
        log_attendance,
        split_user_key,
    ):
        self.camera_backend = camera_backend
        self.camera_source = camera_source
        self.camera_width = camera_width
        self.camera_height = camera_height

        self.ai = ai
        self.db = db
        self.is_already_marked_today = is_already_marked_today
        self.log_attendance = log_attendance
        self.split_user_key = split_user_key

        self.running = False
        self.thread = None
        self.cap = None
        self.picam2 = None

        self.latest_results = {
            "success": True,
            "message": "Live scanner not started",
            "faces_detected": 0,
            "results": [],
            "last_updated": "",
        }

    def start(self):
        if self.running:
            return {
                "success": True,
                "message": "Live scanner already running",
            }

        self.running = True
        self.thread = threading.Thread(target=self.scan_loop, daemon=True)
        self.thread.start()

        return {
            "success": True,
            "message": "Live scanner started",
        }

    def stop(self):
        self.running = False
        self.release_camera()

        return {
            "success": True,
            "message": "Live scanner stopped",
        }

    def get_results(self):
        return self.latest_results

    def open_camera(self):
        if self.camera_backend == "picamera2":
            from picamera2 import Picamera2

            self.picam2 = Picamera2()
            self.picam2.configure(
                self.picam2.create_preview_configuration(
                    main={
                        "size": (self.camera_width, self.camera_height),
                        "format": "RGB888",
                    }
                )
            )
            self.picam2.start()
            time.sleep(2)
            return True

        self.cap = cv2.VideoCapture(self.camera_source)

        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        return True

    def read_frame(self):
        if self.camera_backend == "picamera2":
            frame_rgb = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr

        ret, frame = self.cap.read()
        return ret, frame

    def release_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            if self.picam2 is not None:
                self.picam2.stop()
                self.picam2.close()
                self.picam2 = None
        except Exception:
            pass

    def scan_loop(self):
        opened = self.open_camera()

        if not opened:
            self.latest_results = {
                "success": False,
                "message": "Could not open camera",
                "faces_detected": 0,
                "results": [],
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.running = False
            return

        while self.running:
            ret, frame = self.read_frame()

            if not ret or frame is None:
                self.latest_results = {
                    "success": False,
                    "message": "Failed to read camera frame",
                    "faces_detected": 0,
                    "results": [],
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                time.sleep(1)
                continue

            try:
                faces = self.ai.detect(frame)
                results = []

                for face in faces:
                    feature = self.ai.extract_feature(frame, face)
                    match_name, score = self.db.match_face(feature)

                    if match_name == "Unknown":
                        results.append({
                            "name": "Unknown",
                            "user_id": "",
                            "group": "",
                            "status": "unknown",
                            "message": "Unknown face",
                            "score": float(score),
                        })
                        continue

                    user = self.split_user_key(match_name)

                    if self.is_already_marked_today(user["name"]):
                        results.append({
                            "name": user["name"],
                            "user_id": user["user_id"],
                            "group": user["group"],
                            "status": "already_marked",
                            "message": "Attendance already marked today",
                            "score": float(score),
                        })
                    else:
                        self.log_attendance(
                            user["name"],
                            user["user_id"],
                            user["group"],
                        )

                        results.append({
                            "name": user["name"],
                            "user_id": user["user_id"],
                            "group": user["group"],
                            "status": "marked",
                            "message": "Attendance marked successfully",
                            "score": float(score),
                        })

                self.latest_results = {
                    "success": True,
                    "message": "Live scan running",
                    "faces_detected": len(faces),
                    "results": results,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

            except Exception as e:
                self.latest_results = {
                    "success": False,
                    "message": f"Scanner error: {e}",
                    "faces_detected": 0,
                    "results": [],
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

            time.sleep(2)

        self.release_camera()