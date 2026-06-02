import cv2
import numpy as np
import pickle
import time
import csv
import yaml
from pathlib import Path
from datetime import datetime

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.yaml"
MODELS_DIR = BASE_DIR.parent / "ai_cctv_code" / "models"

# Load config settings
try:
    with open(CONFIG_FILE, "r") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    print("Warning: config.yaml not found! Using default 640x480 resolution.")
    CONFIG = {"camera": {"backend": "picamera2", "source": 0, "width": 640, "height": 480}}

CAMERA_BACKEND = CONFIG["camera"].get("backend", "picamera2")
CAMERA_SOURCE = CONFIG["camera"].get("source", 0)
CAMERA_WIDTH = CONFIG["camera"].get("width", 1280)
CAMERA_HEIGHT = CONFIG["camera"].get("height", 720)
DB_FILE = BASE_DIR / "database.pkl"
ATTENDANCE_FILE = BASE_DIR / "attendance.csv"
RECORDINGS_DIR = BASE_DIR / "recordings"

# Paths to models
DETECTOR_MODEL = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")

# Thresholds and Cooldowns
COSINE_THRESHOLD = CONFIG.get("recognition", {}).get("strictness_threshold", 0.50)
ATTENDANCE_COOLDOWN = 10  # Seconds before logging the same person again
RECORDING_COOLDOWN = 3    # Seconds to keep recording after unknown disappears


# --- MODULES ---

class FaceDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.data = {}
        self.load()

    def load(self):
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                self.data = pickle.load(f)
    
    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.data, f)
            
    def add_face(self, name, feature):
        if name not in self.data:
            self.data[name] = []
        # Backwards compatibility for old databases
        if not isinstance(self.data[name], list):
            self.data[name] = [self.data[name]]
            
        self.data[name].append(feature)
        self.save()

    def match_face(self, feature_to_match):
        best_match = "Unknown"
        highest_score = 0.0
        
        for name, features in self.data.items():
            # Backwards compatibility
            if not isinstance(features, list):
                features = [features]
                
            for db_feature in features:
                score = cv2.FaceRecognizerSF_create(RECOGNIZER_MODEL, "").match(
                    feature_to_match, db_feature, cv2.FaceRecognizerSF_FR_COSINE
                )
                if score > highest_score and score >= COSINE_THRESHOLD:
                    highest_score = score
                    best_match = name
                
        return best_match, highest_score


class AttendanceLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.last_seen = {}
        
        # Create CSV with headers if it doesn't exist
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Name", "Status", "Proof Video"])

    def log_if_needed(self, name, video_name="N/A"):
        current_time = time.time()
        # Check cooldown
        if name in self.last_seen:
            if current_time - self.last_seen[name] < ATTENDANCE_COOLDOWN:
                return  # Skip logging, too soon
        
        # Log it
        self.last_seen[name] = current_time
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "Present" if name != "Unknown" else "Unauthorized Entry"
        
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp_str, name, status, video_name])
        print(f"[ATTENDANCE LOGGED] {name} at {timestamp_str}")


class UnknownRecorder:
    def __init__(self, rec_dir):
        self.rec_dir = rec_dir
        self.rec_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.last_unknown_time = 0
        self.current_filename = ""

    def process_frame(self, frame, has_unknown):
        current_time = time.time()
        
        if has_unknown:
            self.last_unknown_time = current_time
            if self.writer is None:
                # Start a new recording
                filename = self.rec_dir / f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                self.current_filename = filename.name
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # Lowered FPS to 8.0 so it matches the slower processing speed of your laptop
                self.writer = cv2.VideoWriter(str(filename), fourcc, 8.0, (w, h))
                print(f"[RECORDER] Started recording: {filename.name}")
        
        # Write frame if we are currently recording
        if self.writer is not None:
            self.writer.write(frame)
            
            # Stop recording if Unknown has been gone for X seconds
            if current_time - self.last_unknown_time > RECORDING_COOLDOWN:
                self.writer.release()
                self.writer = None
                self.current_filename = ""
                print("[RECORDER] Stopped recording. Saved to disk.")

    def release(self):
        if self.writer is not None:
            self.writer.release()


class FaceAI:
    def __init__(self):
        # NOTE: If moving to Hailo later, these lines can be swapped with HailoRT inferences
        self.detector = cv2.FaceDetectorYN_create(DETECTOR_MODEL, "", (320, 320), 0.8, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF_create(RECOGNIZER_MODEL, "")

    def update_input_size(self, width, height):
        self.detector.setInputSize((width, height))

    def detect(self, image):
        _, faces = self.detector.detect(image)
        return faces if faces is not None else []

    def extract_feature(self, image, face):
        aligned_face = self.recognizer.alignCrop(image, face)
        feature = self.recognizer.feature(aligned_face)
        return feature

class CameraStream:
    def __init__(self):
        self.use_picam = False
        self.cap = None
        self.picam = None

        print(f"Initializing camera hardware (Backend: {CAMERA_BACKEND})...")

        if CAMERA_BACKEND == "picamera2" and HAS_PICAMERA2:
            try:
                print(
                    f"Attempting to connect via native Picamera2 "
                    f"({CAMERA_WIDTH}x{CAMERA_HEIGHT})..."
                )

                self.picam = Picamera2()

                # UPDATED CAMERA CONFIGURATION
                config = self.picam.create_video_configuration(

                    main={
                        "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                        "format": "RGB888"
                    },

                    controls={

                        # FPS
                        "FrameRate": 30.0,

                        # Auto White Balance
                        "AwbMode": 0,

                        # MAIN FIX FOR BLUISH TINT
                        # (Red Gain, Blue Gain)
                        "ColourGains": (1.6, 1.4),

                        # Image enhancements
                        "Sharpness": 2.0,
                        "Contrast": 1.1,
                        "Brightness": 0.0,

                        # Better denoise quality
                        "NoiseReductionMode": 2,

                        # Auto exposure enabled
                        "AeEnable": True,
                    },

                    buffer_count=4,
                )

                self.picam.configure(config)

                self.picam.start()

                # Allow camera ISP to stabilize
                time.sleep(1.0)

                self.use_picam = True

                print("✅ Successfully connected to Raspberry Pi Camera!")

                return

            except Exception as e:
                print(f"Picamera2 init failed: {e}")

        print(f"Attempting to connect via OpenCV WebCam (Source: {CAMERA_SOURCE})...")
        self.cap = cv2.VideoCapture(CAMERA_SOURCE)
        
        # Windows robust fallback
        if not self.cap.isOpened() and isinstance(CAMERA_SOURCE, int):
            print("Standard OpenCV backend failed. Attempting DirectShow (Windows Fallback)...")
            self.cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
            
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            print("✅ Successfully opened USB WebCam/RTSP!")
        else:
            print("❌ ERROR: OpenCV could not open the camera. Ensure no other app (like Zoom/Teams) is using it!")

    def read(self):

        if self.use_picam:
            try:

                frame = self.picam.capture_array()

                if frame is None:
                    return False, None

                # Ensure uint8 format
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)

                # Proper RGB -> BGR conversion
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )

                return True, frame

            except Exception as e:
                print(f"Camera Read Error: {e}")
                return False, None

        else:

            if self.cap and self.cap.isOpened():
                return self.cap.read()

            return False, None

    def release(self):

        if self.use_picam and self.picam is not None:

            self.picam.stop()
            self.picam.close()

        elif self.cap is not None:

            self.cap.release()

def open_camera():
    stream = CameraStream()
    ret, _ = stream.read()
    if not ret:
        stream.release()
        return None
    return stream


# --- MAIN APPLICATION ---

def register_face():
    name = input("\nEnter the Name of the Authorized Person: ").strip()
    if not name:
        return
        
    print(f"\nOpening Camera... Look at the camera and press SPACE multiple times to capture different angles (Front, Left, Right) for '{name}'.")
    print("Press 'Q' when you are totally finished.")
    
    cap = open_camera()
    if cap is None:
        print("\n❌ CRITICAL ERROR: Could not open the camera!")
        print("Try running this command in your Pi terminal instead:")
        print("libcamerify python attendance_app.py")
        return
        
    ai = FaceAI()
    db = FaceDatabase(DB_FILE)
    
    ret, frame = cap.read()
    if not ret:
        print("\n❌ CRITICAL ERROR: Frame dropped immediately after opening!")
        cap.release()
        return
        
    ai.update_input_size(frame.shape[1], frame.shape[0])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        faces = ai.detect(frame)
        
        if len(faces) > 0:
            # Draw a yellow box showing it found a face
            box = list(map(int, faces[0][:4]))
            cv2.rectangle(display_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 255), 2)
            cv2.putText(display_frame, "Face Detected - Press SPACE to capture angle", (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "No face detected...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Register Face", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # SPACEBAR
            if len(faces) > 0:
                feature = ai.extract_feature(frame, faces[0])
                db.add_face(name, feature)
                print(f"✅ Successfully captured an angle for {name}!")
                # Give brief visual feedback
                cv2.putText(display_frame, "CAPTURED!", (box[0], box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.imshow("Register Face", display_frame)
                cv2.waitKey(300)
            else:
                print("Cannot capture. No face detected!")
        elif key == ord('q'):
            print(f"Finished registration for {name}.")
            break

    cap.release()
    cv2.destroyAllWindows()


def monitor():
    print("\nStarting Monitoring Mode... Press 'Q' to exit.")
    
    cap = open_camera()
    if cap is None:
        print("\n❌ CRITICAL ERROR: Could not open the camera!")
        print("Try running this command in your Pi terminal instead:")
        print("libcamerify python attendance_app.py")
        return
        
    ai = FaceAI()
    db = FaceDatabase(DB_FILE)
    logger = AttendanceLogger(ATTENDANCE_FILE)
    recorder = UnknownRecorder(RECORDINGS_DIR)
    
    ret, frame = cap.read()
    if not ret:
        print("\n❌ CRITICAL ERROR: Frame dropped immediately after opening!")
        cap.release()
        return
        
    ai.update_input_size(frame.shape[1], frame.shape[0])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        faces = ai.detect(frame)
        has_unknown_in_frame = False
        authorized_names_in_frame = []
        
        for face in faces:
            box = list(map(int, face[:4]))
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            # Extract features and match
            feature = ai.extract_feature(frame, face)
            match_name, score = db.match_face(feature)
            
            if match_name != "Unknown":
                # AUTHORIZED - GREEN
                color = (0, 255, 0)
                text = f"{match_name} ({score:.2f})"
                authorized_names_in_frame.append(match_name)
            else:
                # UNAUTHORIZED - RED
                color = (0, 0, 255)
                text = "UNKNOWN"
                has_unknown_in_frame = True
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
            # Text background
            cv2.rectangle(display_frame, (x, y-30), (x+w, y), color, -1)
            cv2.putText(display_frame, text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Handle recording state FIRST so the filename is generated
        recorder.process_frame(display_frame, has_unknown_in_frame)
        
        # Now log attendance using the generated filename
        for name in authorized_names_in_frame:
            logger.log_if_needed(name)
            
        if has_unknown_in_frame:
            logger.log_if_needed("Unknown", video_name=recorder.current_filename)
        
        # Show recording indicator
        if recorder.writer is not None:
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Attendance & Security Monitor", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    recorder.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Ensure directories exist
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    while True:
        print("\n" + "="*40)
        print("  STANDALONE FACE ATTENDANCE SYSTEM  ")
        print("="*40)
        print("1. Add New Authorized Face")
        print("2. Start Camera Monitoring (Security & Attendance)")
        print("3. Reset Face Database (Erase All Faces)")
        print("4. Exit")
        
        choice = input("Select an option (1/2/3/4): ").strip()
        
        if choice == '1':
            register_face()
        elif choice == '2':
            monitor()
        elif choice == '3':
            confirm = input("⚠️ Are you sure you want to permanently erase ALL registered faces? (yes/no): ").strip().lower()
            if confirm == 'yes':
                if DB_FILE.exists():
                    DB_FILE.unlink()
                    print("✅ Database successfully reset. All faces have been erased!")
                else:
                    print("Database is already empty!")
            else:
                print("Reset cancelled.")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
