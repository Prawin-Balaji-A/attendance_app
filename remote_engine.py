import os
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from .hailo_infer import HailoModel, HAILO_AVAILABLE
from .tracker import Sort

if HAILO_AVAILABLE:
    from hailo_platform import VDevice
else:
    VDevice = None

class AttendanceEngine:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.vdevice = VDevice() if VDevice is not None else None
        
        self.person_detector = HailoModel('/usr/share/hailo-models/yolox_s_leaky_h8l_rpi.hef', vdevice=self.vdevice)
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        
        yunet_path = os.path.join(models_dir, 'face_detection_yunet_2023mar.onnx')
        self.face_detector = cv2.FaceDetectorYN.create(
            model=yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.4,
            nms_threshold=0.3,
            top_k=5000
        )
        
        sface_path = os.path.join(models_dir, 'face_recognition_sface_2021dec.onnx')
        self.face_recognizer = cv2.FaceRecognizerSF.create(
            model=sface_path,
            config=""
        )
        
        self.recognition_threshold = 0.5

    def detect_persons(self, frame: np.ndarray):
        results = self.person_detector.infer(frame)
        h, w = frame.shape[:2]
        persons = []
        
        if self.person_detector.is_mock:
            return np.empty((0, 5))
            
        out = list(results.values())[0][0]
        if len(out) == 0:
            return np.empty((0, 5))
            
        # out is a list of 80 classes. Index 0 is Person.
        person_detections = out[0]
        
        for det in person_detections:
            # det shape is [ymin, xmin, ymax, xmax, score]
            score = det[4]
            if score > 0.25:
                ymin, xmin, ymax, xmax = det[0], det[1], det[2], det[3]
                x1 = max(0, int(xmin * w))
                y1 = max(0, int(ymin * h))
                x2 = min(w, int(xmax * w))
                y2 = min(h, int(ymax * h))
                persons.append([x1, y1, x2, y2, score])
                
        return np.array(persons) if persons else np.empty((0, 5))

    def detect_face(self, person_crop: np.ndarray):
        h, w = person_crop.shape[:2]
        if h < 10 or w < 10:
            return None
            
        self.face_detector.setInputSize((w, h))
        faces = self.face_detector.detect(person_crop)
        
        if faces[1] is not None:
            best_face = max(faces[1], key=lambda x: x[-1])
            return best_face
        return None

    def extract_encoding(self, person_crop: np.ndarray, face_data: np.ndarray):
        aligned_face = self.face_recognizer.alignCrop(person_crop, face_data)
        feature = self.face_recognizer.feature(aligned_face)
        return feature[0]

    def recognize_face(self, encoding, db_encodings: dict):
        if not db_encodings or encoding is None:
            return None, 0.0
            
        best_match = None
        min_dist = float('inf')
        
        for user_id, known_encodings in db_encodings.items():
            for known_enc in known_encodings:
                dist = cosine(encoding, known_enc)
                if dist < min_dist:
                    min_dist = dist
                    best_match = user_id
                    
        if min_dist < self.recognition_threshold:
            confidence = 1.0 - min_dist
            return best_match, confidence
        return None, 0.0

    def process_frame(self, frame: np.ndarray, db_encodings: dict):
        person_bboxes = self.detect_persons(frame)
        tracked_objects = self.tracker.update(person_bboxes)
        
        recognized_users = []
        
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, track)
            
            h, w = frame.shape[:2]
            px1, py1 = max(0, x1), max(0, y1)
            px2, py2 = min(w, x2), min(h, y2)
            person_crop = frame[py1:py2, px1:px2]
            
            face_data = self.detect_face(person_crop)
            
            label = f"ID: {track_id}"
            color = (0, 255, 255)
            
            if face_data is not None:
                encoding = self.extract_encoding(person_crop, face_data)
                user_id, conf = self.recognize_face(encoding, db_encodings)
                
                if user_id:
                    label = f"{user_id} ({conf:.2f})"
                    color = (0, 255, 0)
                    recognized_users.append((user_id, conf))
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                    
                fx, fy, fw, fh = map(int, face_data[:4])
                cv2.rectangle(frame, (px1 + fx, py1 + fy), (px1 + fx + fw, py1 + fy + fh), color, 1)
            
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return frame, recognized_users
