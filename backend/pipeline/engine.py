"""
pipeline/engine.py — Full attendance pipeline orchestrator.

Pipeline (all MIT/Apache-2.0 — commercial-safe):
  YuNet Face Detector     [MIT]  — detects all faces in full frame
  ByteTracker             [MIT]  — handles low-confidence tracking via IoU
  SFace Face Recognizer   [MIT]  — 128-dim cosine similarity matching
  Voting Engine           [MIT]  — requires 2-5 frames of confident recognition
                                   before locking identity to the track.
"""

import time
import threading
from collections import deque, Counter
import cv2
import numpy as np
from typing import Optional

from .face_detector import FaceDetector
from .recognizer    import FaceRecognizer
from .tracker       import ByteTracker

# Minimum face area (px²) to attempt recognition
MIN_FACE_AREA = 900   # ~30x30 px


class TrackState:
    """Per-track recognition cache and Voting Engine."""
    __slots__ = (
        "user_id", "confidence", "last_recog_time",
        "known", "name", "group", "message",
        "embeddings",
    )

    def __init__(self):
        self.user_id: Optional[str] = None
        self.confidence: float = 0.0
        self.last_recog_time: float = 0.0
        self.known: bool = False
        self.name: str = "Unknown"
        self.group: str = ""
        self.message: str = "Unknown face"
        
        # Rolling window of embeddings for voting
        self.embeddings = deque(maxlen=8)


class AttendanceEngine:
    """
    Main pipeline. Call process_frame() once per camera frame.
    Thread-safe via internal lock.
    """

    def __init__(self, models_dir: str):
        self._lock = threading.Lock()

        # Set threshold to 0.40 to filter out background noise (like clocks)
        self.face_detector = FaceDetector(models_dir, score_threshold=0.40)
        self.recognizer    = FaceRecognizer(models_dir)
        # ByteTracker: hold tracks for 90 frames (3 seconds) to survive extreme head turns
        self.tracker       = ByteTracker(track_thresh=0.60, high_iou_thresh=0.25, low_iou_thresh=0.40, max_age=90, min_hits=2)

        # track_id → TrackState
        self._track_states: dict[int, TrackState] = {}

        # In-memory set of user_ids already marked today
        self._marked_today: set[str] = set()

        print("[Engine] Pipeline ready: YuNet + ByteTrack + SFace + Voting Engine")

    # ── Public API ──────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        db_encodings: dict,
        users_db: list[dict],
        log_attendance_fn,
        already_logged_fn,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Run full pipeline on one BGR frame.
        """
        with self._lock:
            annotated = frame.copy()
            h, w = frame.shape[:2]
            now = time.time()

            # ── 1. Detect ALL faces in the full frame ──────────────────────
            raw_faces = self.face_detector.detect(frame)

            det_boxes_with_scores = []
            for face in raw_faces:
                fx, fy, fw, fh = face["bbox"]
                score = face["score"]
                x1, y1 = max(0, fx), max(0, fy)
                x2, y2 = min(w, fx + fw), min(h, fy + fh)
                if x2 - x1 < 15 or y2 - y1 < 15:
                    continue
                box = (x1, y1, x2, y2)
                det_boxes_with_scores.append((box, score))

            # ── 2. Update ByteTrack tracker ────────────────────────────────
            tracks = self.tracker.update(det_boxes_with_scores)

            # Prune states for dead tracks
            alive_ids = {t.track_id for t in self.tracker.trackers}
            for tid in list(self._track_states.keys()):
                if tid not in alive_ids:
                    del self._track_states[tid]

            results = []

            for track in tracks:
                tid  = track["track_id"]
                bbox = track["bbox"]

                x1, y1, x2, y2 = bbox
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)

                fw = x2 - x1
                fh = y2 - y1
                if fw < 15 or fh < 15:
                    continue

                face_area = fw * fh
                best_face = self._closest_face(raw_faces, (x1, y1, x2, y2))
                
                # Get or create track state
                state = self._track_states.setdefault(tid, TrackState())

                # ── 3. Voting Engine Recognition ────────────────────────────
                if not state.known and face_area >= MIN_FACE_AREA:
                    if not db_encodings:
                        # Database is empty! Impossible to match. 
                        # Resolve to Unknown immediately to avoid infinite "Analyzing..."
                        state.message = "No users registered"
                        state.name = "Unknown"
                    else:
                        # Only extract embeddings if face is clear enough
                        if best_face and best_face["score"] >= 0.25:
                            face_info_bbox = (x1, y1, fw, fh)
                            landmarks = best_face.get("landmarks")
                            
                            embeddings = self.recognizer.extract_embeddings(
                                frame, face_info_bbox, landmarks
                            )
                            if embeddings:
                                state.embeddings.extend(embeddings)

                        # Once we have at least 3 embeddings, run the Voting Engine
                        if len(state.embeddings) >= 3:
                            votes = []
                            best_score = 0.0
                            
                            for emb in state.embeddings:
                                uid, score = self.recognizer.match(emb, db_encodings)
                                if uid:
                                    votes.append(uid)
                                    best_score = max(best_score, score)
                                    
                            # If at least 2 frames agree on the same identity, lock it!
                            if len(votes) >= 2:
                                top_uid = Counter(votes).most_common(1)[0][0]
                                
                                user_info = next((u for u in users_db if u["user_id"] == top_uid), None)
                                if user_info:
                                    state.known = True
                                    state.user_id = top_uid
                                    state.confidence = best_score
                                    state.name = user_info["name"]
                                    state.group = user_info["group"]
                                    
                                    # Mark attendance
                                    if top_uid not in self._marked_today:
                                        if not already_logged_fn(top_uid):
                                            logged = log_attendance_fn(top_uid, user_info["name"], user_info["group"])
                                            state.message = "Attendance Marked" if logged else "Already Marked Today"
                                        else:
                                            state.message = "Already Marked Today"
                                        self._marked_today.add(top_uid)
                                    else:
                                        state.message = "Already Marked Today"
                            else:
                                # Failed to reach consensus or matched no one. 
                                # Rolling window automatically clears old frames.
                                state.message = "Unknown face"
                                state.name = "Unknown"

                # ── 5. Draw annotations ───────────────────────────────────
                is_known   = state.known
                # Color logic: Green = Known, Red = Unknown
                if is_known:
                    box_color = (0, 210, 90)
                else:
                    box_color = (0, 60, 230)
                    
                text_color = (255, 255, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                # Landmarks from raw detection (only draw if confident enough)
                if best_face and "landmarks" in best_face and best_face["score"] >= 0.40:
                    for lx, ly in best_face["landmarks"]:
                        cv2.circle(annotated, (int(lx), int(ly)), 3, box_color, -1)

                label = state.name
                if state.confidence > 0 and is_known:
                    label += f" {state.confidence:.2f}"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1 - 8, th + 6)
                cv2.rectangle(
                    annotated,
                    (x1, label_y - th - 6),
                    (x1 + tw + 6, label_y + 2),
                    box_color, -1,
                )
                cv2.putText(
                    annotated, label,
                    (x1 + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2,
                )

                cv2.putText(
                    annotated, f"#{tid}",
                    (x2 - 30, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1,
                )

                results.append({
                    "track_id":   tid,
                    "known":      state.known,
                    "name":       state.name,
                    "user_id":    state.user_id or "",
                    "group":      state.group,
                    "confidence": round(state.confidence, 3),
                    "status":     "known" if state.known else "unknown",
                    "message":    state.message,
                })

            return annotated, results

    def extract_embedding_from_image(self, image: np.ndarray, augment: bool = False) -> list[np.ndarray]:
        with self._lock:
            # Lowered threshold to 0.40 to ensure we capture side profiles during registration!
            self.face_detector._detector.setScoreThreshold(0.40)
            try:
                faces = self.face_detector.detect(image)
                if not faces: return []
            finally:
                self.face_detector._detector.setScoreThreshold(0.40)

        best = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
        bbox = best["bbox"]
        landmarks = best.get("landmarks")
        
        embs = []
        with self._lock:
            # 1. Original (both aligned and unaligned)
            original_embs = self.recognizer.extract_embeddings(image, bbox, landmarks)
            if original_embs:
                embs.extend(original_embs)
                
            if augment:
                # 2. Extreme Silhouette (simulates 2nd/3rd image scenario)
                shadow = cv2.convertScaleAbs(image, alpha=0.5, beta=-100)
                shadow_embs = self.recognizer.extract_embeddings(shadow, bbox, landmarks)
                if shadow_embs:
                    embs.extend(shadow_embs)
                    
                # 3. Moderate Dark
                dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-50)
                dark_embs = self.recognizer.extract_embeddings(dark, bbox, landmarks)
                if dark_embs:
                    embs.extend(dark_embs)
                    
        return embs

    def reset_daily_marks(self):
        with self._lock:
            self._marked_today.clear()

    def close(self):
        pass

    @staticmethod
    def _closest_face(raw_faces: list[dict], track_bbox: tuple[int, int, int, int]) -> Optional[dict]:
        if not raw_faces: return None
        tx1, ty1, tx2, ty2 = track_bbox
        best_iou  = -1.0
        best_face = None

        for face in raw_faces:
            fx, fy, fw, fh = face["bbox"]
            fx2, fy2 = fx + fw, fy + fh
            ix1 = max(tx1, fx);  iy1 = max(ty1, fy)
            ix2 = min(tx2, fx2); iy2 = min(ty2, fy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0: continue
            union = (tx2-tx1)*(ty2-ty1) + fw*fh - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou  = iou
                best_face = face

        return best_face
