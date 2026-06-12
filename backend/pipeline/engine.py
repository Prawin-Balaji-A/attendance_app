"""
pipeline/engine.py — Full attendance pipeline orchestrator.

Pipeline (all MIT/Apache-2.0 — commercial-safe):
  YuNet Face Detector     [MIT]  — detects all faces in full frame
  SORT Tracker            [MIT]  — stable track IDs across frames
  SFace Face Recognizer   [MIT]  — 128-dim cosine similarity matching
  Decision Engine                — confidence gating + once-per-day attendance

No MediaPipe / AGPL / non-commercial models used.
"""

import time
import threading
import cv2
import numpy as np
from typing import Optional

from .face_detector import FaceDetector
from .recognizer    import FaceRecognizer
from .tracker       import SORTTracker

# Re-run recognition on an existing track every N seconds
RECOG_COOLDOWN = 4.0

# Minimum face area (px²) to attempt recognition
MIN_FACE_AREA = 900   # ~30x30 px


class TrackState:
    """Per-track recognition cache."""
    __slots__ = (
        "user_id", "confidence", "last_recog_time",
        "known", "name", "group", "message",
    )

    def __init__(self):
        self.user_id: Optional[str] = None
        self.confidence: float = 0.0
        self.last_recog_time: float = 0.0
        self.known: bool = False
        self.name: str = "Unknown"
        self.group: str = ""
        self.message: str = "Unknown face"


class AttendanceEngine:
    """
    Main pipeline. Call process_frame() once per camera frame.
    Thread-safe via internal lock.
    """

    def __init__(self, models_dir: str):
        self._lock = threading.Lock()

        # Threshold set to 0.70 to balance false positives with distant faces
        self.face_detector = FaceDetector(models_dir, score_threshold=0.70)
        self.recognizer    = FaceRecognizer(models_dir)
        self.tracker       = SORTTracker(max_age=25, min_hits=2, iou_threshold=0.25)

        # track_id → TrackState
        self._track_states: dict[int, TrackState] = {}

        # In-memory set of user_ids already marked today
        self._marked_today: set[str] = set()

        print("[Engine] Pipeline ready: YuNet + SORT + SFace (all MIT)")

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

        Returns:
            annotated_frame — full-resolution frame with boxes drawn
            results         — list matching /live-results schema
        """
        with self._lock:
            annotated = frame.copy()
            h, w = frame.shape[:2]
            now = time.time()

            # ── 1. Detect ALL faces in the full frame ──────────────────────
            raw_faces = self.face_detector.detect(frame)   # list of face dicts

            # Convert face bboxes (x,y,w,h) → tracker format (x1,y1,x2,y2)
            det_boxes = []
            face_map  = {}   # (x1,y1,x2,y2) → face dict
            for face in raw_faces:
                fx, fy, fw, fh = face["bbox"]
                x1, y1 = max(0, fx), max(0, fy)
                x2, y2 = min(w, fx + fw), min(h, fy + fh)
                if x2 - x1 < 15 or y2 - y1 < 15:
                    continue
                box = (x1, y1, x2, y2)
                det_boxes.append(box)
                face_map[box] = face

            # ── 2. Update SORT tracker ─────────────────────────────────────
            tracks = self.tracker.update(det_boxes)

            # Prune states for dead tracks
            active_ids = {t["track_id"] for t in tracks}
            for tid in list(self._track_states.keys()):
                if tid not in active_ids:
                    del self._track_states[tid]

            results = []

            for track in tracks:
                tid  = track["track_id"]
                bbox = track["bbox"]   # (x1, y1, x2, y2) — tracker-smoothed

                x1, y1, x2, y2 = bbox
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)

                fw = x2 - x1
                fh = y2 - y1
                if fw < 15 or fh < 15:
                    continue

                face_area = fw * fh

                # Find the closest raw face to this track bbox for landmarks
                best_face = self._closest_face(raw_faces, (x1, y1, x2, y2))

                # Get or create track state
                state = self._track_states.setdefault(tid, TrackState())

                # ── 3. Recognition ────────────────────────────────────────
                should_recog = (
                    face_area >= MIN_FACE_AREA
                    and (now - state.last_recog_time) >= RECOG_COOLDOWN
                    and db_encodings
                )

                if should_recog:
                    # Use tracker-smoothed bbox for the face location
                    face_info_bbox = (x1, y1, fw, fh)
                    landmarks = best_face.get("landmarks") if best_face else None

                    embedding = self.recognizer.extract_embedding(
                        frame, face_info_bbox, landmarks
                    )
                    state.last_recog_time = now

                    if embedding is not None:
                        user_id, score = self.recognizer.match(embedding, db_encodings)

                        if user_id:
                            user_info = next(
                                (u for u in users_db if u["user_id"] == user_id), None
                            )
                            if user_info:
                                state.user_id    = user_id
                                state.confidence = score
                                state.known      = True
                                state.name       = user_info["name"]
                                state.group      = user_info["group"]

                                # ── 4. Mark attendance ────────────────────
                                if user_id not in self._marked_today:
                                    if not already_logged_fn(user_id):
                                        logged = log_attendance_fn(
                                            user_id,
                                            user_info["name"],
                                            user_info["group"],
                                        )
                                        if logged:
                                            state.message = "Attendance Marked"
                                        else:
                                            state.message = "Already Marked Today"
                                    else:
                                        state.message = "Already Marked Today"
                                    self._marked_today.add(user_id)
                                else:
                                    state.message = "Already Marked Today"
                        else:
                            # No match found — only reset if not already confirmed
                            if not state.known:
                                state.user_id    = None
                                state.known      = False
                                state.name       = "Unknown"
                                state.group      = ""
                                state.message    = "Unknown face"
                                state.confidence = score

                # ── 5. Draw annotations ───────────────────────────────────
                is_known   = state.known
                box_color  = (0, 210, 90)  if is_known else (0, 60, 230)
                text_color = (255, 255, 255)

                # Face bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                # Landmarks from raw detection
                if best_face and "landmarks" in best_face:
                    for lx, ly in best_face["landmarks"]:
                        cv2.circle(annotated, (int(lx), int(ly)), 3, box_color, -1)

                # Name + confidence label
                label = state.name
                if state.confidence > 0:
                    label += f" {state.confidence:.2f}"

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
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

                # Track ID (small, bottom-right of box)
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

    def extract_embedding_from_image(
        self,
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Extract a face embedding from an image.
        Used for /register-image and /register-live.
        Picks the largest face found.
        """
        with self._lock:
            # Temporarily lower threshold to capture side profiles
            self.face_detector._detector.setScoreThreshold(0.60)
            
            try:
                faces = self.face_detector.detect(image)
                if not faces:
                    return None
            finally:
                # Restore strict threshold
                self.face_detector._detector.setScoreThreshold(0.85)

        # Pick the largest face (most prominent)
        best = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
        
        with self._lock:
            return self.recognizer.extract_embedding(
                image,
                best["bbox"],
                best.get("landmarks"),
            )

    def reset_daily_marks(self):
        """Call at midnight to clear in-memory attendance cache."""
        with self._lock:
            self._marked_today.clear()

    def close(self):
        pass

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _closest_face(
        raw_faces: list[dict],
        track_bbox: tuple[int, int, int, int],
    ) -> Optional[dict]:
        """Find the raw face detection closest to a tracker bbox (by IoU)."""
        if not raw_faces:
            return None

        tx1, ty1, tx2, ty2 = track_bbox
        best_iou  = -1.0
        best_face = None

        for face in raw_faces:
            fx, fy, fw, fh = face["bbox"]
            fx2, fy2 = fx + fw, fy + fh

            ix1 = max(tx1, fx);  iy1 = max(ty1, fy)
            ix2 = min(tx2, fx2); iy2 = min(ty2, fy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0:
                continue
            union = (tx2-tx1)*(ty2-ty1) + fw*fh - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou  = iou
                best_face = face

        return best_face
