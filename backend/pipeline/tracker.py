"""
pipeline/tracker.py — Multi-object tracking using a pure-Python SORT
implementation (Simple Online and Realtime Tracking).

License: MIT — fully commercial-safe.

We implement a lightweight SORT variant (Kalman + IoU) so we have zero
dependency on AGPL-licensed code. supervision's ByteTrack is also
Apache-2.0, but we keep it self-contained for portability.
"""

import numpy as np
from collections import OrderedDict


class KalmanBoxTracker:
    """
    Represents a single tracked bounding box using a simple constant-velocity
    Kalman filter.

    State: [cx, cy, w, h, vx, vy, vw, vh]
    """

    _id_counter = 0

    def __init__(self, bbox: tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1

        # State vector: [cx, cy, w, h, vx, vy, vw, vh]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)

        # Transition matrix (constant velocity)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix (observe cx, cy, w, h)
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0

        self.P = np.eye(8) * 10.0
        self.Q = np.eye(8) * 1.0   # process noise
        self.R = np.eye(4) * 5.0   # measurement noise

        KalmanBoxTracker._id_counter += 1
        self.track_id = KalmanBoxTracker._id_counter
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._to_bbox()

    def update(self, bbox: tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        z = np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=float)

        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    def _to_bbox(self) -> tuple[int, int, int, int]:
        cx, cy, w, h = self.state[:4]
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))

    def get_bbox(self) -> tuple[int, int, int, int]:
        return self._to_bbox()


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


class SORTTracker:
    """
    Lightweight SORT multi-object tracker.

    Returns active tracks with their assigned integer track IDs.
    Tracks are stable across frames, which prevents flicker in recognition.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 2,
        iou_threshold: float = 0.25,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: list[tuple[int, int, int, int]]) \
            -> list[dict]:
        """
        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes.

        Returns:
            List of dicts with keys: track_id, bbox (x1,y1,x2,y2).
            Only returns tracks that have been confirmed (min_hits).
        """
        self.frame_count += 1

        # Predict all existing trackers
        predicted = []
        to_del = []
        for i, t in enumerate(self.trackers):
            p = t.predict()
            predicted.append(p)
            if any(np.isnan(p)):
                to_del.append(i)
        for i in reversed(to_del):
            self.trackers.pop(i)
            predicted.pop(i)

        # Match detections to trackers via IoU
        matched_t = set()
        matched_d = set()
        if predicted and detections:
            iou_matrix = np.zeros((len(detections), len(predicted)))
            for di, det in enumerate(detections):
                for ti, pred in enumerate(predicted):
                    iou_matrix[di, ti] = _iou(det, pred)

            # Greedy match: pick best IoU pairs above threshold
            while True:
                idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[idx] < self.iou_threshold:
                    break
                di, ti = idx
                matched_d.add(di)
                matched_t.add(ti)
                iou_matrix[di, :] = -1
                iou_matrix[:, ti] = -1
                self.trackers[ti].update(detections[di])

        # Create new trackers for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_d:
                self.trackers.append(KalmanBoxTracker(det))

        # Remove dead trackers
        alive = []
        for i, t in enumerate(self.trackers):
            if i in matched_t or t.time_since_update == 0:
                pass  # updated this frame
            if t.time_since_update > self.max_age:
                continue
            alive.append(t)
        self.trackers = alive

        # Return confirmed tracks
        results = []
        for t in self.trackers:
            if t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                results.append({
                    "track_id": t.track_id,
                    "bbox": t.get_bbox(),
                })
        return results
