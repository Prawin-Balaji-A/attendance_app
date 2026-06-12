"""
pipeline/tracker.py — Multi-object tracking using ByteTrack algorithm.

License: MIT — fully commercial-safe.

We implement ByteTrack to maintain tracking of faces even when the detection
confidence drops drastically (e.g., when a person bends down or turns).
This associates low-confidence detections to existing tracks via IoU.
"""

import numpy as np


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

        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)

        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0

        self.P = np.eye(8) * 10.0
        self.Q = np.eye(8) * 1.0
        self.R = np.eye(4) * 5.0

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


class ByteTracker:
    """
    ByteTrack logic implementation.
    Associates high-confidence detections first, then low-confidence detections
    to unmatched tracks, avoiding ID drops on partial occlusions.
    """

    def __init__(
        self,
        track_thresh: float = 0.50,
        high_iou_thresh: float = 0.25,
        low_iou_thresh: float = 0.40,
        max_age: int = 30,
        min_hits: int = 2,
    ):
        self.track_thresh = track_thresh
        self.high_iou_thresh = high_iou_thresh
        self.low_iou_thresh = low_iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: list[tuple[tuple[int, int, int, int], float]]) -> list[dict]:
        """
        detections: list of ((x1, y1, x2, y2), score)
        """
        self.frame_count += 1

        # 1. Split detections by score
        high_dets = []
        low_dets = []
        for det, score in detections:
            if score >= self.track_thresh:
                high_dets.append((det, score))
            else:
                low_dets.append((det, score))

        # 2. Predict trackers
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

        # 3. Match high confidence
        matched_high_d, matched_high_t, unmatched_high_d, unmatched_t = self._match(
            high_dets, predicted, self.high_iou_thresh
        )

        # 4. Match low confidence with unmatched tracks
        unmatched_predicted = [predicted[t] for t in unmatched_t]
        matched_low_d, matched_low_t, _, _ = self._match(
            low_dets, unmatched_predicted, self.low_iou_thresh
        )

        # Map matched_low_t back to original tracker indices
        matched_t_from_low = [unmatched_t[i] for i in matched_low_t]

        # 5. Update state
        for d_idx, t_idx in zip(matched_high_d, matched_high_t):
            self.trackers[t_idx].update(high_dets[d_idx][0])

        for d_idx, t_idx in zip(matched_low_d, matched_t_from_low):
            self.trackers[t_idx].update(low_dets[d_idx][0])

        # 6. Init new tracks
        for d_idx in unmatched_high_d:
            self.trackers.append(KalmanBoxTracker(high_dets[d_idx][0]))

        # 7. Remove dead tracks
        matched_all_t = set(matched_high_t).union(set(matched_t_from_low))
        alive = []
        for i, t in enumerate(self.trackers):
            if i in matched_all_t or t.time_since_update == 0:
                pass
            if t.time_since_update > self.max_age:
                continue
            alive.append(t)
        self.trackers = alive

        # 8. Return confirmed tracks
        results = []
        for t in self.trackers:
            if t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                results.append({
                    "track_id": t.track_id,
                    "bbox": t.get_bbox(),
                })
        return results

    def _match(self, dets, preds, iou_threshold):
        if not dets or not preds:
            return [], [], list(range(len(dets))), list(range(len(preds)))

        iou_matrix = np.zeros((len(dets), len(preds)))
        for di, (det, _) in enumerate(dets):
            for ti, pred in enumerate(preds):
                iou_matrix[di, ti] = _iou(det, pred)

        matched_d = []
        matched_t = []

        while True:
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[idx] < iou_threshold:
                break
            di, ti = idx
            matched_d.append(di)
            matched_t.append(ti)
            iou_matrix[di, :] = -1
            iou_matrix[:, ti] = -1

        unmatched_d = [i for i in range(len(dets)) if i not in matched_d]
        unmatched_t = [i for i in range(len(preds)) if i not in matched_t]

        return matched_d, matched_t, unmatched_d, unmatched_t
