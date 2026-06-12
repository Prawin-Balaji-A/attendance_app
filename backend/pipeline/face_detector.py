"""
pipeline/face_detector.py — Face detection using YuNet (OpenCV DNN).

YuNet is the official OpenCV face detection model.
License: MIT — fully commercial-safe.

Model weights are auto-downloaded from OpenCV's model zoo on first run.
"""

import os
import urllib.request
import cv2
import numpy as np


# YuNet ONNX model — MIT license
# Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"


class FaceDetector:
    """
    Detects faces using YuNet (MIT license, OpenCV DNN).

    Works on full frames OR cropped person regions.
    Returns face crops and bounding boxes with 5 landmarks.
    """

    def __init__(self, models_dir: str, score_threshold: float = 0.55):
        self.score_threshold = score_threshold
        model_path = os.path.join(models_dir, YUNET_FILENAME)
        os.makedirs(models_dir, exist_ok=True)

        if not os.path.exists(model_path):
            print(f"[FaceDetector] Downloading YuNet model (MIT)...")
            urllib.request.urlretrieve(YUNET_URL, model_path)
            print(f"[FaceDetector] Model saved to {model_path}")

        self._detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=20,
        )
        print("[FaceDetector] YuNet loaded (MIT license)")

    def detect(
        self, image: np.ndarray
    ) -> list[dict]:
        """
        Detect all faces in an image.

        Returns list of:
          {
            'bbox':   (x, y, w, h),      # face bounding box
            'score':  float,
            'landmarks': np.ndarray,     # shape (5,2): eye/nose/mouth kps
          }
        """
        h, w = image.shape[:2]
        if h < 10 or w < 10:
            return []

        # YuNet requires image size set before detect
        self._detector.setInputSize((w, h))

        _, faces = self._detector.detect(image)
        if faces is None:
            return []

        results = []
        for face in faces:
            # face columns: x,y,w,h, 10 landmarks floats, score
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])

            # Clamp to image bounds
            x  = max(0, x)
            y  = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)

            if fw < 20 or fh < 20:
                continue

            score = float(face[-1])
            landmarks = face[4:14].reshape(5, 2)

            results.append({
                "bbox": (x, y, fw, fh),
                "score": score,
                "landmarks": landmarks,
            })

        return results

    def crop_face(
        self,
        image: np.ndarray,
        face: dict,
        pad_ratio: float = 0.25,
    ) -> np.ndarray:
        """Return a padded face crop suitable for recognition."""
        x, y, fw, fh = face["bbox"]
        h, w = image.shape[:2]

        pad_x = int(fw * pad_ratio)
        pad_y = int(fh * pad_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)

        return image[y1:y2, x1:x2]
