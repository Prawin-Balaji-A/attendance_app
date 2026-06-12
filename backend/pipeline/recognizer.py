"""
pipeline/recognizer.py — Face recognition using SFace (OpenCV DNN).

SFace is the official OpenCV face recognition model.
License: MIT — fully commercial-safe.

Recognition uses cosine similarity. Multiple stored embeddings per person
(from registration) are all compared and the best match wins.
"""

import os
import urllib.request
import cv2
import numpy as np
from typing import Optional


# SFace ONNX model — MIT license
# Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"

COSINE_THRESHOLD = 0.38   # Reverted back to SFace recommended threshold (0.363 - 0.38)
L2_THRESHOLD = 1.128


class FaceRecognizer:
    """
    Extracts 128-dim face embeddings using SFace (MIT license) and matches
    them against stored encodings using cosine similarity.
    """

    def __init__(self, models_dir: str):
        model_path = os.path.join(models_dir, SFACE_FILENAME)
        os.makedirs(models_dir, exist_ok=True)

        if not os.path.exists(model_path):
            print("[FaceRecognizer] Downloading SFace model (MIT)...")
            urllib.request.urlretrieve(SFACE_URL, model_path)
            print(f"[FaceRecognizer] Model saved to {model_path}")

        self._recognizer = cv2.FaceRecognizerSF.create(model_path, "")
        print("[FaceRecognizer] SFace loaded (MIT license)")

    def train_classifier(self, encodings_db: dict):
        # Stub to avoid breaking main.py which calls this.
        # We no longer use KNN because it ruins accuracy with augmented data.
        pass

    def extract_embedding(
        self,
        image: np.ndarray,
        face_bbox: tuple[int, int, int, int],
        landmarks: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract a 128-dim L2-normalised embedding for a face.
        """
        h, w = image.shape[:2]
        x, y, fw, fh = face_bbox

        if fw < 20 or fh < 20:
            return None

        try:
            # Build face object in OpenCV SFace format
            if landmarks is not None and len(landmarks) == 5:
                lm = landmarks.flatten()
            else:
                lm = np.array([
                    x + fw * 0.30, y + fh * 0.35,
                    x + fw * 0.70, y + fh * 0.35,
                    x + fw * 0.50, y + fh * 0.55,
                    x + fw * 0.35, y + fh * 0.75,
                    x + fw * 0.65, y + fh * 0.75,
                ], dtype=float)

            face_info = np.array([[x, y, fw, fh] + list(lm)], dtype=np.float32)

            aligned = self._recognizer.alignCrop(image, face_info[0])
            embedding = self._recognizer.feature(aligned)
            return embedding.flatten()

        except Exception as e:
            print(f"[FaceRecognizer] embed error: {e}")
            return None

    def match(
        self,
        embedding: np.ndarray,
        encodings_db: dict,  # {user_id: [embedding, ...]}
        threshold: float = COSINE_THRESHOLD,
    ) -> tuple[Optional[str], float]:
        """
        Find the best matching user in the database using purely SFace Cosine similarity.
        """
        if not encodings_db or embedding is None:
            return None, 0.0

        best_id = None
        best_score = -1.0

        for user_id, stored_list in encodings_db.items():
            for stored_emb in stored_list:
                try:
                    score = self._recognizer.match(
                        embedding.reshape(1, -1),
                        np.array(stored_emb, dtype=np.float32).reshape(1, -1),
                        cv2.FaceRecognizerSF_FR_COSINE,
                    )
                    if score > best_score:
                        best_score = score
                        best_id = user_id
                except Exception:
                    continue

        if best_score >= threshold:
            return best_id, best_score
        return None, best_score
