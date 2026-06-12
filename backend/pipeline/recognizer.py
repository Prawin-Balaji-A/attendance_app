"""
pipeline/recognizer.py — Face recognition using SFace and Scikit-Learn KNN.

SFace extracts the 128-dim features.
Scikit-Learn KNeighborsClassifier learns the decision boundaries.
License: MIT/BSD — fully commercial-safe.
"""

import os
import urllib.request
import cv2
import numpy as np
from typing import Optional
from sklearn.neighbors import KNeighborsClassifier

# SFace ONNX model — MIT license
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"

# Stricter fallback threshold if KNN isn't ready
COSINE_THRESHOLD = 0.48   

class FaceRecognizer:
    def __init__(self, models_dir: str):
        model_path = os.path.join(models_dir, SFACE_FILENAME)
        os.makedirs(models_dir, exist_ok=True)

        if not os.path.exists(model_path):
            print("[FaceRecognizer] Downloading SFace model (MIT)...")
            urllib.request.urlretrieve(SFACE_URL, model_path)
            print(f"[FaceRecognizer] Model saved to {model_path}")

        self._recognizer = cv2.FaceRecognizerSF.create(model_path, "")
        self.knn = None
        print("[FaceRecognizer] SFace loaded (MIT license)")

    def train_classifier(self, encodings_db: dict):
        """
        Train a K-Nearest Neighbors classifier on the augmented embeddings.
        This provides massively better accuracy and side-profile resilience
        than simple cosine similarity.
        """
        X = []
        y = []
        for user_id, embeddings in encodings_db.items():
            for emb in embeddings:
                X.append(emb)
                y.append(user_id)
                
        if len(set(y)) < 1 or len(X) < 1:
            self.knn = None
            print("[FaceRecognizer] Not enough users to train ML classifier.")
            return

        # Use 5 neighbors (or less if we have very few samples, though with augmentation we have 100+)
        n_neighbors = min(5, len(X))
        
        # We use metric='cosine' to match SFace's native distance metric.
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine', weights='distance')
        self.knn.fit(X, y)
        print(f"[FaceRecognizer] ML Classifier trained on {len(X)} augmented profiles across {len(set(y))} users.")

    def extract_embedding(
        self,
        image: np.ndarray,
        face_bbox: tuple[int, int, int, int],
        landmarks: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        x, y, fw, fh = face_bbox

        if fw < 20 or fh < 20:
            return None

        try:
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
        encodings_db: dict,
    ) -> tuple[Optional[str], float]:
        """
        Predict the user using the trained KNN classifier.
        Falls back to manual thresholding if KNN predicts someone but the distance is too high (Unknown).
        """
        if embedding is None:
            return None, 0.0

        if self.knn is not None:
            # KNN predict
            distances, indices = self.knn.kneighbors([embedding], n_neighbors=1)
            dist = distances[0][0]
            
            # Scikit-learn cosine distance is (1 - cosine_similarity).
            # If our similarity threshold is 0.48, max allowed distance is (1 - 0.48) = 0.52
            max_allowed_distance = 1.0 - COSINE_THRESHOLD
            
            if dist <= max_allowed_distance:
                predicted_user = self.knn.predict([embedding])[0]
                # Convert distance back to a pseudo-similarity score for the UI
                confidence = 1.0 - dist
                return predicted_user, confidence
            else:
                # Too far from any known cluster -> Unknown
                return None, (1.0 - dist)
                
        else:
            # Fallback to manual 1-NN if KNN isn't trained
            if not encodings_db:
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

            if best_score >= COSINE_THRESHOLD:
                return best_id, best_score
            return None, best_score
