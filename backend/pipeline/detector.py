"""
pipeline/detector.py — Direct full-frame face detection using YuNet (MIT).

Instead of a separate person-detection step (which required MediaPipe),
YuNet detects ALL faces directly in the full camera frame.
This is actually faster and just as accurate on Raspberry Pi.

License: MIT (YuNet, OpenCV) — fully commercial-safe.
"""

import numpy as np


class PersonDetector:
    """
    Stub class kept for API compatibility with engine.py.
    Actual detection is done by FaceDetector (YuNet) on the full frame.
    This class simply returns the full frame as a single region.
    """

    def __init__(self, **kwargs):
        print("[PersonDetector] Using full-frame mode (no separate person detection)")

    def detect(self, frame: np.ndarray) -> list:
        """Return the full frame as a single 'person' region."""
        h, w = frame.shape[:2]
        return [(0, 0, w, h, 1.0)]

    def close(self):
        pass
