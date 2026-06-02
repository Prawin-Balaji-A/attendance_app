import cv2
import time
import threading


class CameraManager:
    def __init__(self, backend="picamera2", source=0, width=1280, height=720):
        self.backend = backend
        self.source = source
        self.width = width
        self.height = height

        self.picam2 = None
        self.cap = None
        self.lock = threading.Lock()
        self.started = False

    def start(self):
        if self.started:
            return

        if self.backend == "picamera2":
            from picamera2 import Picamera2

            self.picam2 = Picamera2()
            self.picam2.configure(
                self.picam2.create_preview_configuration(
                    main={
                        "size": (self.width, self.height),
                        "format": "RGB888",
                    }
                )
            )
            self.picam2.start()
            time.sleep(2)

        else:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.started = True

    def read(self):
        with self.lock:
            if not self.started:
                self.start()

            if self.backend == "picamera2":
                frame_rgb = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                return True, frame_bgr

            ret, frame = self.cap.read()
            return ret, frame

    def stop(self):
        with self.lock:
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

                if self.picam2 is not None:
                    self.picam2.stop()
                    self.picam2.close()
                    self.picam2 = None
            except Exception:
                pass

            self.started = False