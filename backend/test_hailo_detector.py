import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ai_cctv.smart_city.hailo_yolo_detector import HailoYOLODetector


def main():
    detector = HailoYOLODetector(
        hef_path="/usr/share/hailo-models/yolov8s_h8.hef",
        score_threshold=0.35,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not opened")
        return

    print("Hailo detector test running. Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed")
                break

            detections = detector.detect(frame)

            for det in detections:
                x1, y1, x2, y2 = det.box

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                cv2.putText(
                    frame,
                    f"{det.class_name} {det.score:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"Detections: {len(detections)}",
                (16, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Hailo YOLO Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()