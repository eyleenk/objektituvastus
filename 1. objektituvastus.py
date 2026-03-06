# -*- coding: utf-8 -*-
import cv2
import os
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Kaamerat ei saa avada.")
        return

    # Loo kaust piltide jaoks
    os.makedirs("captures", exist_ok=True)

    print("Live YOLO töötab. Vajuta 'c' pildi tegemiseks, 'q' sulgemiseks.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kaamerast ei saa pilti.")
            break

        results = model(frame, stream=False)
        r = results[0]
        annotated = r.plot()

        cv2.imshow("Live YOLO", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            orig_path = f"captures/orig_{timestamp}.jpg"
            yolo_path = f"captures/yolo_{timestamp}.jpg"

            cv2.imwrite(orig_path, frame)
            cv2.imwrite(yolo_path, annotated)

            print(f"Pildid salvestatud:\n - {orig_path}\n - {yolo_path}")

            cv2.imshow("YOLO tulemus", annotated)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
