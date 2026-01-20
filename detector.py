import base64
import json
import os
import time
from datetime import datetime

import cv2
import requests
from ultralytics import YOLO

# Configuration
CAMERA_IP = os.getenv("CAMERA_IP")
USERNAME = os.getenv("CAMERA_USER")
PASSWORD = os.getenv("CAMERA_PASS")
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/stream1"
MICROSERVICE_URL = os.getenv("MICROSERVICE_URL")

# Active hours (only monitor during these times)
ACTIVE_START_HOUR = int(os.getenv("ACTIVE_START_HOUR", "7"))
ACTIVE_END_HOUR = int(os.getenv("ACTIVE_END_HOUR", "23"))

# Detection thresholds
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "1500"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

print("Rocky Detector Starting...")

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(RTSP_URL)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Check active hours
    current_hour = datetime.now().hour
    if current_hour < ACTIVE_START_HOUR or current_hour >= ACTIVE_END_HOUR:
        time.sleep(10)
        continue

    ret, frame = cap.read()
    if not ret:
        time.sleep(1)
        continue

    # Motion detection
    fg_mask = bg_subtractor.apply(frame)
    motion_pixels = cv2.countNonZero(fg_mask)

    if motion_pixels > MOTION_THRESHOLD:
        # YOLO detection
        results = model(frame, classes=[15])

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])

                if confidence > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Save snapshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = f"~/rocky-snapshots/rocky_{timestamp}.jpg"
                    cv2.imwrite(snapshot_path, frame)

                    # Send to microservice
                    detection = {
                        "timestamp": datetime.now().isoformat(),
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "snapshot_path": snapshot_path,
                    }

                    try:
                        requests.post(MICROSERVICE_URL, json=detection, timeout=2)
                        print(f"Detection sent ({confidence:.2f})")
                    except Exception as e:
                        print(f"Failed to send detection: {e}")

    time.sleep(0.2)  # 5 FPS
