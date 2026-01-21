import base64
import json
import os
import time
from datetime import datetime

import cv2
import requests
from ultralytics import YOLO

SNAPSHOTS_DIR = "/home/pi/rocky-snapshots"
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

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

# Session management: how long (in seconds) without detection before starting a new session
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "60"))
# How often to update the picture during an active session (in seconds)
PICTURE_UPDATE_INTERVAL = int(os.getenv("PICTURE_UPDATE_INTERVAL", "300"))  # 5 minutes

print("Rocky Detector Starting...")
print(f"Active hours: {ACTIVE_START_HOUR}:00 - {ACTIVE_END_HOUR}:00")
print(f"Motion threshold: {MOTION_THRESHOLD} pixels")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully")

print(f"Connecting to camera at {CAMERA_IP}...")
cap = cv2.VideoCapture(RTSP_URL)
if cap.isOpened():
    print("Camera connected successfully")
else:
    print("Warning: Failed to connect to camera")

print("Ready to detect Rocky!")

frame_count = 0
detection_count = 0

# Session tracking
last_detection_time = None
last_picture_update_time = None
current_snapshot_path = None
in_detection_session = False

in_active_hours = True
while True:
    # Check active hours
    current_hour = datetime.now().hour
    if current_hour < ACTIVE_START_HOUR or current_hour >= ACTIVE_END_HOUR:
        if in_active_hours:
            print(
                f"Outside active hours (current: {current_hour}:00), pausing monitoring..."
            )
            in_active_hours = False
        time.sleep(10)
        continue
    else:
        if not in_active_hours:
            print(
                f"Entering active hours (current: {current_hour}:00), resuming monitoring..."
            )
            in_active_hours = True

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame, attempting to reconnect...")
        cap.release()
        cap = cv2.VideoCapture(RTSP_URL)
        time.sleep(1)
        continue

    frame_count += 1

    # Process every 10th frame for efficiency
    if frame_count % 10 != 0:
        continue

    # YOLO detection (motion detection disabled)
    results = model(frame, classes=[15], verbose=False)

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence > CONFIDENCE_THRESHOLD:
                current_time = time.time()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Check if this is a new detection session
                is_new_session = (
                    last_detection_time is None
                    or current_time - last_detection_time > SESSION_TIMEOUT
                )

                if is_new_session:
                    detection_count += 1
                    print(f"[{timestamp}] NEW SESSION - Cat detected! Confidence: {confidence:.2f}")

                    # Delete old snapshot if it exists
                    if current_snapshot_path and os.path.exists(current_snapshot_path):
                        os.remove(current_snapshot_path)
                        print(f"  Removed old snapshot")

                    # Save new snapshot
                    snapshot_path = os.path.join(
                        SNAPSHOTS_DIR, f"rocky_{timestamp}.jpg"
                    )
                    cv2.imwrite(snapshot_path, frame)
                    current_snapshot_path = snapshot_path
                    last_picture_update_time = current_time

                    # Send to microservice
                    detection = {
                        "timestamp": datetime.now().isoformat(),
                        "confidence": confidence,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "snapshot_path": snapshot_path,
                        "is_new_session": True,
                    }

                    try:
                        requests.post(MICROSERVICE_URL, json=detection, timeout=2)
                        print(f"  Detection sent successfully")
                    except Exception as e:
                        print(f"  Failed to send detection: {e}")

                    in_detection_session = True
                else:
                    # Still in same session - check if we should update the picture
                    time_since_last_update = current_time - last_picture_update_time
                    if time_since_last_update >= PICTURE_UPDATE_INTERVAL:
                        print(f"[{timestamp}] Session active - Updating snapshot (confidence: {confidence:.2f})")
                        cv2.imwrite(current_snapshot_path, frame)
                        last_picture_update_time = current_time

                        # Send updated picture to microservice
                        detection = {
                            "timestamp": datetime.now().isoformat(),
                            "confidence": confidence,
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "snapshot_path": current_snapshot_path,
                            "is_new_session": False,
                        }

                        try:
                            requests.post(MICROSERVICE_URL, json=detection, timeout=2)
                            print(f"  Updated snapshot sent successfully")
                        except Exception as e:
                            print(f"  Failed to send updated snapshot: {e}")
                    else:
                        # Just log that we still see the cat, no picture update
                        minutes_until_update = int((PICTURE_UPDATE_INTERVAL - time_since_last_update) / 60)
                        print(f"[{timestamp}] Session active - Cat still detected (next update in ~{minutes_until_update}m)")

                last_detection_time = current_time

    if frame_count % 100 == 0:
        print(f"[Status] Frames processed: {frame_count}, Detections: {detection_count}")
