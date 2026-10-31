import base64
import json
import logging
import os
import time
from datetime import datetime

# OpenCV/FFmpeg read these when cv2 is imported, so they must be set before the import.
# setdefault lets an explicit value in .env override the quiet defaults.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # AV_LOG_QUIET

import cv2
import requests
from ultralytics import YOLO

# Logging: "info" logs only meaningful events (SD card friendly), "debug" adds diagnostics
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rocky_detector")
log.setLevel(logging.DEBUG if LOG_LEVEL == "debug" else logging.INFO)
if LOG_LEVEL not in ("info", "debug"):
    log.warning(f"Unknown LOG_LEVEL '{LOG_LEVEL}', falling back to info")

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
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

# Session management: how long (in seconds) without detection before starting a new session
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "300"))  # 5 minutes (matches monitor)
# How often to update the picture during an active session (in seconds)
PICTURE_UPDATE_INTERVAL = int(os.getenv("PICTURE_UPDATE_INTERVAL", "300"))  # 5 minutes

# Performance tuning for Raspberry Pi
FRAME_SKIP_RATIO = int(os.getenv("FRAME_SKIP_RATIO", "60"))  # Process every Nth frame
FRAME_DELAY = float(os.getenv("FRAME_DELAY", "0.1"))  # Delay between frames in seconds

# Camera failure alerting
ALERT_URL = os.getenv("ALERT_URL")
CAMERA_FAILURE_GRACE_PERIOD = int(os.getenv("CAMERA_FAILURE_GRACE_PERIOD", "60"))
ALERT_REMINDER_INTERVAL = int(os.getenv("ALERT_REMINDER_INTERVAL", "3600"))
# Reconnect pacing: fast retries before a failure is declared, slow after
RECONNECT_DELAY_NORMAL = 1
RECONNECT_DELAY_DECLARED = 30

log.info(
    f"Rocky Detector starting (active {ACTIVE_START_HOUR}:00-{ACTIVE_END_HOUR}:00, "
    f"confidence > {CONFIDENCE_THRESHOLD}, every {FRAME_SKIP_RATIO} frames, "
    f"alerting {'on' if ALERT_URL else 'off'}, log level {logging.getLevelName(log.level).lower()})"
)
if not ALERT_URL:
    log.warning("ALERT_URL not set, camera failure alerting is disabled")


def send_alert(event_type, message, failure_duration_seconds):
    if not ALERT_URL:
        return
    alert = {
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "camera_ip": CAMERA_IP,
        "message": message,
        "failure_duration_seconds": int(failure_duration_seconds),
    }
    try:
        requests.post(ALERT_URL, json=alert, timeout=5)
        log.debug(f"Alert sent: {event_type}")
    except Exception as e:
        log.error(f"Failed to send alert ({event_type}): {e}")


def send_detection(frame, confidence):
    # Encode in memory: snapshots never touch the SD card
    ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        log.error("Failed to encode snapshot, skipping detection")
        return

    detection = {
        "timestamp": datetime.now().isoformat(),
        "confidence": confidence,
        "snapshot_base64": base64.b64encode(jpeg.tobytes()).decode("utf-8"),
    }

    try:
        requests.post(MICROSERVICE_URL, json=detection, timeout=5)
        log.debug("Detection sent successfully")
    except Exception as e:
        log.error(f"Failed to send detection: {e}")

log.debug("Loading YOLO model...")
model = YOLO("yolov8n.pt")
log.debug("Model loaded successfully")

# Camera failure tracking
failure_start_time = None
failure_declared = False
last_alert_time = None

log.debug(f"Connecting to camera at {CAMERA_IP}...")
cap = cv2.VideoCapture(RTSP_URL)
if cap.isOpened():
    log.debug("Camera connected successfully")
else:
    log.debug("Failed to connect to camera")
    failure_start_time = time.time()

log.info("Ready to detect Rocky!")

frame_count = 0
detection_count = 0

# Session tracking
last_detection_time = None
last_picture_update_time = None
in_detection_session = False

in_active_hours = True
while True:
    # Check active hours
    current_hour = datetime.now().hour
    if current_hour < ACTIVE_START_HOUR or current_hour >= ACTIVE_END_HOUR:
        if in_active_hours:
            log.info(
                f"Outside active hours (current: {current_hour}:00), pausing monitoring..."
            )
            in_active_hours = False
        time.sleep(10)
        continue
    else:
        if not in_active_hours:
            log.info(
                f"Entering active hours (current: {current_hour}:00), resuming monitoring..."
            )
            in_active_hours = True

    ret, frame = cap.read()
    if not ret:
        current_time = time.time()
        if failure_start_time is None:
            failure_start_time = current_time
        failure_duration = current_time - failure_start_time

        if not failure_declared and failure_duration >= CAMERA_FAILURE_GRACE_PERIOD:
            failure_declared = True
            last_alert_time = current_time
            log.warning(f"Camera unreachable for {int(failure_duration)}s, declaring connection failure")
            send_alert(
                "camera_failure",
                f"Camera at {CAMERA_IP} has been unreachable for {int(failure_duration)}s. "
                "Its IP address may have changed and needs to be updated.",
                failure_duration,
            )
        elif failure_declared and current_time - last_alert_time >= ALERT_REMINDER_INTERVAL:
            last_alert_time = current_time
            send_alert(
                "camera_failure",
                f"Camera at {CAMERA_IP} is still unreachable after {int(failure_duration)}s. "
                "Its IP address may have changed and needs to be updated.",
                failure_duration,
            )

        log.debug("Failed to read frame, attempting to reconnect...")
        cap.release()
        cap = cv2.VideoCapture(RTSP_URL)
        time.sleep(RECONNECT_DELAY_DECLARED if failure_declared else RECONNECT_DELAY_NORMAL)
        continue

    if failure_start_time is not None:
        if failure_declared:
            outage_duration = time.time() - failure_start_time
            log.info(f"Camera connection restored after {int(outage_duration)}s outage")
            send_alert(
                "camera_recovered",
                f"Camera at {CAMERA_IP} is reachable again after an outage of {int(outage_duration)}s.",
                outage_duration,
            )
        failure_start_time = None
        failure_declared = False
        last_alert_time = None

    frame_count += 1

    # Process every Nth frame for efficiency
    if frame_count % FRAME_SKIP_RATIO != 0:
        time.sleep(FRAME_DELAY)
        continue

    # YOLO detection (motion detection disabled)
    results = model(frame, classes=[15], verbose=False)

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])

            # Log detections that didn't meet threshold
            if confidence <= CONFIDENCE_THRESHOLD:
                log.debug(f"Cat detected but confidence too low: {confidence:.2f} (threshold: {CONFIDENCE_THRESHOLD})")
                continue

            if confidence > CONFIDENCE_THRESHOLD:
                current_time = time.time()

                # Check if this is a new detection session
                is_new_session = (
                    last_detection_time is None
                    or current_time - last_detection_time > SESSION_TIMEOUT
                )

                if is_new_session:
                    detection_count += 1
                    log.info(f"NEW SESSION - Cat detected! Confidence: {confidence:.2f}")
                    last_picture_update_time = current_time
                    send_detection(frame, confidence)
                    in_detection_session = True
                else:
                    # Still in same session - check if we should update the picture
                    time_since_last_update = current_time - last_picture_update_time
                    if time_since_last_update >= PICTURE_UPDATE_INTERVAL:
                        log.debug(f"Session active - Updating snapshot (confidence: {confidence:.2f})")
                        last_picture_update_time = current_time
                        send_detection(frame, confidence)
                    else:
                        # Just log that we still see the cat, no picture update
                        minutes_until_update = int((PICTURE_UPDATE_INTERVAL - time_since_last_update) / 60)
                        log.debug(f"Session active - Cat still detected (next update in ~{minutes_until_update}m)")

                last_detection_time = current_time

    if frame_count % 100 == 0:
        log.debug(f"[Status] Frames processed: {frame_count}, Detections: {detection_count}")
