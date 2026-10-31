# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rocky Detector is a real-time cat detection system designed to run on Raspberry Pi 4. It captures frames from a TP-Link Tapo C200 camera via RTSP, runs YOLOv8n inference to detect cats (class 15), and sends detection events with base64-encoded snapshots to a downstream microservice (rocky-monitor).

Part of a distributed ecosystem:
- **rocky-detector** (this repo) - Edge detection on RPi
- **rocky-monitor** - Elixir/Phoenix backend microservice
- **Next.js frontend** - User-facing dashboard

## Commands

```bash
# Build Docker image
docker build . -t rocky_detector

# Run container (production)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Architecture

**Single-file application:** `detector.py`

```
TP-Link Camera (RTSP) → Frame Capture → YOLO Inference → Session Management → HTTP POST to Microservice
```

**Key flow:**
1. Connects to camera RTSP stream
2. Reads frames in continuous loop (skips frames per FRAME_SKIP_RATIO for RPi CPU)
3. Checks active hours window before processing
4. Runs YOLO detection for cat class with confidence threshold
5. Groups detections into sessions (SESSION_TIMEOUT between events)
6. Captures snapshot on first detection + periodic updates (PICTURE_UPDATE_INTERVAL), JPEG-encoded in memory (never written to disk)
7. POSTs base64-encoded image + metadata to MICROSERVICE_URL

**Configuration via .env:**
- `CAMERA_IP`, `CAMERA_USER`, `CAMERA_PASS` - Camera credentials
- `MICROSERVICE_URL` - Backend endpoint
- `ACTIVE_START_HOUR`/`ACTIVE_END_HOUR` - Monitoring window (default 7-23)
- `FRAME_SKIP_RATIO` - Process every Nth frame (default 60, tune for CPU)
- `CONFIDENCE_THRESHOLD` - YOLO confidence (default 0.65)
- `SESSION_TIMEOUT` - Seconds before new session (default 300)
- `PICTURE_UPDATE_INTERVAL` - Snapshot update frequency (default 300s)
- `ALERT_URL` - Endpoint for camera failure/recovery alert events (unset = alerting disabled)
- `CAMERA_FAILURE_GRACE_PERIOD` - Seconds of continuous connection failure before an alert (default 60)
- `ALERT_REMINDER_INTERVAL` - Seconds between reminder alerts during an outage (default 3600)
- `LOG_LEVEL` - `info` (default, only meaningful events) or `debug` (verbose diagnostics; use temporarily, it increases SD card writes)

## Dependencies

- Python 3.11
- ultralytics (YOLOv8n)
- opencv-python-headless
- requests
- python-dotenv

Runs on an SD card, so disk writes are kept minimal: quiet logging by default, no snapshot files, and Docker logs capped (json-file, 1m x 3) in `docker-compose.yml`.
