# Rocky Detector

Real-time cat detection system for monitoring Rocky, the neighborhood cat who visits our balcony.

## Overview

Python script that runs on a Raspberry Pi 4 with a TP-Link Tapo C200 camera. It uses motion detection to trigger YOLO-based cat detection, captures snapshots, and sends detections to an Elixir microservice for further processing and analytics.
```
Camera (RTSP) → Motion Detection → YOLO Inference → Microservice
                (Always running)   (Motion triggered)  (HTTP POST)
```

## Architecture

Part of a distributed system:
- **rocky-detector** (this repo) - Detection layer on Raspberry Pi
- **rocky-monitor** - Elixir microservice for event processing & analytics
- **Next.js app** - Frontend with webhooks

## Features

- Motion-triggered YOLO cat detection
- Automatic snapshot capture on detection
- Integration with Elixir microservice

## Requirements

- Raspberry Pi 4
- TP-Link Tapo C200 camera
- Docker & Docker Compose
- Network access to camera and microservice
