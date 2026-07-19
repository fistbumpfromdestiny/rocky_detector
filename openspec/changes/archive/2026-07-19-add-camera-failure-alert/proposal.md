## Why

When the camera's DHCP lease changes (e.g., after a power outage), the detector can no longer reach the RTSP stream at the configured `CAMERA_IP`. Today this failure is invisible: the detector silently loops on "Failed to read frame, attempting to reconnect..." forever, and the admin only discovers the outage by noticing that detections stopped. The admin needs to be actively notified that the camera connection has failed and the configured IP must be updated.

## What Changes

- Detect persistent camera connection failure: both failure to open the RTSP stream at startup and repeated frame-read failures during operation are tracked, and after a configurable threshold the detector declares the camera unreachable.
- Send a camera-failure alert event to the rocky-monitor microservice (which is responsible for eventually messaging the admin), including timestamp, configured camera IP, and a human-readable error description.
- Throttle alerts so a prolonged outage does not spam the microservice: one alert when the failure is first declared, then periodic reminders at a configurable interval.
- Send a recovery event when the camera connection is re-established after a declared failure, so the admin knows the issue is resolved.
- New configuration via `.env`: failure threshold and alert reminder interval.

Out of scope: rocky-monitor's handling of the alert (persisting it and messaging the admin) lives in the rocky-monitor repo; this change only defines and emits the event from the detector side.

## Capabilities

### New Capabilities
- `camera-failure-alerting`: Detection of persistent camera connection failure, emission of failure/recovery alert events to the microservice, and alert throttling behavior.

### Modified Capabilities

<!-- none: no existing specs in openspec/specs/ -->

## Impact

- `detector.py`: startup connection block and the frame-read reconnect loop gain failure tracking and alert emission; new helper for posting alert events.
- Configuration: new env vars (e.g., `CAMERA_FAILURE_THRESHOLD`, `ALERT_REMINDER_INTERVAL`) documented in CLAUDE.md / `.env`.
- rocky-monitor: must accept the new alert event payload (coordinated change in that repo; the detector defines the contract).
- No new Python dependencies (`requests` already used).
