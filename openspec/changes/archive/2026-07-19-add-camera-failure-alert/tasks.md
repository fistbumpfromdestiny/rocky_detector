## 1. Configuration

- [x] 1.1 Add `ALERT_URL`, `CAMERA_FAILURE_GRACE_PERIOD` (default 60), and `ALERT_REMINDER_INTERVAL` (default 3600) env vars to `detector.py` configuration block
- [x] 1.2 Log at startup whether alerting is enabled (`ALERT_URL` set) or disabled with a warning
- [x] 1.3 Document the new env vars in CLAUDE.md and `.env.example` (create the example entry if the file exists)

## 2. Alert helper

- [x] 2.1 Implement `send_alert(event_type, message, failure_duration_seconds)` that POSTs the JSON payload (event_type, ISO timestamp, camera_ip, message, failure_duration_seconds) to `ALERT_URL` with a timeout, no-ops when `ALERT_URL` is unset, and logs+swallows any request failure

## 3. Failure tracking in the main loop

- [x] 3.1 Introduce failure-tracking state: `failure_start_time`, `failure_declared`, `last_alert_time`
- [x] 3.2 Start failure tracking when the initial `VideoCapture` open fails at startup (replace the current warning-only branch)
- [x] 3.3 In the failed-read branch, start/continue failure tracking and declare failure once `CAMERA_FAILURE_GRACE_PERIOD` has elapsed, sending the initial `camera_failure` alert
- [x] 3.4 Send reminder `camera_failure` alerts every `ALERT_REMINDER_INTERVAL` while the failure persists, with updated duration
- [x] 3.5 Use a 30s sleep between reconnect attempts once failure is declared (keep ~1s before declaration)
- [x] 3.6 On successful frame read: if a failure was declared, send `camera_recovered` with total outage duration; reset all failure/alert state in both cases

## 4. Verification

- [x] 4.1 Simulate an unreachable camera (wrong `CAMERA_IP`) and verify: no alert before the grace period, one `camera_failure` POST at declaration, 30s reconnect backoff in logs
- [x] 4.2 Verify reminder alert fires after `ALERT_REMINDER_INTERVAL` (use a short override value for testing)
- [x] 4.3 Restore the correct IP and verify a single `camera_recovered` event with plausible duration, then normal detection resumes
- [x] 4.4 Verify detector runs unchanged (no POST attempts, startup warning logged) when `ALERT_URL` is unset
