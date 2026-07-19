## ADDED Requirements

### Requirement: Persistent camera failure detection
The detector SHALL track camera connection failures (failure to open the RTSP stream and failed frame reads) and SHALL declare the camera unreachable when failures have persisted continuously for `CAMERA_FAILURE_GRACE_PERIOD` seconds (default 60). A successful frame read SHALL reset the failure tracking.

#### Scenario: Failure declared after grace period
- **WHEN** the RTSP stream cannot be opened or read for at least `CAMERA_FAILURE_GRACE_PERIOD` seconds without interruption
- **THEN** the detector declares the camera unreachable and triggers a camera-failure alert

#### Scenario: Transient blip does not trigger an alert
- **WHEN** frame reads fail but a successful read occurs before `CAMERA_FAILURE_GRACE_PERIOD` elapses
- **THEN** failure tracking is reset and no alert is sent

#### Scenario: Startup connection failure is tracked
- **WHEN** the initial `VideoCapture` open fails at startup
- **THEN** failure tracking starts immediately, using the same grace period and alerting behavior as mid-operation failures

### Requirement: Camera-failure alert event
When a camera failure is declared, the detector SHALL POST a JSON alert event to the configured `ALERT_URL` containing `event_type` `"camera_failure"`, an ISO 8601 `timestamp`, the configured `camera_ip`, a human-readable `message` describing the failure and instructing that the camera IP may need updating, and `failure_duration_seconds`.

#### Scenario: Alert sent on declared failure
- **WHEN** the camera is declared unreachable
- **THEN** a `camera_failure` event with timestamp, camera IP, message, and failure duration is POSTed to `ALERT_URL`

#### Scenario: Alerting disabled when ALERT_URL is unset
- **WHEN** `ALERT_URL` is not configured
- **THEN** no alert POSTs are attempted and a warning is logged at startup that alerting is disabled

#### Scenario: Alert POST failure is non-fatal
- **WHEN** the POST to `ALERT_URL` fails (timeout, connection error, non-2xx)
- **THEN** the detector logs the error and continues operating without crashing

### Requirement: Alert throttling during prolonged outage
The detector SHALL send exactly one alert when a failure is declared and SHALL repeat reminder alerts every `ALERT_REMINDER_INTERVAL` seconds (default 3600) while the outage persists. No other alerts SHALL be sent for the same outage.

#### Scenario: No spam between reminders
- **WHEN** the camera remains unreachable after the initial alert
- **THEN** no further alerts are sent until `ALERT_REMINDER_INTERVAL` seconds have elapsed since the last alert

#### Scenario: Reminder during long outage
- **WHEN** the outage persists for longer than `ALERT_REMINDER_INTERVAL` since the last alert
- **THEN** another `camera_failure` event is sent with the updated `failure_duration_seconds`

### Requirement: Recovery notification
After a declared failure, the detector SHALL POST a `camera_recovered` event to `ALERT_URL` on the first successful frame read, including the total outage duration, and SHALL reset all failure and reminder state.

#### Scenario: Recovery event after outage
- **WHEN** a frame is successfully read after the camera was declared unreachable
- **THEN** a `camera_recovered` event with the total outage duration is POSTed and failure state is reset

#### Scenario: No recovery event without declared failure
- **WHEN** a transient read failure recovers before a failure was declared
- **THEN** no `camera_recovered` event is sent

### Requirement: Reconnect backoff after declared failure
While a failure is declared, the detector SHALL wait at least 30 seconds between reconnect attempts instead of the ~1 second retry used before declaration.

#### Scenario: Slower retries during declared outage
- **WHEN** the camera has been declared unreachable
- **THEN** subsequent reconnect attempts occur no more often than every 30 seconds until the connection is restored
