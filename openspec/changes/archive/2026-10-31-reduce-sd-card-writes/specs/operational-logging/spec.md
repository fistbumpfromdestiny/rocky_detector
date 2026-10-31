## Purpose

Keep the detector's log output limited to meaningful operational events, so it does not wear out or fill the Raspberry Pi's SD card, while still allowing verbose diagnostics on demand.

## ADDED Requirements

### Requirement: Configurable log verbosity
The detector SHALL read a `LOG_LEVEL` environment variable that accepts `info` (default) or `debug`, case-insensitively. If the value is unrecognized, the detector SHALL fall back to `info` and log a single warning at startup.

#### Scenario: Default verbosity
- **WHEN** `LOG_LEVEL` is not set
- **THEN** the detector logs only the events defined for the default level

#### Scenario: Debug verbosity
- **WHEN** `LOG_LEVEL` is set to `debug`
- **THEN** the detector logs both the default-level events and the debug diagnostics

#### Scenario: Invalid value
- **WHEN** `LOG_LEVEL` is set to an unrecognized value such as `loud`
- **THEN** the detector uses `info` and logs one warning naming the invalid value

### Requirement: Default-level log events
At the default level the detector SHALL log only the following:
- a startup summary of at most a few lines: configuration highlights, whether alerting is enabled, and readiness
- camera failure declared
- camera connection restored after a declared failure
- the start of a new cat detection session, with confidence
- entering and leaving active hours
- errors, such as a failed POST to `MICROSERVICE_URL` or `ALERT_URL`, or an unrecognized `LOG_LEVEL`

Every log line SHALL include a timestamp.

#### Scenario: Cat visit
- **WHEN** a cat is detected above the confidence threshold and no session is active
- **THEN** exactly one line is logged announcing the new session and its confidence

#### Scenario: Detection POST fails
- **WHEN** sending a detection to `MICROSERVICE_URL` fails
- **THEN** one error line describing the failure is logged

### Requirement: Outage logging is not repeated
At the default level, a single camera outage SHALL produce at most one "failure declared" line and one "connection restored" line. Reconnect attempts and successfully sent reminder alerts SHALL NOT be logged at the default level. Failed alert POSTs are errors and SHALL still be logged.

#### Scenario: Prolonged outage
- **WHEN** the camera stays unreachable for several hours with reconnect attempts every 30 seconds and hourly reminder alerts that are delivered successfully
- **THEN** the default-level log contains exactly one failure-declared line for that outage

#### Scenario: Transient blip
- **WHEN** frame reads fail briefly and recover before the grace period elapses
- **THEN** nothing is logged at the default level

### Requirement: High-frequency events are debug-only
The following SHALL be logged only at the `debug` level:
- reconnect attempts
- detections below the confidence threshold
- "cat still detected" during an active session
- snapshot updates during an active session
- successfully sent detections and alerts
- periodic frame/detection counters

#### Scenario: Cat stays in view
- **WHEN** a cat remains in view for an hour at the default level
- **THEN** only the new-session line is logged, with no per-frame or per-update lines

#### Scenario: Low-confidence cat
- **WHEN** a cat is detected below the confidence threshold at the default level
- **THEN** nothing is logged

### Requirement: Native library noise suppressed
By default, the detector SHALL suppress non-fatal diagnostic output that OpenCV and its FFmpeg backend write to stderr, such as H.264 decode warnings. An operator SHALL be able to re-enable it by setting the corresponding OpenCV/FFmpeg log-level environment variables explicitly.

#### Scenario: Corrupted stream packets
- **WHEN** the RTSP stream delivers packets that FFmpeg reports as decode errors
- **THEN** no FFmpeg warning lines appear in the container log by default

#### Scenario: Operator override
- **WHEN** the operator sets the OpenCV/FFmpeg log-level environment variable explicitly
- **THEN** the operator's value is used instead of the suppressed default

### Requirement: Bounded container log storage
The deployment configuration SHALL cap the container's stored log size so that log output cannot grow without bound on the host's storage, and logs SHALL remain viewable with `docker-compose logs`.

#### Scenario: Long-running deployment
- **WHEN** the container runs for months
- **THEN** its stored log files stay within the configured size cap

#### Scenario: Viewing logs
- **WHEN** the operator runs `docker-compose logs -f`
- **THEN** recent log output is shown
