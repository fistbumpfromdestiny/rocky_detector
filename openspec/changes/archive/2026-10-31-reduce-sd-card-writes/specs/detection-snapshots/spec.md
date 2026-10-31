## Purpose

Deliver detection snapshots to rocky-monitor as base64-encoded JPEG images without writing them to the Raspberry Pi's local storage.

## ADDED Requirements

### Requirement: Snapshots are not persisted locally
The detector SHALL NOT write snapshot images to local storage. Each snapshot SHALL be encoded as a JPEG in memory and sent as base64.

#### Scenario: New session snapshot
- **WHEN** a new cat detection session starts
- **THEN** the frame is JPEG-encoded in memory and sent, and no image file is created on disk

#### Scenario: Session snapshot update
- **WHEN** `PICTURE_UPDATE_INTERVAL` has elapsed during an active session and the cat is still detected
- **THEN** an updated in-memory JPEG is sent, and no image file is created or modified on disk

### Requirement: Detection payload unchanged
Detection events POSTed to `MICROSERVICE_URL` SHALL keep the existing JSON shape: `timestamp` (ISO 8601), `confidence` (number), and `snapshot_base64` (a base64-encoded JPEG).

#### Scenario: Monitor compatibility
- **WHEN** rocky-monitor receives a detection event after this change
- **THEN** it decodes `snapshot_base64` into a valid JPEG image, exactly as before

### Requirement: Encoding failure is non-fatal
If a frame cannot be JPEG-encoded, the detector SHALL log an error, skip that detection POST, and keep running.

#### Scenario: Encode fails
- **WHEN** JPEG encoding of a frame fails
- **THEN** an error is logged, no detection is POSTed for that frame, and detection continues with the next frames
