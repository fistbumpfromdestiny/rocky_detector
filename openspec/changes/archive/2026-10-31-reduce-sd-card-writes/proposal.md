## Why

The detector runs on a Raspberry Pi that boots from an SD card, and it writes to that card far more than it needs to. Every `print()` goes unbuffered to Docker's json-file log, which has no rotation, so the log grows without limit. In normal operation the detector emits thousands of lines a day: a status line every ~30s, one line per reconnect attempt during an outage (~2,900/day), and one line per processed frame while a cat is in view. On top of that, every detection writes a JPEG snapshot to disk only to read it back for base64 encoding. All of this wears the card and can eventually fill it.

## What Changes

- Quiet logging by default. Only these events are logged:
  - a short startup summary
  - camera failure declared (once per outage)
  - camera recovered (once per outage)
  - new cat session
  - active-hours pause/resume
  - real errors, such as a failed detection or alert POST
- These high-frequency lines are removed from the default output:
  - per-retry reconnect messages
  - low-confidence detections
  - "cat still detected" messages
  - periodic status counters
  - per-reminder "alert sent" messages
  - snapshot-update messages
- New `LOG_LEVEL` env var (default `info`). Setting it to `debug` restores the verbose diagnostics for troubleshooting and threshold tuning.
- OpenCV/FFmpeg native stderr output (e.g. `[h264 @ ...] error while decoding`) is silenced by default.
- **BREAKING**: Snapshots are no longer written to disk. Frames are JPEG-encoded in memory for the base64 payload, and the `./snapshots` volume mount and `/home/pi/rocky-snapshots` directory are no longer used. The payload sent to rocky-monitor is unchanged.
- Container logs are size-capped (json-file `max-size`/`max-file`) in `docker-compose.yml` as a safety net.
- CLAUDE.md is updated to cover `LOG_LEVEL` and to remove the snapshot-directory note.

## Capabilities

### New Capabilities
- `operational-logging`: what the detector logs at the default and debug verbosity levels, how verbosity is configured, suppression of native library noise, and bounded container log storage.
- `detection-snapshots`: how detection snapshots are produced and delivered to rocky-monitor without persisting them to local storage.

### Modified Capabilities
<!-- None: camera-failure-alerting behavior (alert events, throttling, error logging on POST failure) is unchanged. -->

## Impact

- `detector.py`:
  - all `print()` calls are replaced by levelled logging
  - snapshot code switches from disk writes to in-memory encoding
  - `SNAPSHOTS_DIR` and the unused `MOTION_THRESHOLD` startup print are removed
- `docker-compose.yml`: the `volumes` mount is removed and a `logging` block is added.
- `CLAUDE.md`: configuration and snapshot documentation.
- rocky-monitor: no change, because the detection payload (`timestamp`, `confidence`, `snapshot_base64`) stays identical.
- Operators: run `docker-compose up -d` to recreate the container so the log cap applies. An existing `./snapshots` folder on the Pi is left in place and can be deleted by hand.
