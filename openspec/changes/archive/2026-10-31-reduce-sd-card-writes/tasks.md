## 1. Logging setup

- [x] 1.1 Add `LOG_LEVEL` parsing (info/debug, case-insensitive, invalid → info plus one warning) and `logging.basicConfig` with a timestamped format at the top of `detector.py`; verify by running `python3 -c` snippets or starting the container with `LOG_LEVEL=loud` and seeing one warning
- [x] 1.2 Before `import cv2`, set `OPENCV_LOG_LEVEL` / `OPENCV_FFMPEG_LOGLEVEL` via `os.environ.setdefault`, with a comment explaining the ordering; verify that `python3 -c "import detector"`-style startup shows no OpenCV info lines

## 2. Convert log calls

- [x] 2.1 Replace the startup prints with a ~2-line `info` summary (drop the `MOTION_THRESHOLD` print and env read) and keep the "alerting disabled" message as a `warning`; verify that the startup output is ≤3 lines at info
- [x] 2.2 Camera failure path: keep "failure declared" and "connection restored" at `info`; move "Failed to read frame, attempting to reconnect" to `debug`; verify by grep that no reconnect message is logged at info level
- [x] 2.3 `send_alert`: move "Alert sent" to `debug` and keep the POST failure at `error`; verify by reading the code
- [x] 2.4 Detection loop: keep NEW SESSION at `info`; move low-confidence, session-active, and snapshot-update messages to `debug`; verify by reading the code
- [x] 2.5 Move the periodic `[Status]` counter to `debug`, and keep the active-hours pause/resume at `info`; verify that `grep -n "print(" detector.py` returns nothing

## 3. In-memory snapshots

- [x] 3.1 Add a `send_detection(frame, confidence, ...)` helper that JPEG-encodes with `cv2.imencode`, logs an error and returns on encode failure, POSTs the unchanged payload, logs success at `debug`, and logs failure at `error`; verify that the payload keys are `timestamp`, `confidence`, `snapshot_base64`
- [x] 3.2 Replace both POST blocks (new session and picture update) with the helper, and remove `SNAPSHOTS_DIR`, `os.makedirs`, `current_snapshot_path`, `cv2.imwrite`, and `os.remove`; verify that `grep -nE "imwrite|SNAPSHOTS_DIR|os.remove" detector.py` returns nothing

## 4. Deployment config and docs

- [x] 4.1 In `docker-compose.yml`, remove the `volumes` mount and add a `logging` block (json-file, `max-size: "1m"`, `max-file: "3"`); verify with `docker-compose config`
- [x] 4.2 Update CLAUDE.md: add `LOG_LEVEL` to the config list, remove the snapshots-directory line, and adjust the flow step that says "Captures snapshot"; verify by reading the diff

## 5. Verification on the Pi

- [ ] 5.1 Deploy with `docker-compose up -d --build`, let it run through a cat visit, and confirm that `docker-compose logs` shows only startup, the new-session line, and active-hours transitions
- [ ] 5.2 Unplug the camera (or block its IP) for more than 2 minutes, then restore it, and confirm exactly one failure-declared line and one restored line with no `[h264 @` noise; if FFmpeg noise appears, apply the fallback from design.md
- [ ] 5.3 Restart with `LOG_LEVEL=debug` and confirm that the verbose diagnostics return; confirm that rocky-monitor still renders the snapshot image
