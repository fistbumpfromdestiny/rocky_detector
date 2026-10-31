## Context

`detector.py` is a single-file script that has about 30 `print()` calls, and many of them run inside the hot loop. The Dockerfile runs `python3 -u`, so each line becomes its own write to Docker's default json-file driver. That driver has no rotation configured in `docker-compose.yml`. Snapshots are written to `/home/pi/rocky-snapshots` (bind-mounted to `./snapshots`), deleted or overwritten per session, and read back immediately for base64 encoding. Nothing else reads that directory. See proposal.md for the motivation and specs/ for the required behavior.

## Goals / Non-Goals

**Goals:**
- Default runtime writes to the SD card come from a handful of meaningful events per day, not thousands.
- Verbose diagnostics are one env var away, with no code change needed.
- Snapshot handling never touches the disk.

**Non-Goals:**
- Fixing the RTSP buffer lag (frames read slower than the camera produces them). This is a separate change, even though it is the likely root cause of FFmpeg decode noise.
- Retrying or queueing failed detection POSTs.
- Host-level SD protection (log2ram, tmpfs `/var/log`, journald storage). This is outside the repo.
- Restructuring `detector.py` beyond what the logging and snapshot changes require.

## Decisions

### Use stdlib `logging` instead of filtered `print()`
Configure one logger with `logging.basicConfig(level=..., format="%(asctime)s %(levelname)s %(message)s")`. Default-level events use `info`/`warning`/`error`, and diagnostics use `debug`.
- *Why*: Level filtering comes for free, timestamps come from the formatter (which replaces the hand-rolled `datetime.now().strftime` strings scattered through the loop), and no new dependency is needed.
- *Alternative*: a `VERBOSE` flag with `if VERBOSE: print(...)` guards. This is simpler on day one, but it scatters conditionals and still needs manual timestamps.
- `LOG_LEVEL` accepts only `info`/`debug`. An invalid value falls back to `info` with one warning (see spec). Supporting `warning`/`error` adds nothing, because the default is already near-silent.

### Silence OpenCV/FFmpeg from Python before importing cv2
Before `import cv2`, call `os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")` and `os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")` (AV_LOG_QUIET).
- *Why*: The suppression lives next to the code it affects. Because `setdefault` is used, an operator can override it through `.env`, which satisfies the override scenario.
- *Alternative*: `ENV` lines in the Dockerfile. That works too, but it hides the behavior away from the code and needs an image rebuild to change.
- The ordering constraint (set the environment before the cv2 import) is a subtle trap, so it gets a short comment at that spot.

### Encode snapshots in memory with `cv2.imencode`
`ok, buf = cv2.imencode(".jpg", frame)`, then `base64.b64encode(buf.tobytes())`. If `ok` is false, log an error and skip the POST. The duplicated "encode + build payload + POST" block in the new-session and update branches is folded into one small `send_detection(frame, confidence, label)` helper, because both call sites have to change anyway.
- *Why*: This removes three disk operations per snapshot (write, read, delete). `cv2.imencode` produces the same JPEG bytes as `cv2.imwrite` with default quality, so the payload stays the same.
- `SNAPSHOTS_DIR`, `os.makedirs`, and `current_snapshot_path` are removed. The `volumes` entry is dropped from `docker-compose.yml`.

### Cap container logs with json-file rotation
Add the following to the service:
```yaml
logging:
  driver: json-file
  options:
    max-size: "1m"
    max-file: "3"
```
- *Why*: This bounds the log to about 3 MB in the worst case (for example a crash loop, or noise we missed) and keeps `docker-compose logs` working.
- *Alternatives*: The `local` driver (compressed and rotated by default) is fine but less familiar. `none` gives zero writes but breaks `docker-compose logs`, which CLAUDE.md relies on.

### Remove the stale `MOTION_THRESHOLD` startup print
It reports a setting that has no effect. The startup summary shrinks to about 2 lines: config highlights, alerting status, and a ready line. The env var read can be removed too, because nothing uses it.

## Risks / Trade-offs

- [Real problems get harder to diagnose once the log is quiet] → Errors still log by default, and `LOG_LEVEL=debug` with a container restart brings back full diagnostics.
- [FFmpeg env var names or semantics differ across opencv-python versions on ARM, so the suppression may not take effect] → Verify on the Pi during implementation by checking that a stream glitch produces no `[h264 @` lines. If it doesn't work, fall back to `cv2.utils.logging.setLogLevel` and/or Dockerfile `ENV`. The Docker log cap still bounds the damage either way.
- [The log cap only applies when the container is recreated] → The migration step runs `docker-compose up -d`, which recreates the container when the compose config changes.
- [Debug mode left on reintroduces heavy writes] → The log cap bounds it. Debug is documented as temporary.

## Migration Plan

1. Deploy the new image and compose file: `docker-compose up -d --build`.
2. Confirm that the startup summary appears in `docker-compose logs` and that no status lines follow.
3. Optionally delete the old `./snapshots` folder on the Pi.
4. Rollback: redeploy the previous commit. The snapshot volume and verbose prints come back, and nothing downstream depends on either.
