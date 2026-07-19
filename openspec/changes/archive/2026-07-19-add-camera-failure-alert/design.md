## Context

`detector.py` is a single-file loop on a Raspberry Pi 4. Camera access happens in two places:

1. **Startup** (`detector.py:49-53`): `cv2.VideoCapture(RTSP_URL)` — on failure it only prints a warning and continues into the main loop.
2. **Main loop** (`detector.py:85-91`): when `cap.read()` fails it prints "Failed to read frame, attempting to reconnect...", recreates the capture, sleeps 1s, and retries forever.

When the camera's IP changes (e.g., DHCP re-lease after a power outage), the reconnect loop spins indefinitely and nobody is told. The downstream rocky-monitor (Elixir/Phoenix) is the component that reaches the admin, so the detector's job is to emit a machine-readable alert event, not to send email/push itself.

## Goals / Non-Goals

**Goals:**
- Declare a persistent camera failure after a grace period of continuous connection/read failures.
- Notify rocky-monitor with a structured alert event (failure + recovery), throttled during long outages.
- Reduce pointless reconnect churn once a failure has been declared.

**Non-Goals:**
- Auto-discovering the camera's new IP (mDNS/ARP scan) — a possible future change.
- Implementing admin messaging (email/push) — that is rocky-monitor's responsibility.
- Retry queues or persistence for alerts that fail to send.

## Decisions

### 1. Time-based failure declaration, not attempt counting
Track `failure_start_time` from the first failed open/read. Once `time.time() - failure_start_time >= CAMERA_FAILURE_GRACE_PERIOD` (default **60s**), declare the camera unreachable and emit the alert. Any successful frame read resets the tracker.

*Why:* transient RTSP hiccups (Wi-Fi blips, camera reboot) recover within seconds; a time window filters them out regardless of how fast the retry loop spins. Counting attempts would couple the threshold to retry timing.

*Alternative considered:* consecutive-failure counter — rejected because its meaning changes whenever retry sleep durations are tuned.

### 2. Alerts go to a dedicated endpoint via `ALERT_URL`
New env var `ALERT_URL` (e.g., the monitor's `/api/alerts`). Payload:

```json
{
  "event_type": "camera_failure" | "camera_recovered",
  "timestamp": "<ISO 8601>",
  "camera_ip": "<CAMERA_IP>",
  "message": "<human-readable description incl. last error>",
  "failure_duration_seconds": 120
}
```

If `ALERT_URL` is unset, alerting is disabled and a warning is logged at startup (backward compatible).

*Why:* the existing detection payload posted to `MICROSERVICE_URL` has no type discriminator; mixing alert events into that endpoint would force rocky-monitor to sniff payload shapes. A separate endpoint keeps both contracts clean.

*Alternative considered:* add `event_type` to the existing endpoint — rejected; it changes the existing contract and risks breaking current monitor parsing.

### 3. Throttled reminders + recovery event
- One alert immediately when failure is declared.
- Reminder alerts every `ALERT_REMINDER_INTERVAL` seconds (default **3600**) while the outage persists.
- One `camera_recovered` event on the first successful frame read after a declared failure, including total outage duration.

*Why:* a multi-hour outage should not generate thousands of events; a single silent alert could also be missed, so periodic reminders keep it visible. Recovery closes the loop so the admin knows no action is needed anymore.

### 4. Back off reconnect attempts after declared failure
Before declaration: keep the current ~1s retry (fast recovery from blips). After declaration: sleep **30s** between reconnect attempts.

*Why:* once the IP is wrong, reconnecting every second just burns RPi CPU and floods logs. `cv2.VideoCapture` on a dead host can also block for several seconds per attempt.

### 5. Alert sending is best-effort
If the POST to `ALERT_URL` fails (monitor down, network out), log the error and rely on the next reminder tick. No queueing.

*Why:* keeping the detector simple is a project constraint (single-file, minimal deps). If the whole LAN is down, nothing the detector queues can reach the admin anyway; the reminder cadence acts as a natural retry.

### 6. Failure tracking runs only during active hours
The active-hours check precedes frame reading, so outside the window no failures accumulate. `failure_start_time` and reminder timers carry over naturally if an outage spans the window boundary.

## Risks / Trade-offs

- [Grace period too short → false alarms on camera reboots] → default 60s covers a typical Tapo reboot; configurable via env.
- [Monitor unreachable at the same time as camera (whole-house outage)] → alerts are lost until the reminder tick after power returns; acceptable — on recovery either the camera works again or reminders resume.
- [Startup failure path and mid-loop failure path diverge] → unify both through the same failure tracker/alert helper so behavior is identical.
- [rocky-monitor must implement `/api/alerts` before this ships] → detector degrades gracefully (logged POST failure, retried on reminder cadence); coordinate the monitor-side change separately.

## Migration Plan

1. Deploy detector with `ALERT_URL` unset — behavior unchanged except the new backoff logging.
2. Implement alert intake + admin messaging in rocky-monitor.
3. Set `ALERT_URL` in `.env` and restart the container.

Rollback: unset `ALERT_URL`.

## Open Questions

- Exact rocky-monitor endpoint path and auth (if any) — placeholder `/api/alerts`, to be settled when the monitor-side change is proposed.
