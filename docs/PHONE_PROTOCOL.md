# Phone ↔ Mac protocol (GazeTeacher)

The Mac is the server; the phone connects OUT and owns reconnection.
After one tap of Start the app never needs another interaction — it
streams whenever a Mac-side listener exists and quietly retries when it
doesn't.

## Channels

| channel | transport | direction | content |
|---|---|---|---|
| gaze | UDP :5577 | phone → Mac | one JSON per ARKit update (~60 Hz): `t, look[3] (FACE space), face[16], leye[16], reye[16], blinkL, blinkR` (4×4s column-major) |
| frames + control | TCP :5578 | both | length-prefixed messages (4-byte big-endian length + JSON body) |

Frame message: `{"type":"frame","t":<unix>,"jpg":"<base64 JPEG 640w q0.55>"}` at ≤15 fps.
Control messages (Mac → phone):
`{"cmd":"session_start"|"session_stop"|"stream_on"|"stream_off"|"ping"}`.

## Phone-side state machine

- **Auto-arm**: on app launch the control TCP connect loop starts
  immediately — no tap needed. Retry every 2 s forever while foreground.
- `session_start` / the Start button → run ARKit (gaze UDP + frames).
- `session_stop` / the Stop button → pause ARKit; control stays connected.
- `stream_off` pauses FRAMES only (gaze + ARKit keep running);
  `stream_on` resumes. Preview always works while the session runs.
- UI states: `armed (waiting for Mac)` / `connected` / `tracking` /
  `paused-by-server`.

## Mac-side roles

- `PhoneCamera` (`--camera phone`): accepts :5578, decodes frames,
  auto-orients, logs gaze in-process. May send control messages on the
  accepted socket.
- `PhoneCamera` sends `session_start` when it opens and `session_stop`
  when it releases — running any `--camera phone` command drives the
  phone with zero taps.
- `gazekit camera app|cam`: sets the default camera source
  (`data/config.json`; an explicit `--camera` always overrides).
- `gazekit camera status`: detection — listens, reports whether the phone
  connects and its frame/gaze rates.
- `gazekit camera start|stop`: remote-control the ARKit session.
- `gazekit camera on|off`: pause/resume frames only.

## Invariants

- Reconnection is ALWAYS the phone's job; Mac listeners come and go per
  command with no coordination.
- Ports are fixed (5577/5578); one Mac listener at a time per port —
  a second binder gets a clear "already running" error.
- Any new message type must be additive (unknown `type`/`cmd` ignored).
