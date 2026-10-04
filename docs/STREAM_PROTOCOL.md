# Gaze stream protocol (`gazekit stream`)

How other apps consume live gaze. gazekit is the producer; consumers
(e.g. GOZ's adaptive player) only listen — there is no handshake.

## Producer

`python -m gazekit stream [--camera N] [--backend ridge] [--port 5590] [--no-align]`

1. Loads the deployed model for the camera source (same predictor as `live`).
2. Runs the 3-point quick alignment in a brief fullscreen window, then
   closes it — the consumer app needs the screen.
3. Publishes one datagram per camera frame (~30 Hz) until Ctrl+C.
4. Returns `{samples, blinks, valid_frac}` so the journal records the run.

## Datagram (UDP, 127.0.0.1:5590, one JSON object)

| field | meaning |
|---|---|
| `t` | producer wall clock, unix seconds (same machine as consumer → same clock) |
| `x`, `y` | smoothed gaze in macOS screen **points**, top-left origin |
| `sw`, `sh` | screen size in points |
| `valid` | `false` while blinking / face lost — `x,y` then hold the last value |
| `blink` | eye currently closed (BlinkGate hysteresis, same as `live`) |
| `blink_score` | raw max eyeBlink blendshape 0..1 |
| `yaw`, `pitch`, `roll` | head direction, degrees |
| `face` | a face was found in this frame |

Browser mapping: a page converts `x,y` to element coordinates with
`window.screenX/Y` + `(outerHeight - innerHeight)` + the element's
`getBoundingClientRect()` at 100% zoom; fullscreen windows make the chrome
offset zero.

## Invariants

- Fields are additive only; consumers ignore unknown keys.
- One producer per port; the consumer owns the port binding.
- No model is saved or modified by streaming.
