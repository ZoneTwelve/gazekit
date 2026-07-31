# gazekit — webcam eye tracking with a real training process

Tracks where you look on screen using your laptop camera **or iPhone**
(via Continuity Camera — the iPhone shows up as a normal camera on macOS).

Two gaze backends:

| backend | how it works | when |
|---|---|---|
| `ridge` (default) | MediaPipe iris landmarks → binocular + head-pose features → linear ridge (alpha via leave-one-target-out CV) | calibrates in ~2 min, real-time on CPU |
| `cnn` | MobileNetV2 (ImageNet-pretrained) post-trained on **your** eye crops | after 2–3 calibration sessions, best quality |

Every calibration session also records eye crops + targets into
`data/dataset/`, so the CNN gets more personal training data each time.

## Setup

```sh
uv pip install --python ~/.venv/bin/python -r requirements.txt
mkdir -p models && curl -sL -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

First run will ask for camera permission for your terminal
(System Settings → Privacy & Security → Camera).

## Usage

```sh
PY=~/.venv/bin/python

$PY -m gazekit auto               # << start here: guided end-to-end —
                                  #    collects whatever is missing, trains,
                                  #    evaluates, then keeps adapting in
                                  #    ambient mode. Steps + timings logged
                                  #    to data/auto_log.jsonl
$PY -m gazekit cameras            # list cameras (iPhone appears here too)
$PY -m gazekit doctor             # check lighting / distance / pose
$PY -m gazekit calibrate          # full training process (~90 s)
$PY -m gazekit live               # live gaze dot (ridge backend)

# after 2–3 calibrate sessions:
$PY -m gazekit train-cnn          # fine-tune MobileNetV2 on your data (MPS)
$PY -m gazekit live --backend cnn
```

Live-mode: **click anywhere to teach** (you look where you click — each
click refits the model instantly), `r` = 1-point drift recenter,
`c` = camera preview, `q` = quit.

## Growing the dataset (better stability over time)

```sh
$PY -m gazekit collect posture    # same grid at 3 sitting positions
                                  #   -> fixes "I moved and it broke"
$PY -m gazekit collect edges      # corners/edges the normal grid misses
$PY -m gazekit collect vor        # eyes locked on a fixed dot while you
                                  #   move your head -> head-pose robustness
$PY -m gazekit collect blinks     # open/closed-eye calibration -> tunes
                                  #   blink freezing to YOUR eyes
$PY -m gazekit collect pursuit    # smooth-pursuit sweep, whole-screen
                                  #   coverage (CNN training only)
```

`posture`/`edges` immediately refit the ridge model on the newest sessions;
all scenarios (plus live-mode clicks) feed the CNN dataset for `train-cnn`.

## Verifying accuracy with the mouse

```sh
$PY -m gazekit verify              # roam: look AT your cursor, live error HUD
$PY -m gazekit verify --mode path  # follow a wide guided track to 100%
$PY -m gazekit verify --teach      # also save samples for retraining
```

Ground truth = your cursor (lag-compensated); shows live + per-region
error and appends a summary to `data/verify_log.jsonl`.

## Dataset lifecycle

```sh
$PY -m gazekit iterate            # clean -> train -> validate -> evaluate
                                  #   -> update; run after every collection
$PY -m gazekit iterate --cnn      # also retrain the CNN on cleaned data
```

Cleaning is non-destructive (bad sample ids go to `data/dataset/pruned.json`
and all loaders skip them). Validation is leave-one-session-out (cross-
session drift) plus leave-one-target-out (interpolation). The deployed
model is only replaced when the candidate wins on the newest held-out
session. History accumulates in `data/eval_history.jsonl`.

## Ambient trainer (runs while you work)

```sh
$PY -m gazekit ambient        # leave running in a spare terminal
```

Every 15–45 s (tune with `--min-wait/--max-wait`) a small click-through
dot pops up over whatever you're doing — glance at it and it disappears (~2 s). 40% of popups re-test old
calibration points and log accuracy drift to `data/ambient_log.jsonl`
(macOS notification if it degrades badly); the rest are new training
points that refit the model on the fly. Popups you ignore are detected
and discarded.

## The training (calibration) process

1. **Environment gate** — won't start until face detection, lighting,
   camera distance, and head pose are all good and held for 2 s.
2. **Point collection** — 4x4 grid × 2 rounds, shuffled; between rounds you
   are asked to shift posture slightly so the model learns to compensate
   drift. Each point: settle animation (frames discarded), then 40 samples
   gated on blink < 0.35 and head pose < 22°. Too few good samples → point
   auto-redone.
3. **Outlier rejection** — per point, samples > 3.5 MAD from the median
   iris features are dropped (micro-saccades, tracker glitches).
4. **Train + honest validation** — linear ridge fit (regularization chosen
   by leave-one-target-out CV), then 6 *fresh* probe points the model never
   saw measure true accuracy.
5. **Auto-repair + final refit** — marginal accuracy triggers extra points
   at the worst regions and a retrain; the saved model is refit including
   the probe samples.

Verdict thresholds: **STABLE** ≤ 4.5 % of screen diagonal,
**USABLE** ≤ 7.5 %, else recalibrate. A report is saved next to the model
(`data/gaze_model.report.json`).

## Stability tips

- Recalibrate when you change chair height, lighting, or camera position.
- Use `r` (recenter) in live mode instead of a full recalibration for drift.
- More calibration sessions → better CNN. Vary lighting slightly across
  sessions so the CNN learns robustness.
