# AGENT.md — instructions for an AI agent setting up gazekit

You are setting up **gazekit**, a webcam/iPhone eye tracker, on a fresh
machine. Everything below is executable as-is. Do not invent extra steps.

## One-line init

```sh
bash setup.sh
```

That installs dependencies, fetches the MediaPipe face model, and runs the
automated self-test. If it prints `10/10 passed`, the project is ready.

Manual equivalent (only if `setup.sh` is unavailable):

```sh
uv pip install -r requirements.txt \
  && mkdir -p models data \
  && curl -sL -o models/face_landmarker.task \
     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  && python -m gazekit selftest
```

## Optional: pretrained models from Hugging Face

Public model repo: **https://huggingface.co/ZoneTwelve/gazekit**

Downloads go to `models/pretrained/` — **never** to `data/`, which holds
the user's own trained models. Never overwrite `data/gaze_model*.pkl` or
`data/gaze_cnn.pt` with a download.

```sh
hf download ZoneTwelve/gazekit --local-dir models/pretrained
# or, without the CLI:
mkdir -p models/pretrained && cd models/pretrained
wget https://huggingface.co/ZoneTwelve/gazekit/resolve/main/gaze_cnn.pt
wget https://huggingface.co/ZoneTwelve/gazekit/resolve/main/gaze_model.pkl
```

Use one explicitly without touching the local model:

```sh
python -m gazekit live --model models/pretrained/gaze_model.pkl
```

These weights are **personalized to the author's face, camera and screen**
— a reference artifact, not a working tracker for anyone else. Every user
runs `gazekit auto` to build their own. The training dataset is not public.

## What the user does next

```sh
python -m gazekit auto      # guided: calibrate -> train -> evaluate -> done
python -m gazekit live      # the live gaze dot
```

`auto` is self-contained and terminates on its own. **The built-in
webcam is the default and the only requirement** — an iPhone is optional
and adds a second, more accurate mode. Nothing else is needed for a
working tracker.

macOS will ask for camera permission for the terminal on the first run;
grant it and re-run.

## Facts you need before answering questions about this repo

- **Python**: run everything through the project's interpreter (this repo's
  author uses `uv` with a shared `~/.venv`; `~/.venv/bin/python -m gazekit …`).
  Training is PyTorch (MPS on Apple silicon).
- **Camera sources are separate domains.** A model trained on a webcam
  scores ~1200px on iPhone frames. `gazekit camera cam|app` switches the
  source *and* the model file (`gaze_model.pkl` / `gaze_model_phone.pkl`).
  Never mix them into one model.
- **iPhone mode** additionally needs the app in `ios/` (open
  `ios/GazeTeacher.xcodeproj`, set a signing team, run on a Face ID
  iPhone). It streams frames + ARKit gaze and reconnects on its own; the
  Mac drives it (`gazekit camera status|start|stop`).
- **Never save a model outside a deploy gate.** `calibrate`, `train-cnn`
  and `iterate` each compare a candidate against the incumbent on held-out
  data and keep the winner. A direct save once replaced a 189px model with
  a 1191px one.
- **Evaluation is session-split only** (leave-one-session-out plus the
  aligned live protocol). Random row splits leak — neighbouring frames are
  near-duplicates and score ~2x optimistically.
- **Every CLI run is journaled** to `data/journal.jsonl`; read it with
  `python -m gazekit journal` before assuming what has or hasn't been run.

## Before changing anything

Read `CLAUDE.md` (project rules), `docs/COLLECTION_STANDARD.md` (the
five-stage contract every data-collection mode implements) and
`docs/RELEASE_CHECKLIST.md`. New features follow the existing standard —
consistency is a requirement here, not a preference.

## Verify your work

```sh
python -m gazekit selftest    # must stay 10/10
```
