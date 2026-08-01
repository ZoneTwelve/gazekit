#!/usr/bin/env bash
# gazekit one-line init. Usage: bash setup.sh
set -euo pipefail

MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Python: prefer an existing shared venv, then uv, then whatever is on PATH
if [ -x "$HOME/.venv/bin/python" ]; then
  PY="$HOME/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  uv venv .venv >/dev/null
  PY="$PWD/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi
echo "python: $PY"

echo "installing dependencies..."
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" -r requirements.txt -q
else
  "$PY" -m pip install -q -r requirements.txt
fi

mkdir -p models data
# default to the built-in webcam — an iPhone is optional, so a fresh
# install must work with nothing but the laptop's own camera
if [ ! -f data/config.json ]; then
  echo '{"camera": "0"}' > data/config.json
fi
if [ ! -s models/face_landmarker.task ]; then
  echo "fetching the MediaPipe face model (~4 MB)..."
  curl -fsSL -o models/face_landmarker.task "$MODEL_URL"
fi

echo "running self-test..."
"$PY" -m gazekit selftest

cat <<EOF

ready. next:
  $PY -m gazekit auto     # guided calibrate -> train -> evaluate
  $PY -m gazekit live     # live gaze dot
EOF
