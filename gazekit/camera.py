"""Camera open/enumeration. On macOS the iPhone appears as a regular
AVFoundation device via Continuity Camera, so index selection covers both."""

import json
import subprocess
import sys

import cv2


def list_cameras(max_index: int = 5) -> list[dict]:
    names = []
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["system_profiler", "SPCameraDataType", "-json"],
                capture_output=True, text=True, timeout=15).stdout
            names = [c.get("_name", "?") for c in
                     json.loads(out).get("SPCameraDataType", [])]
        except Exception:
            pass
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                found.append({
                    "index": i,
                    "name": names[i] if i < len(names) else "unknown",
                    "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                })
        cap.release()
    return found


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {index}. If this is the first run, grant "
            "camera permission to your terminal in System Settings > Privacy "
            "& Security > Camera, then retry.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def read_mirrored(cap) -> "cv2.typing.MatLike | None":
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.flip(frame, 1)
