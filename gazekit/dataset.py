"""Personal gaze dataset recorder: eye crops + features + targets on disk.

Every calibration/validation session appends here, so the CNN has more data
each time you calibrate. Layout:

    data/dataset/
        session_YYYYmmdd_HHMMSS/
            samples.jsonl      one line per sample (metadata + target)
            crops/000123_R.png, 000123_L.png
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np


class DatasetWriter:
    def __init__(self, root: str | Path, screen_size: tuple[int, int]):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / f"session_{stamp}"
        self.crops = self.dir / "crops"
        self.crops.mkdir(parents=True, exist_ok=True)
        self._f = open(self.dir / "samples.jsonl", "w")
        self._n = 0
        self._f.write(json.dumps({"meta": True, "screen_size": list(screen_size)}) + "\n")

    def add(self, obs, target_xy: tuple[float, float], tag: str = "calib"):
        if obs.eye_crops is None:
            return
        i = self._n
        self._n += 1
        cv2.imwrite(str(self.crops / f"{i:06d}_R.png"), obs.eye_crops[0])
        cv2.imwrite(str(self.crops / f"{i:06d}_L.png"), obs.eye_crops[1])
        self._f.write(json.dumps({
            "i": i, "tag": tag,
            "target": [float(target_xy[0]), float(target_xy[1])],
            "features": [float(v) for v in obs.features],
            "yaw": float(obs.yaw), "pitch": float(obs.pitch),
            "roll": float(obs.roll), "blink": float(obs.blink),
        }) + "\n")

    def close(self):
        self._f.close()
        if self._n == 0:  # don't leave empty session dirs around
            import shutil
            shutil.rmtree(self.dir, ignore_errors=True)
        return self._n


DWELL_TAGS = {"calib", "probe", "posture", "edges", "click", "repair", "vor",
              "ambient"}


def load_pruned(root: str | Path) -> dict[str, set]:
    """Sample ids flagged bad by `gazekit iterate` — {session_name: {ids}}."""
    try:
        with open(Path(root) / "pruned.json") as f:
            return {k: set(v) for k, v in json.load(f).items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_dwell_features(root: str | Path, last_n: int = 4):
    """(X, Y) of dwell-quality samples from the newest sessions — the ridge
    training base for live-mode refits. Pursuit samples are excluded (their
    labels carry smooth-pursuit lag noise; fine for the CNN, not for ridge)."""
    root = Path(root)
    pruned = load_pruned(root)
    X, Y = [], []
    used = 0
    for sess in sorted(root.glob("session_*"), reverse=True):
        jl = sess / "samples.jsonl"
        if not jl.exists():
            continue
        bad = pruned.get(sess.name, set())
        n_before = len(X)
        for line in open(jl):
            rec = json.loads(line)
            if (rec.get("meta") or rec.get("tag") not in DWELL_TAGS
                    or rec.get("i") in bad):
                continue
            X.append(rec["features"])
            Y.append(rec["target"])
        if len(X) > n_before:  # only sessions that contributed count
            used += 1
            if used >= last_n:
                break
    if not X:
        return None, None
    return np.array(X), np.array(Y)


def load_sessions(root: str | Path):
    """Yield (right_crop, left_crop, head_feats, target_norm) across all sessions."""
    root = Path(root)
    pruned = load_pruned(root)
    for sess in sorted(root.glob("session_*")):
        jl = sess / "samples.jsonl"
        if not jl.exists():
            continue
        bad = pruned.get(sess.name, set())
        screen = None
        for line in open(jl):
            rec = json.loads(line)
            if rec.get("meta"):
                screen = rec["screen_size"]
                continue
            if rec.get("tag") == "closed" or rec.get("i") in bad:
                continue  # no valid gaze label / flagged by iterate
            r = cv2.imread(str(sess / "crops" / f"{rec['i']:06d}_R.png"),
                           cv2.IMREAD_GRAYSCALE)
            l = cv2.imread(str(sess / "crops" / f"{rec['i']:06d}_L.png"),
                           cv2.IMREAD_GRAYSCALE)
            if r is None or l is None or screen is None:
                continue
            head = np.array([rec["yaw"], rec["pitch"], rec["roll"]], dtype=np.float32)
            tgt = np.array([rec["target"][0] / screen[0],
                            rec["target"][1] / screen[1]], dtype=np.float32)
            yield r, l, head, tgt
