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
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .quality import observation_quality, quality_weight, record_quality_score

# eye-region landmark indices logged raw for future 3D-eyeball fitting:
# iris rings (468-477), corners (33,133,362,263), lids (159,145,386,374)
EYE_LM_IDX = list(range(468, 478)) + [33, 133, 362, 263, 159, 145, 386, 374]
DATA_SCHEMA_VERSION = 2
FEATURE_SCHEMA = "ridge-raw14"


def camera_source():
    """Which camera the current run uses — phone frames and webcam frames
    are DIFFERENT DOMAINS (a webcam-trained model scored 1227px on phone
    probes), so models and evaluation are kept per source."""
    try:
        return json.loads(CONFIG_PATH.read_text()).get("camera", "0")
    except (OSError, json.JSONDecodeError):
        return "0"


CONFIG_PATH = Path("data/config.json")


def model_path_for(source=None, base="data/gaze_model"):
    src = source if source is not None else camera_source()
    return f"{base}.pkl" if src != "phone" else f"{base}_phone.pkl"


class DatasetWriter:
    def __init__(self, root: str | Path, screen_size: tuple[int, int]):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / f"session_{stamp}"
        self.crops = self.dir / "crops"
        self.crops.mkdir(parents=True, exist_ok=True)
        self._f = open(self.dir / "samples.jsonl", "w")
        self._n = 0
        self.camera = camera_source()
        self._f.write(json.dumps({
            "meta": True,
            "schema_version": DATA_SCHEMA_VERSION,
            "session_id": self.dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "screen_size": list(screen_size),
            "camera": self.camera,
            "feature_schema": FEATURE_SCHEMA,
        }) + "\n")

    def save_context(self, frame_bgr):
        """One full-frame snapshot per session for offline environment
        annotation (`gazekit annotate`). Stored OUTSIDE data/dataset/ so it
        never ends up in the published dataset tar — full-face frames are
        far more identifying than 64x48 eye crops."""
        ctx = self.dir.parent.parent / "context"
        ctx.mkdir(parents=True, exist_ok=True)
        path = ctx / f"{self.dir.name}.jpg"
        if not path.exists():
            h, w = frame_bgr.shape[:2]
            scale = 640 / max(w, 1)
            small = cv2.resize(frame_bgr, (640, int(h * scale)))
            cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 88])

    def add(self, obs, target_xy: tuple[float, float], tag: str = "calib"):
        if obs.eye_crops is None:
            return
        i = self._n
        self._n += 1
        cv2.imwrite(str(self.crops / f"{i:06d}_R.png"), obs.eye_crops[0])
        cv2.imwrite(str(self.crops / f"{i:06d}_L.png"), obs.eye_crops[1])
        quality = observation_quality(obs)
        record = {
            "i": i, "tag": tag,
            "target": [float(target_xy[0]), float(target_xy[1])],
            "features": [float(v) for v in obs.features],
            "yaw": float(obs.yaw), "pitch": float(obs.pitch),
            "roll": float(obs.roll), "blink": float(obs.blink),
            "brightness": round(float(obs.brightness), 1),
            "t": round(time.time(), 3),
            "interocular_px": round(float(obs.interocular_px), 2),
            # raw eye-region landmarks + head transform: required to fit the
            # 3D eyeball model later (14-dim features are too reduced)
            "eye_lm": (np.round(obs.landmarks_px[EYE_LM_IDX], 2).tolist()
                       if obs.landmarks_px is not None else None),
            "tmatrix": (np.round(obs.extras["tmatrix"], 5).tolist()
                        if "tmatrix" in obs.extras else None),
            "quality_score": quality["quality_score"],
            "quality_components": quality["quality_components"],
        }
        if quality["frame_size"] is not None:
            record["frame_size"] = quality["frame_size"]
        self._f.write(json.dumps(record) + "\n")

    def close(self):
        self._f.close()
        if self._n == 0:  # don't leave empty session dirs around
            import shutil
            shutil.rmtree(self.dir, ignore_errors=True)
        return self._n


DWELL_TAGS = {"calib", "probe", "posture", "edges", "click", "repair", "vor",
              "ambient", "mouse"}


def load_pruned(root: str | Path) -> dict[str, set]:
    """Sample ids flagged bad by `gazekit iterate` — {session_name: {ids}}."""
    try:
        with open(Path(root) / "pruned.json") as f:
            return {k: set(v) for k, v in json.load(f).items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


RECENCY_DECAY = 1.0     # sweep verdict: with session auto-alignment doing
                        # the drift correction, recency decay only hurts
                        # (val aligned: decay 1.0 = 205px, 0.75 = 223px)
LOW_TRUST_WEIGHT = 0.6  # single-source labels: click / ambient / mouse


def load_dwell_features(root: str | Path, last_n: int = 4):
    """(X, Y, w) of dwell-quality samples from the newest sessions — the
    ridge training base. w downweights older sessions (drift) and low-trust
    tags. Pursuit samples are excluded (labels carry smooth-pursuit lag
    noise; fine for the CNN, not for ridge)."""
    root = Path(root)
    pruned = load_pruned(root)
    X, Y, W = [], [], []
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
            trust = (LOW_TRUST_WEIGHT
                     if rec["tag"] in ("click", "ambient", "mouse") else 1.0)
            W.append(trust * quality_weight(record_quality_score(rec))
                     * RECENCY_DECAY ** used)
        if len(X) > n_before:  # only sessions that contributed count
            used += 1
            if used >= last_n:
                break
    if not X:
        return None, None, None
    return np.array(X), np.array(Y), np.array(W)


def load_sessions(root: str | Path, with_session: bool = False,
                  exclude_tags: tuple = ()):
    """Yield (right_crop, left_crop, head_feats, target_norm) across all
    sessions; with_session=True prepends the session name (for honest
    session-held-out validation splits)."""
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
            if (rec.get("tag") == "closed" or rec.get("i") in bad
                    or rec.get("tag") in exclude_tags):
                continue  # no valid gaze label / flagged / excluded
            r = cv2.imread(str(sess / "crops" / f"{rec['i']:06d}_R.png"),
                           cv2.IMREAD_GRAYSCALE)
            l = cv2.imread(str(sess / "crops" / f"{rec['i']:06d}_L.png"),
                           cv2.IMREAD_GRAYSCALE)
            if r is None or l is None or screen is None:
                continue
            head = np.array([rec["yaw"], rec["pitch"], rec["roll"]], dtype=np.float32)
            tgt = np.array([rec["target"][0] / screen[0],
                            rec["target"][1] / screen[1]], dtype=np.float32)
            if with_session:
                yield sess.name, r, l, head, tgt
            else:
                yield r, l, head, tgt
