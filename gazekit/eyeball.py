"""Prototype 3D eyeball gaze model (Wang et al. SIGGRAPH'16 flavor).

Each eyeball is a ~12.5mm-radius sphere rigid in the HEAD frame. We only
have 2D image landmarks plus the facial transformation matrix, so instead
of full 3D reconstruction we use a weak-perspective shortcut: un-rotate the
eye-region landmarks by the head rotation's in-plane 2x2 (which also undoes
yaw/pitch foreshortening approximately) and normalize by the outer-corner
span, giving a head-stabilized 2D frame where the eyeball center (cx, cy)
is fixed. Iris displacement from that center, divided by the anatomical
radius, is the (x, y) part of the gaze direction; direction -> screen is a
per-eye affine map solved with lstsq, and the two eyes are averaged.

Caveat: the app mirrors frames and mediapipe's camera y points up, so axis
signs in the stabilized frame are only consistent, not physical — the
learned affine map absorbs any fixed sign convention.

eye_lm row order (see dataset.EYE_LM_IDX): iris ring right 468-472, iris
ring left 473-477, corners 33,133,362,263, lids 159,145,386,374.
"""

import json
from pathlib import Path

import numpy as np

from .dataset import DWELL_TAGS, load_pruned

EYEBALL_RADIUS_MM = 12.5
INTEROCULAR_MM = 63.0  # adult mean; sets the mm gauge in corner-span units

_R_IRIS, _L_IRIS = slice(0, 5), slice(5, 10)
_CORNERS = slice(10, 14)          # 33, 133, 362, 263
_OUTER = (10, 13)                 # 33, 263: widest head-rigid span


def _stabilize(eye_lm, tmatrix):
    """18x2 image-px landmarks -> head-stabilized, scale-normalized frame.

    Un-rotates by the head rotation's upper-left 2x2 (weak-perspective image
    of the head-frame x/y axes) after flipping image y into camera y, then
    divides by the outer-corner span so interocular distance ~= 1.
    """
    pts = np.asarray(eye_lm, dtype=np.float64)
    A = np.asarray(tmatrix, dtype=np.float64)[:2, :2]
    origin = pts[_CORNERS].mean(axis=0)
    rel = (pts - origin) * [1.0, -1.0]  # image y down -> camera y up
    q = (np.linalg.pinv(A) @ rel.T).T
    span = np.linalg.norm(q[_OUTER[1]] - q[_OUTER[0]])
    return q / max(span, 1e-9)


def _iris_centers(q):
    """(right, left) iris centers in the stabilized frame, ring-averaged."""
    return q[_R_IRIS].mean(axis=0), q[_L_IRIS].mean(axis=0)


class EyeballModel:
    """Per-eye eyeball center + radius in the stabilized frame, plus a
    per-eye affine direction->screen map. Center is estimated as the mean
    iris position over the session (gaze offsets average out when targets
    cover the screen); radius comes from anatomy, 12.5/63 corner-span
    units. With an affine output map center/radius are gauge parameters,
    but keeping them physical leaves room for the sqrt(1-|d|^2) z-term."""

    def __init__(self, screen_size=None):
        self.screen_size = screen_size
        self.centers = None   # (2, 2): right/left eyeball center
        self.radius = EYEBALL_RADIUS_MM / INTEROCULAR_MM
        self.maps = None      # (2, 3, 2): per-eye [dx, dy, 1] -> (x, y)

    def _directions(self, eye_lm, tmatrix):
        r, l = _iris_centers(_stabilize(eye_lm, tmatrix))
        return (np.stack([r, l]) - self.centers) / self.radius

    def fit(self, samples, screen_size):
        """samples: dicts with eye_lm (18x2), tmatrix (4x4), target (px).
        Returns mean training error in px (optimistic; use evaluate())."""
        self.screen_size = tuple(screen_size)
        iris = np.array([_iris_centers(_stabilize(s["eye_lm"], s["tmatrix"]))
                         for s in samples])            # (N, 2 eyes, 2)
        targets = np.array([s["target"] for s in samples], dtype=np.float64)
        self.centers = iris.mean(axis=0)
        dirs = (iris - self.centers) / self.radius
        self.maps = np.empty((2, 3, 2))
        preds = np.empty_like(iris)
        for e in range(2):
            A = np.hstack([dirs[:, e], np.ones((len(dirs), 1))])
            self.maps[e], *_ = np.linalg.lstsq(A, targets, rcond=None)
            preds[:, e] = A @ self.maps[e]
        err = np.linalg.norm(preds.mean(axis=1) - targets, axis=1)
        return float(err.mean())

    def predict(self, eye_lm, tmatrix):
        d = self._directions(eye_lm, tmatrix)
        out = np.mean([np.append(d[e], 1.0) @ self.maps[e] for e in range(2)],
                      axis=0)
        w, h = self.screen_size
        return np.clip(out, [0, 0], [w - 1, h - 1])


def load_training_samples(dataset_root="data/dataset"):
    """Yield dwell-quality records that carry the raw eye_lm + tmatrix
    fields (older sessions predate them), augmented with "session" and
    "screen_size". Skips closed/pursuit tags and pruned ids."""
    root = Path(dataset_root)
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
            if (rec.get("tag") not in DWELL_TAGS or rec.get("i") in bad
                    or rec.get("eye_lm") is None or rec.get("tmatrix") is None
                    or screen is None):
                continue
            rec["session"], rec["screen_size"] = sess.name, screen
            yield rec


def evaluate(dataset_root="data/dataset"):
    """Leave-one-session-out mean px error over sessions with raw
    landmarks. Returns the overall mean, or None if no such data yet."""
    by_sess = {}
    for rec in load_training_samples(dataset_root):
        by_sess.setdefault(rec["session"], []).append(rec)
    if not by_sess:
        print("no sessions with raw landmarks yet — collect new data first")
        return None
    if len(by_sess) < 2:
        print(f"only {len(by_sess)} session with raw landmarks — "
              "need >=2 for leave-one-session-out")
        return None
    errs = []
    for name in sorted(by_sess):
        train = [r for s, rs in by_sess.items() if s != name for r in rs]
        model = EyeballModel()
        model.fit(train, train[0]["screen_size"])
        e = [float(np.linalg.norm(
                model.predict(r["eye_lm"], r["tmatrix"]) - np.array(r["target"])))
             for r in by_sess[name]]
        errs.append(float(np.mean(e)))
        print(f"  {name}: {errs[-1]:.0f}px  (n={len(e)})")
    overall = float(np.mean(errs))
    print(f"overall LOSO mean: {overall:.0f}px over {len(errs)} sessions")
    return overall
