"""Gaze regression: landmark features -> screen coordinates (ridge baseline).

Strictly LINEAR model. Degree-2 polynomial features memorized the calibration
grid (train 11px / validation 197px on real data); linear features with the
binocular transform below cross-validate ~2x better between targets.

Regularization strength is chosen by leave-one-TARGET-out CV: samples within
one calibration point are heavily correlated, so per-sample CV picks alphas
that overfit.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def transform(X: np.ndarray) -> np.ndarray:
    """Raw 14-dim tracker features -> 11-dim gaze features.

    Binocular mean averages out per-eye landmark noise; vergence keeps the
    horizontal/depth information the mean discards; openness carries most of
    the vertical signal; pose+translation let the model compensate posture
    drift between sessions of sitting.
    """
    X = np.atleast_2d(X)
    R, L = X[:, 0:4], X[:, 4:8]
    bino = (R[:, :2] + L[:, :2]) / 2
    verg = R[:, :2] - L[:, :2]
    opn = (R[:, 3:4] + L[:, 3:4]) / 2
    return np.hstack([bino, verg, opn, X[:, 8:11], X[:, 11:14]])


class GazeModel:
    def __init__(self, screen_size: tuple[int, int]):
        self.screen_size = screen_size
        self.pipe = None
        self.alpha = None
        self.bias = np.zeros(2)  # runtime drift correction, not persisted

    @staticmethod
    def _make(alpha: float):
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X: np.ndarray, Y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> float:
        """Fit with leave-one-target-out alpha selection. Returns the CV
        error in pixels (honest, unlike training RMSE). sample_weight lets
        callers downweight stale sessions / low-trust labels."""
        Xt = transform(X)
        w = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
        targets = np.unique(Y, axis=0)
        best = (None, np.inf)
        for alpha in ALPHAS:
            errs = []
            for t in targets:
                m = (Y == t).all(axis=1)
                if m.all() or not m.any():
                    continue
                pipe = self._make(alpha)
                kw = {"ridge__sample_weight": w[~m]} if w is not None else {}
                pipe.fit(Xt[~m], Y[~m], **kw)
                pred = np.median(pipe.predict(Xt[m]), axis=0)
                errs.append(float(np.hypot(*(pred - t))))
            cv = float(np.mean(errs)) if errs else np.inf
            if cv < best[1]:
                best = (alpha, cv)
        self.alpha = best[0] if best[0] is not None else 10.0
        self.pipe = self._make(self.alpha)
        kw = {"ridge__sample_weight": w} if w is not None else {}
        self.pipe.fit(Xt, Y, **kw)
        return best[1]

    def refit(self, X: np.ndarray, Y: np.ndarray,
              sample_weight: np.ndarray | None = None):
        """Fast refit reusing the already-selected alpha (for live updates)."""
        self.pipe = self._make(self.alpha if self.alpha is not None else 10.0)
        kw = ({"ridge__sample_weight": np.asarray(sample_weight, dtype=float)}
              if sample_weight is not None else {})
        self.pipe.fit(transform(X), Y, **kw)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        out = self.pipe.predict(transform(feats))[0] + self.bias
        w, h = self.screen_size
        return np.clip(out, [0, 0], [w - 1, h - 1])

    def save(self, path: str | Path, report: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipe": self.pipe, "screen_size": self.screen_size,
                         "alpha": self.alpha}, f)
        if report is not None:
            meta = dict(report)
            meta["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(path.with_suffix(".report.json"), "w") as f:
                json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GazeModel":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        m = cls(tuple(blob["screen_size"]))
        m.pipe = blob["pipe"]
        m.alpha = blob.get("alpha")
        return m
