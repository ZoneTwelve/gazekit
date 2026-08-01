"""Live gaze dot with online learning.

Hotkeys / mouse:
  q/Esc  quit
  r      recenter (look at the ring for 1 s) — quick bias fix after moving
  c      toggle camera thumbnail
  CLICK  teach: you look where you click, so every click becomes a labeled
         sample; the ridge model refits immediately and the sample is saved
         for CNN training.

Blink handling: freeze with hysteresis (on at blink>0.28 or eyelid collapse,
off at blink<0.18) plus a 250 ms hold after reopening — iris landmarks are
garbage while the lid moves, which is what used to make the dot fly away.
"""

import time
from collections import deque

import cv2
import numpy as np

from . import ui
from .camera import open_camera, read_mirrored
from .dataset import DatasetWriter, load_dwell_features
from .model import GazeModel
from .tracker import FaceTracker

BLINK_ON = 0.28        # freeze when blink score rises above this
BLINK_OFF = 0.18       # unfreeze only when it falls below this
OPEN_MIN = 0.16        # eyelid openness collapse -> freeze (faster signal)
REOPEN_HOLD_S = 0.25   # stay frozen briefly after the eye reopens
CLICK_WEIGHT = 20      # one click ~ half a calibration dwell point


class BlinkGate:
    def __init__(self, profile_path="data/blink_profile.json"):
        self.frozen = False
        self._reopened_at = None
        self.on, self.off, self.open_min = BLINK_ON, BLINK_OFF, OPEN_MIN
        try:
            import json
            with open(profile_path) as f:
                p = json.load(f)
            self.on, self.off = p["blink_on"], p["blink_off"]
            self.open_min = p["open_min"]
            print(f"blink profile loaded: on={self.on} off={self.off} "
                  f"open_min={self.open_min}")
        except FileNotFoundError:
            pass  # generic thresholds; run `collect blinks` to personalize

    def update(self, obs) -> bool:
        if not obs.ok:
            self.frozen = True
            self._reopened_at = None
            return True
        openness = min(obs.features[3], obs.features[7])
        closed = obs.blink > self.on or openness < self.open_min
        opened = obs.blink < self.off and openness > self.open_min * 1.15
        now = time.monotonic()
        if closed:
            self.frozen = True
            self._reopened_at = None
        elif self.frozen and opened:
            if self._reopened_at is None:
                self._reopened_at = now
            elif now - self._reopened_at > REOPEN_HOLD_S:
                self.frozen = False
        return self.frozen


def _recenter(win, cap, tracker, predict, current_bias):
    """1-point drift correction: median prediction error at screen center."""
    cx, cy = win.w / 2, win.h / 2
    preds = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.2:
        frame = read_mirrored(cap)
        if frame is None:
            continue
        obs = tracker.process(frame, want_crops=True)
        img = win.canvas()
        ui.draw_target(img, cx, cy, (time.monotonic() - t0) / 1.2, ui.GOOD)
        ui.center_text(img, "look at the ring", int(win.h * 0.4), 0.8)
        win.show(img)
        if obs.ok and obs.blink < 0.35 and time.monotonic() - t0 > 0.4:
            p = predict(obs)
            if p is not None:
                preds.append(p)
    if len(preds) >= 5:
        med = np.median(np.array(preds), axis=0)
        return current_bias + np.array([cx, cy]) - med
    return current_bias


def _quick_align(win, cap, tracker, predict):
    """3-point per-axis gain+offset correction on top of the frozen model —
    kills session-to-session drift (posture, chair height, camera nudge)
    in ~8 s. Returns (ax, bx, ay, by): corrected = a * pred + b."""
    pts = [(0.5 * win.w, 0.5 * win.h), (0.22 * win.w, 0.25 * win.h),
           (0.78 * win.w, 0.75 * win.h)]
    P, T = [], []
    for pt in pts:
        preds = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.5:
            frame = read_mirrored(cap)
            if frame is None:
                continue
            obs = tracker.process(frame, want_crops=True)
            img = win.canvas()
            ui.draw_target(img, *pt, (time.monotonic() - t0) / 0.6, ui.GOOD)
            ui.center_text(img, "quick alignment — look at the ring",
                           int(win.h * 0.4), 0.8)
            win.show(img)
            if obs.ok and obs.blink < 0.3 and time.monotonic() - t0 > 0.6:
                p = predict(obs)
                if p is not None:
                    preds.append(p)
        if len(preds) >= 5:
            P.append(np.median(np.array(preds), axis=0))
            T.append(pt)
    if len(P) < 2:
        return (1.0, 0.0, 1.0, 0.0)
    P, T = np.array(P), np.array(T)
    out = []
    for ax in (0, 1):
        var = float(np.var(P[:, ax]))
        if var < 1e-6:
            out += [1.0, float(np.mean(T[:, ax]) - np.mean(P[:, ax]))]
            continue
        a = float(np.cov(P[:, ax], T[:, ax], bias=True)[0, 1] / var)
        a = float(np.clip(a, 0.5, 1.8))
        b = float(np.mean(T[:, ax]) - a * np.mean(P[:, ax]))
        out += [a, b]
    return tuple(out)


def run(camera_index=0, backend="ridge", model_path=None,
        landmarker="models/face_landmarker.task", screen=None,
        dataset_root="data/dataset", align=True):
    from .screen import screen_size
    from .filters import GazeSmoother
    sw, sh = screen or screen_size()

    ridge = cnn = eyeball = None
    if backend in ("cnn", "hybrid"):
        from .cnn import CnnPredictor
        cnn = CnnPredictor("data/gaze_cnn.pt" if backend == "hybrid"
                           else (model_path or "data/gaze_cnn.pt"), (sw, sh))
    if backend in ("ridge", "hybrid"):
        ridge = GazeModel.load("data/gaze_model.pkl" if backend == "hybrid"
                               else (model_path or "data/gaze_model.pkl"))
    if backend == "eyeball":
        from .eyeball import EyeballPredictor
        eyeball = EyeballPredictor((sw, sh))

    # offline ensemble sweep: blend error is U-shaped in alpha with the
    # minimum at ~0.4 ridge / 0.6 cnn — better than either model alone
    HYBRID_RIDGE_W = 0.4

    def predict(obs):
        if eyeball is not None:
            return eyeball.predict(obs)
        if ridge is not None and cnn is not None:
            pr, pc = ridge.predict(obs.features), cnn.predict(obs)
            return (pr if pc is None
                    else HYBRID_RIDGE_W * pr + (1 - HYBRID_RIDGE_W) * pc)
        if cnn is not None:
            return cnn.predict(obs)
        return ridge.predict(obs.features)

    active = ridge if ridge is not None else (cnn if cnn is not None
                                              else eyeball)

    # base data for click-teach refits (ridge backend only)
    base_X, base_Y, base_w = (None, None, None)
    click_X, click_Y = [], []
    if ridge is not None:
        base_X, base_Y, base_w = load_dwell_features(dataset_root)
    writer = DatasetWriter(dataset_root, (sw, sh))
    recent = deque(maxlen=8)  # (t, obs) buffer for click labeling

    win = ui.FullscreenWindow("gazekit-live", (sw, sh))
    clicks = []
    cv2.setMouseCallback(win.name,
                         lambda ev, x, y, flags, param:
                         clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN
                         else None)
    cap = open_camera(camera_index)
    tracker = FaceTracker(landmarker)
    smoother = GazeSmoother()
    gate = BlinkGate()
    show_thumb = True
    last_xy = (sw / 2, sh / 2)
    flash_until = 0.0
    ax, bx, ay, by = (1.0, 0.0, 1.0, 0.0)

    try:
        if align:
            ax, bx, ay, by = _quick_align(win, cap, tracker, predict)
            print(f"aligned: gain=({ax:.2f},{ay:.2f}) "
                  f"offset=({bx:.0f},{by:.0f})")
        while True:
            frame = read_mirrored(cap)
            if frame is None:
                continue
            obs = tracker.process(frame, want_crops=True)
            now = time.monotonic()
            if obs.ok and obs.blink < 0.25:
                recent.append((now, obs))

            img = win.canvas()
            frozen = gate.update(obs)
            if not frozen:
                p = predict(obs)
                if p is not None:
                    px = min(max(ax * float(p[0]) + bx, 0.0), sw - 1.0)
                    py = min(max(ay * float(p[1]) + by, 0.0), sh - 1.0)
                    last_xy = smoother.apply(px, py, now)
            ui.draw_gaze_dot(img, *last_xy, frozen=frozen)

            # -- click-to-teach ------------------------------------------
            while clicks:
                cx, cy = clicks.pop(0)
                fresh = [o for (t, o) in recent if now - t < 0.6]
                if len(fresh) < 3:
                    continue
                feats = np.median([o.features for o in fresh], axis=0)
                # accidental-click guard: if the gaze estimate is nowhere
                # near the click, you weren't looking there — not a label
                p_now = predict(fresh[-1])
                if p_now is not None and np.hypot(p_now[0] - cx,
                                                  p_now[1] - cy) > 400:
                    flash_until = now + 0.6
                    continue
                writer.add(fresh[-1], (cx, cy), tag="click")
                if ridge is not None:
                    # undo the alignment so the taught label lives in raw
                    # model space (alignment is applied after predict)
                    tx = (cx - bx) / ax if ax else cx
                    ty = (cy - by) / ay if ay else cy
                    click_X.extend([feats] * CLICK_WEIGHT)
                    click_Y.extend([(tx, ty)] * CLICK_WEIGHT)
                    if base_X is not None:
                        ridge.refit(np.vstack([base_X, click_X]),
                                    np.vstack([base_Y, click_Y]),
                                    np.concatenate([base_w,
                                                    np.ones(len(click_X))]))
                    else:
                        ridge.refit(np.array(click_X), np.array(click_Y))
                    ridge.bias = np.zeros(2)
                flash_until = now + 0.6
            if now < flash_until:
                ui.center_text(img, "learned +1", 90, 0.8, ui.GOOD)

            n_clicks = len(click_Y) // CLICK_WEIGHT
            ui.center_text(img,
                           f"q quit   a align   r recenter   c camera   "
                           f"click = teach ({n_clicks} taught)",
                           win.h - 24, 0.55, (110, 110, 110))
            if show_thumb:
                ui.thumbnail(img, frame, obs)

            key = win.show(img)
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                show_thumb = not show_thumb
            if key == ord("r"):
                active.bias = _recenter(win, cap, tracker, predict, active.bias)
            if key == ord("a"):
                ax, bx, ay, by = _quick_align(win, cap, tracker, predict)
    finally:
        if ridge is not None and click_Y:
            ridge.save("data/gaze_model.pkl",
                       {"refit_from": "live-clicks",
                        "clicks": len(click_Y) // CLICK_WEIGHT})
            print(f"saved model updated with {len(click_Y) // CLICK_WEIGHT} "
                  "click samples")
        n = writer.close()
        if n:
            print(f"dataset: {n} click samples appended under {writer.dir}")
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
