"""Ambient trainer: runs while you work. Every 40-110 s a small dot pops up
over whatever you're doing (click-through, always on top). Look at it for
~2 s and it disappears.

  ~40% of popups are OLD calibration points  -> measures real accuracy drift
  ~60% are NEW random points                 -> fresh training data

An attention check (median gaze must land near the dot) discards popups you
ignored. Accepted explore points refit the ridge model every few samples;
validation errors are logged to data/ambient_log.jsonl and a rolling report
prints in the terminal. If accuracy degrades badly you get a macOS
notification suggesting recalibration.

Run it in a spare terminal:  python -m gazekit ambient      (Ctrl+C to stop)
"""

import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory,
                    NSBackingStoreBuffered, NSBezierPath, NSColor, NSMakeRect,
                    NSScreen, NSScreenSaverWindowLevel, NSView, NSWindow,
                    NSWindowStyleMaskBorderless)
from Foundation import NSDate, NSRunLoop

from .camera import open_camera, read_mirrored
from .dataset import DatasetWriter, load_dwell_features
from .model import GazeModel
from .tracker import FaceTracker

DOT = 72                 # dot size, points
DWELL_S = 1.4            # sampling time once the eye has settled
REACT_S = 0.4            # full-screen ring holds: human reaction time
SHRINK_S = 1.0           # ring contracts onto the target
FIX_S = 0.35             # fixation settle before sampling starts
VALIDATE_P = 0.4         # fraction of popups that re-test old points
ATTENTION_RADIUS = 300.0 # median gaze must land this close, else discarded
REFIT_EVERY = 5          # accepted explore points between ridge refits
DEGRADE_PX = 260.0       # rolling validation error that triggers a warning


class _OverlayView(NSView):
    """Full-screen view: a huge ring that contracts onto the target point,
    dragging the user's gaze with it (phase 0 = covers screen, 1 = dot)."""
    tx = 0.0
    ty = 0.0   # bottom-left-origin view coordinates
    phase = 1.0

    def drawRect_(self, rect):
        import math
        b = self.bounds().size
        p = min(max(self.phase, 0.0), 1.0)
        ease = 1.0 - (1.0 - p) ** 3
        # faint veil that fades as the ring contracts
        if p < 1.0:
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                0.0, 0.0, 0.0, 0.22 * (1.0 - p)).set()
            NSBezierPath.fillRect_(self.bounds())
        r0 = max(math.hypot(self.tx - cx, self.ty - cy)
                 for cx in (0.0, b.width) for cy in (0.0, b.height)) + 30.0
        r = r0 * (1.0 - ease) + 15.0 * ease
        NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 0.62, 0.1, 0.95).set()
        ring = NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(self.tx - r, self.ty - r, 2 * r, 2 * r))
        ring.setLineWidth_(3.0 + 5.0 * (1.0 - p))
        ring.stroke()
        NSColor.whiteColor().set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(self.tx - 4, self.ty - 4, 8, 8)).fill()


class OverlayDot:
    """Borderless, transparent, click-through, always-on-top full-screen
    overlay. The show() animation starts screen-sized and shrinks to the
    target so the dot cannot be missed."""

    def __init__(self):
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        app.finishLaunching()  # windows may never display without this
        frame = NSScreen.mainScreen().frame()
        self.screen_h = frame.size.height
        self.win = NSWindow.alloc(
        ).initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, frame.size.width, frame.size.height),
            NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
        self.win.setReleasedWhenClosed_(False)
        self.win.setOpaque_(False)
        self.win.setBackgroundColor_(NSColor.clearColor())
        # screensaver level + fullscreen-auxiliary so the dot floats over
        # everything, including fullscreen browser Spaces
        self.win.setLevel_(NSScreenSaverWindowLevel)
        self.win.setIgnoresMouseEvents_(True)
        self.win.setCollectionBehavior_(
            (1 << 0)      # NSWindowCollectionBehaviorCanJoinAllSpaces
            | (1 << 8))   # NSWindowCollectionBehaviorFullScreenAuxiliary
        self.view = _OverlayView.alloc().initWithFrame_(
            NSMakeRect(0, 0, frame.size.width, frame.size.height))
        self.win.setContentView_(self.view)

    def show(self, x: float, y: float, phase: float):
        """x, y in top-left-origin screen points (our gaze coordinates).
        phase 0 -> ring covers the screen; 1 -> small dot at (x, y)."""
        self.view.tx = float(x)
        self.view.ty = float(self.screen_h - y)
        self.view.phase = phase
        self.view.setNeedsDisplay_(True)
        self.win.orderFrontRegardless()
        self.win.displayIfNeeded()
        self._pump()

    def hide(self):
        self.win.orderOut_(None)
        self._pump()

    @staticmethod
    def _pump():
        NSRunLoop.currentRunLoop().runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(0.01))


def _notify(title, text):
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{text}" with title "{title}"'],
                       capture_output=True, timeout=5)
    except Exception:
        pass


def overlay_test():
    """Visibility check: three popups with the full guide animation.
    Run `gazekit ambient --test` and follow the shrinking rings."""
    import random as _r
    from .screen import screen_size
    from .ui import say
    sw, sh = screen_size()
    dot = OverlayDot()
    for i in range(3):
        target = (_r.uniform(0.1, 0.9) * sw, _r.uniform(0.1, 0.9) * sh)
        say("look")
        t0 = time.monotonic()
        total = REACT_S + SHRINK_S + FIX_S + DWELL_S
        while time.monotonic() - t0 < total:
            t = time.monotonic() - t0
            phase = min(max((t - REACT_S) / SHRINK_S, 0.0), 1.0)
            dot.show(*target, phase)
            time.sleep(0.016)
        dot.hide()
        time.sleep(0.8)
    print("test done — you should have seen 3 rings shrink onto dots. "
          "If not, tell me your monitor setup (built-in / external / "
          "fullscreen app).")


class TargetPolicy:
    """UCB bandit over a 4x4 screen grid: each popup is an arm pull, and the
    reward signal is the measured prediction error there. The policy sends
    popups where they earn the most — high recent error, under-sampled, or
    stale cells — instead of uniformly at random. State persists across runs
    (data/ambient_policy.json) so it keeps learning.

    This is the honest place for RL in a gaze tracker: the regression itself
    has labels (supervised beats RL there); the *sampling policy* doesn't.
    """
    GRID = 4
    C = 0.55          # exploration strength
    EPS = 0.2         # uniform-exploration floor: never tunnel-vision
    EWMA = 0.35       # how fast cell error estimates track new evidence
    STALE_H = 2.0     # hours for the staleness bonus to saturate

    def __init__(self, sw, sh, path="data/ambient_policy.json"):
        self.sw, self.sh = sw, sh
        self.diag = float(np.hypot(sw, sh))
        self.path = Path(path)
        g = self.GRID
        self.n = np.ones((g, g))
        self.err = np.full((g, g), 0.10)   # prior: ~10% of diagonal
        self.last = np.zeros((g, g))
        try:
            s = json.loads(self.path.read_text())
            self.n = np.array(s["n"])
            self.err = np.array(s["err"])
            self.last = np.array(s["last"])
        except (FileNotFoundError, KeyError, ValueError):
            pass

    def _cell(self, x, y):
        g = self.GRID
        return (min(int(y / self.sh * g), g - 1),
                min(int(x / self.sw * g), g - 1))

    def choose(self, rng):
        if rng.random() < self.EPS:
            return rng.randrange(self.GRID), rng.randrange(self.GRID)
        stale = np.clip((time.time() - self.last) / (self.STALE_H * 3600),
                        0.0, 1.0) * 0.05
        ucb = self.err + self.C * np.sqrt(
            np.log(self.n.sum() + 1.0) / self.n) * 0.05 + stale
        ucb = ucb + np.random.uniform(0, 1e-4, ucb.shape)  # tie-break
        gy, gx = np.unravel_index(np.argmax(ucb), ucb.shape)
        return int(gy), int(gx)

    def point_in(self, cell, rng):
        gy, gx = cell
        g = self.GRID
        x = (gx + rng.uniform(0.2, 0.8)) / g * self.sw
        y = (gy + rng.uniform(0.2, 0.8)) / g * self.sh
        return (float(np.clip(x, 20, self.sw - 20)),
                float(np.clip(y, 20, self.sh - 20)))

    def update(self, x, y, err_px):
        gy, gx = self._cell(x, y)
        e = min(err_px / self.diag, 0.5)
        self.err[gy, gx] = (1 - self.EWMA) * self.err[gy, gx] + self.EWMA * e
        self.n[gy, gx] += 1
        self.last[gy, gx] = time.time()
        self.path.write_text(json.dumps({
            "n": self.n.tolist(), "err": np.round(self.err, 4).tolist(),
            "last": self.last.tolist()}))

    def summary(self):
        worst = np.unravel_index(np.argmax(self.err), self.err.shape)
        return (f"policy: worst cell row{worst[0]} col{worst[1]} "
                f"~{self.err[worst] * self.diag:.0f}px, "
                f"{int(self.n.sum())} pulls")


def _history_points(dataset_root, limit=40):
    _, Y, _ = load_dwell_features(dataset_root)
    if Y is None:
        return []
    return [tuple(t) for t in np.unique(Y, axis=0)][:limit]


def _popup(dot, cap, tracker, model, target, voice, cue="look"):
    """One guided popup: animation, then gated sampling.
    Returns (outcome, err, feats, kept):
      outcome 'no-face'  — person not visible during the dwell
              'occluded' — face there but eyes unreadable (blinks/occlusion)
              'looked'   — enough samples; err = median prediction error"""
    if voice:
        from .ui import say
        say(cue)
    total = REACT_S + SHRINK_S + FIX_S
    t0 = time.monotonic()
    while time.monotonic() - t0 < total:
        t = time.monotonic() - t0
        phase = min(max((t - REACT_S) / SHRINK_S, 0.0), 1.0)
        dot.show(*target, phase)
        frame = read_mirrored(cap)
        if frame is not None:
            tracker.process(frame)

    feats, kept = [], []
    n_frames = n_face = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < DWELL_S:
        dot.show(*target, 1.0)
        frame = read_mirrored(cap)
        if frame is None:
            continue
        n_frames += 1
        obs = tracker.process(frame, want_crops=True)
        if obs.ok:
            n_face += 1
            if obs.blink < 0.3:
                feats.append(obs.features)
                kept.append(obs)
    dot.hide()

    if len(feats) < 8:
        if n_face < 0.5 * max(n_frames, 1):
            return "no-face", None, [], []
        return "occluded", None, [], []
    preds = np.array([model.predict(f) for f in feats])
    med = np.median(preds, axis=0)
    err = float(np.hypot(med[0] - target[0], med[1] - target[1]))
    return "looked", err, feats, kept


def run(camera_index=0, model_path=None,
        dataset_root="data/dataset", landmarker="models/face_landmarker.task",
        interval=(15.0, 45.0), screen=None, voice=True):
    from .dataset import model_path_for
    from .screen import screen_size
    model_path = model_path or model_path_for()
    sw, sh = screen or screen_size()
    model = GazeModel.load(model_path)
    tracker = FaceTracker(landmarker)
    def _waiting(elapsed):
        img = win.canvas()
        ui.center_text(img, "connecting to the camera...", int(sh * 0.44), 1.0)
        ui.center_text(img, f"{elapsed:.0f}s — phone: keep GazeTeacher in the "
                       "FOREGROUND, unlocked   (q to quit)", int(sh * 0.51),
                       0.7, (150, 150, 150))
        win.show(img)

    _waiting(0)
    cap = open_camera(camera_index, on_wait=_waiting)
    writer = DatasetWriter(dataset_root, (sw, sh))
    dot = OverlayDot()
    log_path = Path("data/ambient_log.jsonl")
    log_f = open(log_path, "a")

    def log(kind, **kw):
        rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind}
        rec.update(kw)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    base_X, base_Y, base_w = load_dwell_features(dataset_root)
    new_X, new_Y = [], []
    history = _history_points(dataset_root)
    policy = TargetPolicy(sw, sh)
    print(policy.summary())
    suspect_queue = []   # regions where a triage confirmed model error
    anchor_deltas = []   # (target - prediction) from recent looked popups
    val_errors = []
    accepted = tested = 0
    backoff = 1.0
    rng = random.Random()

    print("ambient trainer running — dots will pop up while you work. "
          "Ctrl+C to stop.")
    first = True
    try:
        while True:
            # first dot comes fast so you can see it's working
            wait = (rng.uniform(6.0, 12.0) if first
                    else rng.uniform(*interval) * backoff)
            first = False
            print(f"          next dot in {wait:.0f}s")
            t_end = time.monotonic() + wait

            # idle wait with presence monitoring: don't pop at an empty chair
            last_face = time.monotonic()
            last_check = 0.0
            paused = False
            while True:
                now = time.monotonic()
                if now >= t_end and now - last_face < 6.0:
                    break
                if now - last_check > 2.5:
                    last_check = now
                    frame = read_mirrored(cap)
                    if frame is not None and tracker.process(frame).ok:
                        last_face = now
                        writer.save_context(frame)
                    elif now >= t_end and not paused:
                        paused = True
                        log("paused", reason="no-face")
                        print("          paused — nobody in front of the "
                              "camera; resuming when you're back")
                else:
                    cap.grab()  # keep the driver buffer fresh
                time.sleep(0.15)
            if paused:
                print("          welcome back")

            is_val = bool(history) and rng.random() < VALIDATE_P
            cell = policy.choose(rng)
            if suspect_queue and not is_val:
                bx, by = suspect_queue.pop(0)  # revisit confirmed-bad regions
                target = (float(np.clip(bx + rng.uniform(-80, 80), 20, sw - 20)),
                          float(np.clip(by + rng.uniform(-80, 80), 20, sh - 20)))
            elif is_val:
                # validate the bandit-chosen region using a known old point
                cx, cy = policy.point_in(cell, rng)
                target = min(history,
                             key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            else:
                target = policy.point_in(cell, rng)

            outcome, err, feats, kept = _popup(dot, cap, tracker, model,
                                               target, voice)
            if outcome == "no-face":
                log("skip", reason="absent")
                backoff = min(backoff * 1.5, 6.0)
                continue
            if outcome == "occluded":
                log("skip", reason="eyes-occluded")
                continue

            if err > ATTENTION_RADIUS:
                # triage: is the model wrong, or were you just not looking?
                time.sleep(1.0)
                probe = (rng.uniform(0.35, 0.65) * sw,
                         rng.uniform(0.35, 0.65) * sh)
                o2, e2, _, _ = _popup(dot, cap, tracker, model, probe,
                                      voice, cue="look again")
                if o2 == "looked" and e2 <= ATTENTION_RADIUS:
                    # you clearly engaged the recheck -> first miss was model
                    suspect_queue.append(target)
                    policy.update(*target, err)  # big error boosts that cell
                    log("triage", result="model-suspect",
                        target=[round(target[0]), round(target[1])],
                        error_px=round(err, 1), recheck_px=round(e2, 1))
                    print(f"[triage   ] model suspect at "
                          f"({target[0]:.0f},{target[1]:.0f}) err {err:.0f}px "
                          "— queued for extra training")
                    backoff = 1.0
                else:
                    log("triage", result="user-away")
                    backoff = min(backoff * 2.0, 8.0)
                continue
            backoff = 1.0

            policy.update(*target, err)

            # session auto-anchor: the measured offset from every engaged
            # popup feeds back as a bias correction (Finding 1: per-session
            # vertical bias swings ±170px — this cancels it while you work)
            preds = np.array([model.predict(f) for f in feats])
            med = np.median(preds, axis=0)
            anchor_deltas.append(np.array(target) - med)
            if len(anchor_deltas) >= 3:
                shift = 0.5 * np.median(anchor_deltas, axis=0)
                model.bias = model.bias + shift
                anchor_deltas.clear()
                if float(np.hypot(*shift)) > 15:
                    log("anchor", shift=[round(float(shift[0]), 1),
                                         round(float(shift[1]), 1)])
                    print(f"[anchor   ] bias nudged by "
                          f"({shift[0]:+.0f},{shift[1]:+.0f})px")

            if is_val:
                tested += 1
                val_errors.append(err)
                recent = val_errors[-10:]
                log("validate", target=[round(target[0]), round(target[1])],
                    error_px=round(err, 1))
                print(f"[validate #{tested}] {err:5.0f}px at "
                      f"({target[0]:.0f},{target[1]:.0f})  "
                      f"rolling({len(recent)}): {np.mean(recent):.0f}px")
                if len(recent) >= 5 and np.mean(recent) > DEGRADE_PX:
                    _notify("gazekit", f"accuracy degraded to "
                            f"{np.mean(recent):.0f}px — consider recalibrating")
                    val_errors.clear()
            else:
                accepted += 1
                for o in kept:
                    writer.add(o, target, tag="ambient")
                new_X.extend(feats)
                new_Y.extend([target] * len(feats))
                log("learn", target=[round(target[0]), round(target[1])],
                    n=len(feats))
                print(f"[learn    #{accepted}] +{len(feats)} samples at "
                      f"({target[0]:.0f},{target[1]:.0f})")
                if accepted % REFIT_EVERY == 0 and base_X is not None:
                    model.refit(np.vstack([base_X, new_X]),
                                np.vstack([base_Y, new_Y]),
                                np.concatenate([base_w,
                                                np.ones(len(new_X))]))
                    model.save(model_path, {"refit_from": "ambient",
                                            "explore_points": accepted,
                                            "validations": tested})
                    print(f"          model refit + saved "
                          f"({len(base_X) + len(new_X)} samples)")
                    history = _history_points(dataset_root)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        log_f.close()
        n = writer.close()
        if n:
            print(f"dataset: {n} ambient samples appended under {writer.dir}")
        tracker.close()
        cap.release()
