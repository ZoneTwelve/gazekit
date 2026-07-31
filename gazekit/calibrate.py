"""Calibration = the training process. Five gated stages:

  1. Environment gate   — refuses to calibrate until lighting/distance/pose
                          are good and held stable for 2 s.
  2. Point collection   — N grid points x R rounds, shuffled. Per point:
                          settle animation (frames discarded), then samples
                          collected with blink/pose gating; too few good
                          samples -> the point is automatically redone.
  3. Outlier rejection  — per point, samples > 3.5 MAD from the median
                          feature vector are dropped (micro-saccades, drift).
  4. Train + validate   — ridge fit, then 6 fresh probe points measure true
                          accuracy (points the model has never seen).
  5. Auto-repair        — if validation error is marginal, extra calibration
                          points are collected near the worst regions and the
                          model is retrained once, then revalidated.

Every accepted sample is also written to the on-disk dataset for CNN
post-training (see dataset.py / cnn.py).
"""

import random
import time

import cv2
import numpy as np

from . import ui
from .camera import open_camera, read_mirrored
from .dataset import DatasetWriter
from .model import GazeModel
from .tracker import FaceTracker

MARGIN = 0.07          # grid margin as fraction of screen
SETTLE_S = 0.55        # target animation time, frames discarded
SAMPLES_PER_POINT = 40
MIN_GOOD = 22
POINT_TIMEOUT_S = 4.0
MAX_RETRIES = 2
BLINK_MAX = 0.35
POSE_MAX_DEG = 22.0
PASS_FRAC = 0.045      # mean validation error <= 4.5% of diagonal -> stable
MARGINAL_FRAC = 0.075


def grid_points(w, h, n=16):
    if n >= 16:  # 4x4: extra row/column mainly buys vertical resolution
        xs = np.linspace(MARGIN * w, (1 - MARGIN) * w, 4)
        ys = np.linspace(MARGIN * h, (1 - MARGIN) * h, 4)
        return [(x, y) for y in ys for x in xs]
    xs = np.linspace(MARGIN * w, (1 - MARGIN) * w, 3)
    ys = np.linspace(MARGIN * h, (1 - MARGIN) * h, 3)
    pts = [(x, y) for y in ys for x in xs]
    if n >= 13:  # add inner diagonal points for edge-to-center coverage
        for fx, fy in ((0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)):
            pts.append((fx * w, fy * h))
    return pts


def probe_points(w, h, n=6, seed=None):
    rng = random.Random(seed)
    return [(rng.uniform(0.12, 0.88) * w, rng.uniform(0.12, 0.88) * h)
            for _ in range(n)]


class Aborted(Exception):
    pass


def info_screen(win, lines):
    """Show instructions until the user presses a key."""
    img = win.canvas()
    y = int(win.h * 0.38)
    for line in lines:
        ui.center_text(img, line, y, 0.85)
        y += 44
    ui.center_text(img, "press any key to continue", y + 20, 0.7, (150, 150, 150))
    cv2.imshow(win.name, img)
    if (cv2.waitKey(0) & 0xFF) in (27, ord("q")):
        raise Aborted


def _tick(win, img):
    key = win.show(img)
    if key in (27, ord("q")):
        raise Aborted


def environment_gate(win, cap, tracker, hold_s=2.0):
    """Stage 1: block until conditions are good and held for hold_s seconds."""
    good_since = None
    while True:
        frame = read_mirrored(cap)
        if frame is None:
            continue
        obs = tracker.process(frame)
        img = win.canvas()

        checks = []
        if not obs.ok:
            checks.append(("Face not detected - center yourself in the frame", False))
        else:
            iod = obs.interocular_px
            checks.append(("Distance OK" if 55 <= iod <= 240 else
                           ("Move CLOSER to the camera" if iod < 55 else
                            "Move FARTHER from the camera"), 55 <= iod <= 240))
            bright_ok = 60 <= obs.brightness <= 215
            checks.append(("Lighting OK" if bright_ok else
                           "Fix lighting (light your face, avoid strong backlight)",
                           bright_ok))
            pose_ok = abs(obs.yaw) < 15 and abs(obs.pitch) < 15
            checks.append(("Head pose OK" if pose_ok else
                           "Face the screen straight on", pose_ok))
            checks.append(("Eyes open" if obs.blink < BLINK_MAX else "Eyes open?",
                           obs.blink < BLINK_MAX))

        all_ok = obs.ok and all(ok for _, ok in checks)
        now = time.monotonic()
        if all_ok:
            good_since = good_since or now
        else:
            good_since = None

        ui.center_text(img, "SETUP CHECK", int(win.h * 0.22), 1.2)
        y = int(win.h * 0.32)
        for text, ok in checks:
            ui.center_text(img, ("OK  " if ok else "!!  ") + text, y, 0.8,
                           ui.GOOD if ok else ui.BAD)
            y += 40
        if good_since:
            frac = (now - good_since) / hold_s
            ui.center_text(img, "Hold still...", y + 20, 0.8)
            ui.progress_bar(img, frac, y + 40)
            if frac >= 1.0:
                return
        else:
            ui.center_text(img, "Fix the items above to continue (q = quit)",
                           y + 20, 0.7, (150, 150, 150))
        ui.thumbnail(img, frame, obs)
        _tick(win, img)


def collect_point(win, cap, tracker, pt, writer=None, tag="calib",
                  label="", color=ui.ACCENT):
    """Stage 2 for one point: settle, then gather gated samples.
    Returns (features_list, obs_list) or None if the point failed."""
    x, y = pt
    t0 = time.monotonic()
    while time.monotonic() - t0 < SETTLE_S:      # settle: discard frames
        frame = read_mirrored(cap)
        if frame is None:
            continue
        tracker.process(frame)
        img = win.canvas()
        ui.draw_target(img, x, y, (time.monotonic() - t0) / SETTLE_S, color)
        if label:
            ui.center_text(img, label, 50, 0.7, (150, 150, 150))
        _tick(win, img)

    feats, kept = [], []
    t0 = time.monotonic()
    while len(feats) < SAMPLES_PER_POINT:
        if time.monotonic() - t0 > POINT_TIMEOUT_S:
            break
        frame = read_mirrored(cap)
        if frame is None:
            continue
        obs = tracker.process(frame, want_crops=writer is not None)
        if (obs.ok and obs.blink < BLINK_MAX
                and abs(obs.yaw) < POSE_MAX_DEG and abs(obs.pitch) < POSE_MAX_DEG):
            feats.append(obs.features)
            kept.append(obs)
        img = win.canvas()
        ui.draw_target(img, x, y, 1.0, color)
        if label:
            ui.center_text(img, label, 50, 0.7, (150, 150, 150))
        _tick(win, img)

    if len(feats) < MIN_GOOD:
        return None

    # Stage 3: MAD outlier rejection on the iris features (dims 0..7)
    F = np.array(feats)
    med = np.median(F[:, :8], axis=0)
    mad = np.median(np.abs(F[:, :8] - med), axis=0) + 1e-9
    dev = np.max(np.abs(F[:, :8] - med) / mad, axis=1)
    keep = dev < 3.5
    if keep.sum() < MIN_GOOD * 0.6:
        return None
    if writer is not None:
        for o, k in zip(kept, keep):
            if k:
                writer.add(o, pt, tag=tag)
    return F[keep]


def collect_points(win, cap, tracker, points, writer=None, tag="calib",
                   label_prefix="", color=ui.ACCENT):
    X, Y = [], []
    order = list(points)
    random.shuffle(order)
    for i, pt in enumerate(order):
        label = f"{label_prefix}{i + 1} / {len(order)}"
        got = None
        for _ in range(MAX_RETRIES + 1):
            got = collect_point(win, cap, tracker, pt, writer, tag, label, color)
            if got is not None:
                break
        if got is None:
            continue  # point kept failing; the rest of the grid still trains
        X.append(got)
        Y.append(np.tile(pt, (len(got), 1)))
    if not X:
        raise Aborted
    return np.vstack(X), np.vstack(Y)


def validate(win, cap, tracker, model, writer=None, seed=None):
    """Stage 4: fresh probe points.
    Returns (mean_err_px, per_point list, X_probe, Y_probe) — the probe
    samples are honest labels too, so the final model gets refit on them."""
    results, Xp, Yp = [], [], []
    for pt in probe_points(win.w, win.h, seed=seed):
        got = collect_point(win, cap, tracker, pt, writer, tag="probe",
                            label="validation", color=(200, 120, 255))
        if got is None:
            continue
        preds = np.array([model.predict(f) for f in got])
        med = np.median(preds, axis=0)
        err = float(np.hypot(med[0] - pt[0], med[1] - pt[1]))
        spread = float(np.mean(np.std(preds, axis=0)))
        results.append({"target": [round(pt[0]), round(pt[1])],
                        "error_px": round(err, 1), "jitter_px": round(spread, 1)})
        Xp.append(got)
        Yp.append(np.tile(pt, (len(got), 1)))
    if not results:
        return float("inf"), [], None, None
    return (float(np.mean([r["error_px"] for r in results])), results,
            np.vstack(Xp), np.vstack(Yp))


def run(camera_index=0, points=16, rounds=2, model_out="data/gaze_model.pkl",
        dataset_root="data/dataset", landmarker="models/face_landmarker.task",
        screen=None):
    from .screen import screen_size
    sw, sh = screen or screen_size()
    diag = float(np.hypot(sw, sh))

    win = ui.FullscreenWindow("gazekit", (sw, sh))
    cap = open_camera(camera_index)
    tracker = FaceTracker(landmarker)
    writer = DatasetWriter(dataset_root, (sw, sh))
    report = {"screen": [sw, sh], "points": points, "rounds": rounds}

    try:
        environment_gate(win, cap, tracker)

        grid = grid_points(sw, sh, points)
        Xs, Ys = [], []
        for r in range(rounds):
            if r > 0:
                # posture drift is the top real-world error source: varying it
                # deliberately teaches the pose features to compensate
                info_screen(win, [
                    f"Round {r + 1} of {rounds}",
                    "Shift your posture slightly:",
                    "sit a bit closer or farther, small head shift is OK.",
                    "Keep facing the screen.",
                ])
            X, Y = collect_points(win, cap, tracker, grid, writer,
                                  label_prefix=f"round {r + 1}/{rounds}   point ")
            Xs.append(X)
            Ys.append(Y)
        X, Y = np.vstack(Xs), np.vstack(Ys)

        model = GazeModel((sw, sh))
        cv_err = model.fit(X, Y)
        report["train_samples"] = int(len(X))
        report["cv_error_px"] = round(cv_err, 1)

        mean_err, details, Xp, Yp = validate(win, cap, tracker, model, writer)
        report["validation"] = details
        report["mean_error_px"] = round(mean_err, 1)
        report["mean_error_frac_diag"] = round(mean_err / diag, 4)

        if PASS_FRAC < mean_err / diag <= MARGINAL_FRAC and details:
            # Stage 5: auto-repair around the two worst probe points
            worst = sorted(details, key=lambda r: -r["error_px"])[:2]
            extra = [tuple(r["target"]) for r in worst]
            Xr, Yr = collect_points(win, cap, tracker, extra, writer,
                                    label_prefix="repair ", color=ui.BAD)
            X, Y = np.vstack([X, Xr]), np.vstack([Y, Yr])
            model.fit(X, Y)
            mean_err, details, Xp, Yp = validate(win, cap, tracker, model,
                                                 writer, seed=7)
            report["repaired"] = True
            report["mean_error_px"] = round(mean_err, 1)
            report["mean_error_frac_diag"] = round(mean_err / diag, 4)
            report["validation"] = details

        # final refit folds the probe samples in — 6 extra honest targets
        if Xp is not None:
            X, Y = np.vstack([X, Xp]), np.vstack([Y, Yp])
            model.fit(X, Y)
            report["final_train_samples"] = int(len(X))

        frac = mean_err / diag
        verdict = ("STABLE" if frac <= PASS_FRAC else
                   "USABLE" if frac <= MARGINAL_FRAC else "POOR - recalibrate")
        report["verdict"] = verdict
        model.save(model_out, report)

        img = win.canvas()
        ui.center_text(img, f"Verdict: {verdict}", int(sh * 0.4), 1.3,
                       ui.GOOD if frac <= PASS_FRAC else ui.ACCENT)
        ui.center_text(img, f"mean error {mean_err:.0f}px "
                            f"({100 * frac:.1f}% of diagonal)", int(sh * 0.48), 0.9)
        ui.center_text(img, f"model saved to {model_out} - press any key",
                       int(sh * 0.56), 0.8, (150, 150, 150))
        cv2.imshow(win.name, img)
        cv2.waitKey(0)
        return report
    except Aborted:
        print("calibration aborted")
        return None
    finally:
        n = writer.close()
        print(f"dataset: {n} samples appended under {writer.dir}")
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
