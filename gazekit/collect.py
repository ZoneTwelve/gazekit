"""Extra training-data scenarios beyond the calibration grid.

    pursuit   slow Lissajous sweep — continuous coverage of the whole screen
              (labels lag-compensated; used by the CNN, excluded from ridge)
    edges     tight-margin points: corners and screen edges the grid misses
    posture   9-point grid repeated at 3 instructed sitting positions —
              THE fix for "I moved and tracking went wrong"

Each run appends to data/dataset/ and (except pursuit-only) refits the
ridge model on the newest sessions so live mode improves immediately.
"""

import math
import time

import cv2
import numpy as np

from . import ui
from .calibrate import (Aborted, collect_points, environment_gate, info_screen,
                        grid_points)
from .camera import open_camera, read_mirrored
from .dataset import DatasetWriter, load_dwell_features
from .model import GazeModel
from .tracker import FaceTracker

PURSUIT_LAG_S = 0.12   # eye trails a moving target by ~100-150 ms
PURSUIT_SPEED = 0.055  # Lissajous base frequency, keeps target < ~350 px/s


def _pursuit(win, cap, tracker, writer, duration=45.0):
    """Slow Lissajous sweep; each accepted frame is labeled with where the
    target was PURSUIT_LAG_S ago."""
    w, h = win.w, win.h

    def pos(t):
        x = w * (0.5 + 0.44 * math.sin(2 * math.pi * PURSUIT_SPEED * t))
        y = h * (0.5 + 0.42 * math.sin(2 * math.pi * PURSUIT_SPEED * 0.62 * t + 1.1))
        return x, y

    t0 = time.monotonic()
    n = 0
    while True:
        t = time.monotonic() - t0
        if t > duration:
            break
        frame = read_mirrored(cap)
        if frame is None:
            continue
        obs = tracker.process(frame, want_crops=True)
        x, y = pos(t)
        # skip the ramp-in and blinks; label with the lagged target position
        if t > 2.0 and obs.ok and obs.blink < 0.3:
            writer.add(obs, pos(t - PURSUIT_LAG_S), tag="pursuit")
            n += 1
        img = win.canvas()
        ui.draw_target(img, x, y, 1.0)
        ui.center_text(img, f"follow the dot  ({int(duration - t)}s)", 50,
                       0.7, (150, 150, 150))
        key = win.show(img)
        if key in (27, ord("q")):
            raise Aborted
    return n


def _edge_points(w, h):
    m = 0.02
    xs = np.linspace(m * w, (1 - m) * w, 4)
    return ([(x, m * h) for x in xs] + [(x, (1 - m) * h) for x in xs]
            + [(m * w, h / 2), ((1 - m) * w, h / 2)])


BLINK_PROFILE = "data/blink_profile.json"


_say = ui.say


def _blink_phase(win, cap, tracker, writer, duration, tag=None,
                 skip_s=0.7, hint=""):
    """Collect (blink_score, min_openness) pairs. Screen shows only the dot
    (plus an optional small hint) — instructions come by voice."""
    vals = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        t = time.monotonic() - t0
        frame = read_mirrored(cap)
        if frame is None:
            continue
        obs = tracker.process(frame, want_crops=tag is not None)
        img = win.canvas()
        ui.draw_target(img, win.w / 2, win.h / 2, 1.0)
        if hint:
            ui.center_text(img, hint, 50, 0.6, (120, 120, 120))
        key = win.show(img)
        if key in (27, ord("q")):
            raise Aborted
        if obs.ok and t > skip_s:
            vals.append((float(obs.blink),
                         float(min(obs.features[3], obs.features[7]))))
            if tag:
                writer.add(obs, (0.0, 0.0), tag=tag)
    return np.array(vals)


def _blinks(win, cap, tracker, writer):
    import json
    info_screen(win, ["Blink calibration (~25 s) — follow the VOICE prompts.",
                      "1. Look at the dot with your eyes OPEN.",
                      "2. When told, CLOSE your eyes until told to open.",
                      "3. Then blink normally a few times.",
                      "(turn your sound on)"])
    _say("Look at the dot. Keep your eyes open.")
    open_v = _blink_phase(win, cap, tracker, writer, 4.5, skip_s=1.5,
                          hint="eyes open, look at the dot")
    _say("Now close your eyes, and keep them closed.")
    closed_v = _blink_phase(win, cap, tracker, writer, 5.0, tag="closed",
                            skip_s=2.4)  # speech + reaction time
    _say("Open your eyes.")
    _blink_phase(win, cap, tracker, writer, 1.5, skip_s=99)  # discard
    _say("Now blink normally a few times.")
    natural = _blink_phase(win, cap, tracker, writer, 6.0, skip_s=0.5,
                           hint="blink normally")

    if len(open_v) < 10 or len(closed_v) < 10:
        print(f"not enough tracked frames (open={len(open_v)}, "
              f"closed={len(closed_v)}) — profile not saved. The tracker "
              "may lose your face with eyes closed; try better lighting.")
        return
    ob, oo = np.median(open_v, axis=0)
    cb, co = np.median(closed_v, axis=0)
    if cb - ob < 0.15 or oo - co < 0.03:
        print("open/closed distributions overlap too much — profile not "
              "saved, keeping defaults. (Was the face tracked while closed?)")
        return
    profile = {
        "open_blink": round(ob, 3), "closed_blink": round(cb, 3),
        "open_openness": round(oo, 3), "closed_openness": round(co, 3),
        # 0.45 interpolation: offline sweep vs labeled closed/open frames
        # showed the lower threshold gains recall at zero precision cost
        "blink_on": round(ob + 0.45 * (cb - ob), 3),
        "blink_off": round(ob + 0.30 * (cb - ob), 3),
        "open_min": round(co + 0.40 * (oo - co), 3),
    }
    with open(BLINK_PROFILE, "w") as f:
        json.dump(profile, f, indent=2)
    n_blinks = int((np.diff((natural[:, 0] > profile["blink_on"])
                            .astype(int)) == 1).sum())
    print(f"blink profile saved -> {BLINK_PROFILE}")
    print(f"  open blink={ob:.2f} openness={oo:.2f} | "
          f"closed blink={cb:.2f} openness={co:.2f}")
    print(f"  thresholds: on={profile['blink_on']} off={profile['blink_off']} "
          f"open_min={profile['open_min']}")
    print(f"  sanity: detected {n_blinks} natural blinks in phase 4")


VOR_POINTS = [(0.5, 0.5), (0.25, 0.3), (0.75, 0.3), (0.25, 0.75), (0.75, 0.75)]
VOR_PHASES = [("turn your head LEFT and RIGHT, slowly", 5.0),
              ("now tilt your head UP and DOWN, slowly", 5.0)]


def _vor_point(win, cap, tracker, writer, pt):
    """Fixed dot + moving head: eyes counter-rotate (vestibulo-ocular reflex),
    so every head pose gets a sample with a perfectly known gaze label.
    No MAD rejection here — the feature spread is the point."""
    n = 0
    for instruction, duration in VOR_PHASES:
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            frame = read_mirrored(cap)
            if frame is None:
                continue
            obs = tracker.process(frame, want_crops=True)
            if (time.monotonic() - t0 > 0.8 and obs.ok and obs.blink < 0.3
                    and abs(obs.yaw) < 40 and abs(obs.pitch) < 40):
                writer.add(obs, pt, tag="vor")
                n += 1
            img = win.canvas()
            ui.draw_target(img, *pt, 1.0, ui.GOOD)
            ui.center_text(img, "eyes LOCKED on the dot", 50, 0.7,
                           (150, 150, 150))
            ui.center_text(img, instruction, 92, 0.8)
            key = win.show(img)
            if key in (27, ord("q")):
                raise Aborted
    return n


POSTURES = [
    ("normal", "Sit how you NORMALLY sit."),
    ("close", "Lean IN, about 10 cm closer to the screen."),
    ("far", "Lean BACK in your chair, farther from the screen."),
]


def run(scenario, camera_index=0, dataset_root="data/dataset",
        model_out="data/gaze_model.pkl",
        landmarker="models/face_landmarker.task", screen=None):
    from .screen import screen_size
    sw, sh = screen or screen_size()
    win = ui.FullscreenWindow("gazekit-collect", (sw, sh))
    cap = open_camera(camera_index)
    tracker = FaceTracker(landmarker)
    writer = DatasetWriter(dataset_root, (sw, sh))
    refit_ridge = scenario not in ("pursuit", "blinks")

    try:
        environment_gate(win, cap, tracker)
        frame = read_mirrored(cap)
        if frame is not None:
            writer.save_context(frame)

        if scenario == "pursuit":
            info_screen(win, ["Smooth pursuit sweep (~45 s).",
                              "Follow the moving dot.",
                              "Move your head naturally, like real usage."])
            n = _pursuit(win, cap, tracker, writer)
            print(f"pursuit: {n} samples")

        elif scenario == "vor":
            info_screen(win, ["Head-movement training (5 points).",
                              "Keep your EYES locked on each dot while",
                              "slowly moving your head as instructed.",
                              "This teaches head-pose compensation."])
            total = 0
            for fx, fy in VOR_POINTS:
                total += _vor_point(win, cap, tracker, writer,
                                    (fx * sw, fy * sh))
            print(f"vor: {total} samples")

        elif scenario == "edges":
            info_screen(win, ["Edge & corner points.",
                              "These are the spots the normal grid misses."])
            collect_points(win, cap, tracker, _edge_points(sw, sh), writer,
                           tag="edges", label_prefix="edges ")

        elif scenario == "blinks":
            _blinks(win, cap, tracker, writer)

        elif scenario == "daily":
            # standardized ~2min daily probe: run 2-3x per day under
            # DIFFERENT lighting (morning/evening/lamp). Fills the two
            # measured dataset gaps: illumination coverage (brightness was
            # never logged before 2026-08-01) and cross-day variation.
            info_screen(win, ["Daily probe (~2 min).",
                              "Run this 2-3x per day in different lighting.",
                              "Grid, then edges, then two head-movement dots."])
            collect_points(win, cap, tracker, grid_points(sw, sh, 9),
                           writer, tag="calib", label_prefix="grid ")
            edge = _edge_points(sw, sh)[::2]
            collect_points(win, cap, tracker, edge, writer,
                           tag="edges", label_prefix="edges ")
            for fx, fy in ((0.3, 0.5), (0.7, 0.5)):
                _vor_point(win, cap, tracker, writer, (fx * sw, fy * sh))

        elif scenario == "posture":
            for name, instruction in POSTURES:
                info_screen(win, [f"Posture: {name}", instruction,
                                  "Then follow the dots as usual."])
                collect_points(win, cap, tracker, grid_points(sw, sh, 9),
                               writer, tag="posture",
                               label_prefix=f"{name} ")

        if refit_ridge:
            writer._f.flush()
            X, Y, w = load_dwell_features(dataset_root)
            if X is not None:
                model = GazeModel((sw, sh))
                cv_err = model.fit(X, Y, sample_weight=w)
                model.save(model_out, {"refit_from": "collect:" + scenario,
                                       "samples": int(len(X)),
                                       "cv_error_px": round(cv_err, 1)})
                print(f"ridge refit on {len(X)} samples "
                      f"(cv {cv_err:.0f}px) -> {model_out}")
        return True
    except Aborted:
        print("collect aborted")
        return False
    finally:
        n = writer.close()
        print(f"dataset: {n} samples appended under {writer.dir}")
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
