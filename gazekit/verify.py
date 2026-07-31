"""`gazekit verify` — mouse-as-ground-truth error measurement.

You look where your cursor is; the cursor position (lag-compensated) becomes
the label and the live gaze prediction is scored against it.

  --mode free   move the mouse anywhere you like; live error HUD
  --mode path   follow a wide, simple guided track (rounded loop + middle
                sweep) until coverage hits 100% — structured verification
                over the whole screen

Results: live 3x3 region error grid, summary printed + appended to
data/verify_log.jsonl. With --teach, every accepted sample is also saved to
the dataset (tag "mouse", low-trust: cleaned by `iterate`) so verified-bad
regions immediately become training data.
"""

import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import ui
from .camera import open_camera, read_mirrored
from .dataset import DatasetWriter
from .model import GazeModel
from .tracker import FaceTracker

LAG_S = 0.12          # the eye trails the moving cursor
MAX_SPEED = 700.0     # px/s — faster mouse motion isn't a reliable label
PATH_WIDTH = 90.0     # guided track half... full stroke width (wide + simple)
PATH_N = 420


def _path_points(w, h):
    """Wide simple track: rounded rectangle loop + a middle horizontal
    sweep. Returns (N, 2) points in drawing order."""
    mx, my = 0.12 * w, 0.14 * h
    pts = []
    # rectangle loop (corners softened by sampling density)
    corners = [(mx, my), (w - mx, my), (w - mx, h - my), (mx, h - my)]
    per_side = PATH_N // 5
    for i in range(4):
        a, b = np.array(corners[i]), np.array(corners[(i + 1) % 4])
        for t in np.linspace(0, 1, per_side, endpoint=False):
            pts.append(a + (b - a) * t)
    # middle sweep left -> right
    for t in np.linspace(0, 1, PATH_N - len(pts)):
        pts.append(np.array([mx + (w - 2 * mx) * t, h / 2]))
    return np.array(pts)


class MouseTrack:
    """Cursor history for lag compensation + speed estimation."""

    def __init__(self):
        self.hist = deque(maxlen=60)

    def push(self, x, y):
        self.hist.append((time.monotonic(), float(x), float(y)))

    def at(self, t_ago):
        """Position ~t_ago seconds in the past, or None."""
        if not self.hist:
            return None
        t_want = time.monotonic() - t_ago
        best = min(self.hist, key=lambda r: abs(r[0] - t_want))
        if abs(best[0] - t_want) > 0.25:
            return None
        return np.array(best[1:])

    def speed(self):
        if len(self.hist) < 2:
            return None
        (t0, x0, y0), (t1, x1, y1) = self.hist[-2], self.hist[-1]
        dt = t1 - t0
        if dt <= 0 or t1 < time.monotonic() - 0.3:
            return 0.0  # cursor resting
        return float(np.hypot(x1 - x0, y1 - y0) / dt)


def run(camera_index=0, mode="free", teach=False,
        model_path="data/gaze_model.pkl", dataset_root="data/dataset",
        landmarker="models/face_landmarker.task", screen=None):
    from .screen import screen_size
    sw, sh = screen or screen_size()
    model = GazeModel.load(model_path)
    tracker = FaceTracker(landmarker)
    cap = open_camera(camera_index)
    win = ui.FullscreenWindow("gazekit-verify", (sw, sh))
    writer = DatasetWriter(dataset_root, (sw, sh)) if teach else None

    mouse = MouseTrack()
    cv2.setMouseCallback(win.name,
                         lambda ev, x, y, flags, param: mouse.push(x, y))

    path = _path_points(sw, sh) if mode == "path" else None
    visited = np.zeros(len(path), dtype=bool) if path is not None else None

    errs, region = [], [[[] for _ in range(3)] for _ in range(3)]
    n_saved = 0
    try:
        while True:
            frame = read_mirrored(cap)
            if frame is None:
                continue
            obs = tracker.process(frame, want_crops=teach)
            img = win.canvas()

            if path is not None:
                done = visited.mean()
                for i in range(len(path) - 1):
                    col = ui.GOOD if visited[i] else (70, 70, 70)
                    cv2.line(img, tuple(path[i].astype(int)),
                             tuple(path[i + 1].astype(int)), col,
                             int(PATH_WIDTH), cv2.LINE_AA)
                ui.center_text(img, f"follow the track with your mouse — "
                               f"{100 * done:.0f}% covered", 50, 0.8)
                if done >= 0.999:
                    break
            else:
                ui.center_text(img, "move the mouse, look AT the cursor "
                               "(q = finish)", 50, 0.7, (140, 140, 140))

            label = mouse.at(LAG_S)
            speed = mouse.speed()
            ok_sample = (obs.ok and obs.blink < 0.3 and label is not None
                         and speed is not None and speed < MAX_SPEED)
            if ok_sample and path is not None:
                d = np.linalg.norm(path - label, axis=1)
                near = int(np.argmin(d))
                if d[near] <= PATH_WIDTH:
                    visited[max(0, near - 3):near + 4] = True
                else:
                    ok_sample = False  # off the track

            if ok_sample:
                pred = model.predict(obs.features)
                e = float(np.hypot(*(pred - label)))
                errs.append(e)
                gx = min(int(label[0] / sw * 3), 2)
                gy = min(int(label[1] / sh * 3), 2)
                region[gy][gx].append(e)
                if teach:
                    writer.add(obs, tuple(label), tag="mouse")
                    n_saved += 1
                cv2.line(img, tuple(pred.astype(int)),
                         tuple(label.astype(int)), (90, 90, 200), 1,
                         cv2.LINE_AA)
                ui.draw_gaze_dot(img, *pred)

            if errs:
                recent = errs[-90:]
                ui.center_text(img, f"error now {np.mean(recent[-15:]):4.0f}px"
                               f"   session mean {np.mean(errs):4.0f}px"
                               f"   n={len(errs)}", sh - 60, 0.7)
            # live 3x3 grid, bottom-right corner
            for gy in range(3):
                for gx in range(3):
                    cell = region[gy][gx]
                    txt = f"{np.mean(cell):.0f}" if cell else "-"
                    cv2.putText(img, txt,
                                (sw - 170 + gx * 52, sh - 120 + gy * 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (150, 150, 150), 1, cv2.LINE_AA)

            key = win.show(img)
            if key in (27, ord("q")):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        if writer is not None:
            n = writer.close()
            if n:
                print(f"teach: {n} mouse-labeled samples saved — run "
                      "`gazekit iterate` to clean + retrain with them")

    if not errs:
        print("no samples collected")
        return
    summary = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode,
        "n": len(errs), "mean_px": round(float(np.mean(errs)), 1),
        "median_px": round(float(np.median(errs)), 1),
        "p90_px": round(float(np.percentile(errs, 90)), 1),
        "region_px": [[round(float(np.mean(c)), 0) if c else None
                       for c in row] for row in region],
        "taught": n_saved,
    }
    Path("data").mkdir(exist_ok=True)
    with open("data/verify_log.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"\nverify ({mode}): {summary['n']} samples  "
          f"mean {summary['mean_px']}px  median {summary['median_px']}px  "
          f"p90 {summary['p90_px']}px")
    print("region map (px):")
    for row in summary["region_px"]:
        print("   " + " ".join(f"{int(c):>5}" if c is not None else "    -"
                               for c in row))
    return summary
