"""ARKit gaze-teacher pipeline (iPhone TrueDepth via the GazeTeacher app).

  receive   `gazekit arkit` — UDP listener, records the phone's stream to
            data/arkit/stream_*.jsonl and prints the frame rate. Leave it
            running while you do any normal gazekit collection.
  pair+fit  `gazekit arkit --fit` — joins collection samples (their wall
            time "t") with the nearest ARKit frame (<=60ms), fits a
            regression from ARKit gaze geometry to screen px, and reports
            leave-one-session-out error. That mapping turns the phone into
            a continuous gaze teacher for distillation.
"""

import json
import socket
import time
from pathlib import Path

import numpy as np

PORT = 5577
ARKIT_DIR = Path("data/arkit")
PAIR_TOL_S = 0.06


def _local_ip():
    """Prefer the Wi-Fi/Ethernet LAN address — the default-route trick can
    return a VPN/Tailscale address the phone can't reach."""
    import subprocess
    for iface in ("en0", "en1"):
        try:
            ip = subprocess.run(["ipconfig", "getifaddr", iface],
                                capture_output=True, text=True,
                                timeout=3).stdout.strip()
            if ip:
                return ip
        except Exception:
            pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def receive(port=PORT):
    ARKIT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARKIT_DIR / f"stream_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        raise SystemExit(
            f"port {port} already in use — a receiver is already running "
            "and recording (check `pgrep -fl 'gazekit arkit'`); no need to "
            "start another")
    sock.settimeout(1.0)
    print(f"listening on {_local_ip()}:{port} — enter that IP in the "
          f"GazeTeacher app.  recording to {out}   (Ctrl+C to stop)")
    n, t_last = 0, time.monotonic()
    try:
        with open(out, "w") as f:
            while True:
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                try:
                    pkt = json.loads(data)
                except json.JSONDecodeError:
                    continue
                pkt["t_recv"] = time.time()
                f.write(json.dumps(pkt) + "\n")
                n += 1
                if time.monotonic() - t_last > 5:
                    print(f"  {n} frames  (~{n / (time.monotonic() - t_last + 1e-9):.0f}"
                          " fps recent)" if n else "  waiting for packets...")
                    n, t_last = 0, time.monotonic()
    except KeyboardInterrupt:
        print(f"\nstopped -> {out}")
    finally:
        sock.close()


def monitor():
    """Live viewer for the phone stream: shows exactly what ARKit sees
    (frames via tcp:5578) plus the latest gaze/blink numbers — the Mac-side
    debug window for the GazeTeacher app."""
    import cv2
    from .phonecam import PhoneCamera
    cam = PhoneCamera()
    print("monitor: q to quit")
    last_gaze = ""
    n, t0 = 0, time.monotonic()
    fps = 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue
            n += 1
            if time.monotonic() - t0 >= 2.0:
                fps = n / (time.monotonic() - t0)
                n, t0 = 0, time.monotonic()
            if cam.gaze_path and cam.gaze_path.exists():
                try:
                    with open(cam.gaze_path, "rb") as f:
                        f.seek(max(f.seek(0, 2) - 400, 0))
                        line = f.read().splitlines()[-1]
                    pkt = json.loads(line)
                    last_gaze = (f"look=({pkt['look'][0]:+.2f},"
                                 f"{pkt['look'][1]:+.2f}) "
                                 f"blink={max(pkt['blinkL'], pkt['blinkR']):.2f}")
                except (OSError, IndexError, json.JSONDecodeError, KeyError):
                    pass
            cv2.putText(frame, f"{fps:.0f} fps  {last_gaze}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 255), 2)
            cv2.imshow("GazeTeacher monitor", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


def _record_stream_bg(stop_event, port=PORT):
    """In-process ARKit stream recorder (background thread).
    Returns a stats dict whose "n" counts received packets — the liveness
    signal for connection gating."""
    import threading
    stats = {"n": 0}

    def loop():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            print(f"(gaze port {port} busy — assuming another recorder runs)")
            return
        sock.settimeout(1.0)
        ARKIT_DIR.mkdir(parents=True, exist_ok=True)
        out = ARKIT_DIR / f"stream_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        n = 0
        with open(out, "w") as f:
            while not stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(4096)
                    pkt = json.loads(data)
                    pkt["t_recv"] = time.time()
                    f.write(json.dumps(pkt) + "\n")
                    n += 1
                    stats["n"] = n
                    stats["blink"] = max(pkt.get("blinkL", 0),
                                         pkt.get("blinkR", 0))
                    stats["path"] = str(out)
                    if n % 15 == 0:
                        f.flush()
                except (socket.timeout, json.JSONDecodeError):
                    continue
        sock.close()
        print(f"stream: {n} gaze frames -> {out}")

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stats


def calib(points=13):
    """Camera-free calibration for the ARKit teacher: the screen shows
    targets and logs (t_start, t_end, target); the phone's stream provides
    the eyes — and is RECORDED IN-PROCESS, so this one command is
    self-sufficient. Works while GazeTeacher holds the iPhone camera."""
    import threading
    import cv2
    from . import ui
    from .calibrate import grid_points
    from .screen import screen_size
    from .ui import say
    sw, sh = screen_size()
    stop = threading.Event()
    stats = _record_stream_bg(stop)

    def rate(window=0.5):
        n0 = stats["n"]
        t0 = time.monotonic()
        while time.monotonic() - t0 < window:
            time.sleep(0.05)
        return (stats["n"] - n0) / window
    win = ui.FullscreenWindow("gazekit-arkit-calib", (sw, sh))
    ARKIT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARKIT_DIR / f"calib_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    pts = grid_points(sw, sh, points)
    import random
    random.shuffle(pts)
    try:
        # pre-flight gate (collection standard): stream alive at >20 pkt/s
        # AND eyes readable (blink not stuck high) held for 1.5 s
        say("waiting for the phone stream")
        good_since = None
        while True:
            r = rate(0.5)
            blink = stats.get("blink", 1.0)
            ok = r > 20 and blink < 0.4
            now_m = time.monotonic()
            good_since = (good_since or now_m) if ok else None
            img = win.canvas()
            checks = [(f"stream {r:.0f} pkt/s", r > 20),
                      (f"eyes open (blink {blink:.2f})", blink < 0.4)]
            y = int(sh * 0.40)
            for text, passed in checks:
                ui.center_text(img, ("OK  " if passed else "!!  ") + text,
                               y, 0.8, ui.GOOD if passed else ui.BAD)
                y += 40
            if good_since and now_m - good_since >= 1.5:
                break
            if not ok:
                ui.center_text(img, "open GazeTeacher, check the IP, tap "
                               "Start", y + 10, 0.65, (150, 150, 150))
            if win.show(img) in (27, ord("q")):
                raise KeyboardInterrupt
        say("connected")

        with open(out, "w") as f:
            f.write(json.dumps({"meta": True, "screen_size": [sw, sh]}) + "\n")
            # blink break: a fresh tear film before sampling — dry eyes
            # drift the iris fit and quietly poison the whole session
            say("blink a few times, then look at the dots")
            t0 = time.monotonic()
            while time.monotonic() - t0 < 2.5:
                img = win.canvas()
                ui.center_text(img, "eyes dry? blink a few times now",
                               int(sh * 0.45), 1.0)
                if win.show(img) in (27, ord("q")):
                    raise KeyboardInterrupt

            # collection standard stage 4: grid points train, then probe
            # points the mapping never fit on measure honest error
            import random as _rnd
            probes = [( _rnd.uniform(0.15, 0.85) * sw,
                        _rnd.uniform(0.15, 0.85) * sh) for _ in range(4)]
            all_pts = [(p, False) for p in pts] + [(p, True) for p in probes]
            i = 0
            while i < len(all_pts):
                (x, y), is_probe = all_pts[i]
                t0 = time.monotonic()
                while time.monotonic() - t0 < 0.8:   # settle, not logged
                    img = win.canvas()
                    ui.draw_target(img, x, y, (time.monotonic() - t0) / 0.8,
                                   (200, 120, 255) if is_probe else ui.ACCENT)
                    ui.center_text(img, f"{i + 1} / {len(all_pts)}", 50, 0.7,
                                   (150, 150, 150))
                    if win.show(img) in (27, ord("q")):
                        raise KeyboardInterrupt
                t_start = time.time()
                n_start = stats["n"]
                t0 = time.monotonic()
                while time.monotonic() - t0 < 1.3:   # dwell, logged window
                    img = win.canvas()
                    ui.draw_target(img, x, y, 1.0,
                                   (200, 120, 255) if is_probe else ui.ACCENT)
                    if win.show(img) in (27, ord("q")):
                        raise KeyboardInterrupt
                got = stats["n"] - n_start
                if got < 15:
                    # stream died mid-point: window is invalid — wait for
                    # the phone to come back, then REDO this point
                    say("phone stream lost")
                    while rate(0.5) < 20:
                        img = win.canvas()
                        ui.center_text(img, "stream lost — waiting for the "
                                       "phone...", int(sh * 0.45), 1.0)
                        if win.show(img) in (27, ord("q")):
                            raise KeyboardInterrupt
                    say("resumed")
                    continue
                f.write(json.dumps({"t_start": t_start, "t_end": time.time(),
                                    "target": [x, y], "frames": got,
                                    "probe": is_probe}) + "\n")
                i += 1

        # validate: fit on grid windows, score on probe windows
        report = {"n_points": len(pts), "n_probes": len(probes)}
        stop.set()
        time.sleep(1.3)   # recorder flush
        try:
            frames = [json.loads(l) for l in open(stats["path"])]
            times = np.array([p["t_recv"] for p in frames])
            Xg, Yg, Xp, Yp = [], [], [], []
            for line in open(out):
                rec = json.loads(line)
                if rec.get("meta"):
                    continue
                lo = int(np.searchsorted(times, rec["t_start"] + 0.15))
                hi = int(np.searchsorted(times, rec["t_end"]))
                for j in range(lo, hi):
                    pkt = frames[j]
                    if max(pkt.get("blinkL", 0), pkt.get("blinkR", 0)) > 0.35:
                        continue
                    (Xp if rec.get("probe") else Xg).append(_features(pkt))
                    (Yp if rec.get("probe") else Yg).append(rec["target"])
            if len(Xg) > 100 and len(Xp) > 20:
                from sklearn.linear_model import Ridge
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import (PolynomialFeatures,
                                                   StandardScaler)
                m = make_pipeline(StandardScaler(),
                                  PolynomialFeatures(2, include_bias=False),
                                  Ridge(alpha=10.0)).fit(np.array(Xg),
                                                         np.array(Yg))
                err = float(np.mean(np.linalg.norm(
                    m.predict(np.array(Xp)) - np.array(Yp), axis=1)))
                diag = float(np.hypot(sw, sh))
                verdict = ("STABLE" if err / diag <= 0.045 else
                           "USABLE" if err / diag <= 0.075 else
                           "POOR - recalibrate")
                report.update(probe_err_px=round(err, 1), verdict=verdict,
                              pairs=len(Xg) + len(Xp))
                say(f"verdict {verdict.split()[0]}")
                print(f"validation: {err:.0f}px on {len(Xp)} held-out probe "
                      f"samples -> {verdict}")
        except (OSError, KeyError, ValueError) as e:
            print(f"(validation skipped: {e})")
        print(f"done -> {out}  (run `gazekit arkit --fit` to update the "
              "teacher)")
        return report
    except KeyboardInterrupt:
        print(f"\naborted -> {out}")
    finally:
        stop.set()
        time.sleep(1.2)   # let the recorder flush and report
        cv2.destroyAllWindows()


def _load_calib_pairs(frames, times):
    """(features, target) pairs from camera-free calib windows."""
    X, Y, S = [], [], []
    for p in sorted(ARKIT_DIR.glob("calib_*.jsonl")):
        for line in open(p):
            rec = json.loads(line)
            if rec.get("meta"):
                continue
            lo = int(np.searchsorted(times, rec["t_start"] + 0.15))
            hi = int(np.searchsorted(times, rec["t_end"]))
            for j in range(lo, hi):
                pkt = frames[j]
                if max(pkt.get("blinkL", 0), pkt.get("blinkR", 0)) > 0.35:
                    continue
                X.append(_features(pkt))
                Y.append(rec["target"])
                S.append(p.stem)
    return X, Y, S


def _load_streams():
    frames = []
    for p in sorted(ARKIT_DIR.glob("stream_*.jsonl")):
        for line in open(p):
            try:
                pkt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "look" in pkt and "face" in pkt:
                frames.append(pkt)
    frames.sort(key=lambda p: p.get("t_recv", p.get("t", 0)))
    return frames


def _features(pkt):
    """ARKit gaze geometry -> feature vector for the screen mapping."""
    look = np.array(pkt["look"], dtype=float)
    face = np.array(pkt["face"], dtype=float).reshape(4, 4, order="F")
    R, t = face[:3, :3], face[:3, 3]
    look_world = R @ look + t          # lookAtPoint into camera/world space
    fwd = R @ np.array([0.0, 0.0, 1.0])
    return np.concatenate([look_world, fwd, t, [pkt.get("blinkL", 0),
                                                pkt.get("blinkR", 0)]])


def fit(dataset_root="data/dataset"):
    frames = _load_streams()
    if not frames:
        raise SystemExit("no ARKit streams recorded yet — run "
                         "`gazekit arkit` while collecting")
    times = np.array([p.get("t_recv", p["t"]) for p in frames])

    from .dataset import DWELL_TAGS, load_pruned
    root = Path(dataset_root)
    pruned = load_pruned(root)
    X, Y, S = _load_calib_pairs(frames, times)  # camera-free calib windows
    for sess in sorted(root.glob("session_*")):
        jl = sess / "samples.jsonl"
        if not jl.exists():
            continue
        bad = pruned.get(sess.name, set())
        for line in open(jl):
            rec = json.loads(line)
            if (rec.get("meta") or rec.get("tag") not in DWELL_TAGS
                    or rec.get("i") in bad or "t" not in rec):
                continue
            k = int(np.searchsorted(times, rec["t"]))
            best = min((abs(times[j] - rec["t"]), j)
                       for j in range(max(0, k - 1), min(len(times), k + 2)))
            if best[0] > PAIR_TOL_S:
                continue
            pkt = frames[best[1]]
            if max(pkt.get("blinkL", 0), pkt.get("blinkR", 0)) > 0.35:
                continue
            X.append(_features(pkt))
            Y.append(rec["target"])
            S.append(sess.name)
    if len(X) < 200:
        raise SystemExit(f"only {len(X)} paired samples — run `gazekit "
                         "arkit` during a calibrate/collect session first")
    X, Y, S = np.array(X), np.array(Y, dtype=float), np.array(S)
    print(f"paired {len(X)} samples across {len(set(S))} session(s)")

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    def make():
        return make_pipeline(StandardScaler(),
                             PolynomialFeatures(2, include_bias=False),
                             Ridge(alpha=10.0))
    # sessions with too few distinct targets (aborted calibs) can't give a
    # meaningful held-out estimate — they still train, just aren't judges
    def n_targets(s):
        return len({tuple(y) for y in Y[S == s]})
    sessions = sorted(set(S))
    judges = [s for s in sessions if n_targets(s) >= 4]
    if len(sessions) >= 2 and judges:
        errs = []
        for s in judges:
            m = make().fit(X[S != s], Y[S != s])
            e = np.linalg.norm(m.predict(X[S == s]) - Y[S == s], axis=1)
            errs.append(float(np.mean(e)))
            print(f"  {s}: {errs[-1]:.0f}px (n={int((S == s).sum())})")
        print(f"ARKit-teacher LOSO: {np.mean(errs):.0f}px "
              f"over {len(judges)} session(s)")
    model = make().fit(X, Y)
    import pickle
    with open("data/arkit_teacher.pkl", "wb") as f:
        pickle.dump(model, f)
    print("teacher mapping saved -> data/arkit_teacher.pkl "
          "(ARKit stream + this mapping = continuous gaze labels)")
