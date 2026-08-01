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
    X, Y, S = [], [], []
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
    sessions = sorted(set(S))
    if len(sessions) >= 2:
        errs = []
        for s in sessions:
            m = make().fit(X[S != s], Y[S != s])
            e = np.linalg.norm(m.predict(X[S == s]) - Y[S == s], axis=1)
            errs.append(float(np.mean(e)))
            print(f"  {s}: {errs[-1]:.0f}px (n={int((S == s).sum())})")
        print(f"ARKit-teacher LOSO: {np.mean(errs):.0f}px")
    model = make().fit(X, Y)
    import pickle
    with open("data/arkit_teacher.pkl", "wb") as f:
        pickle.dump(model, f)
    print("teacher mapping saved -> data/arkit_teacher.pkl "
          "(ARKit stream + this mapping = continuous gaze labels)")
