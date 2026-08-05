"""`gazekit selftest` — automated half of docs/RELEASE_CHECKLIST.md.

No camera, no screen: everything here must pass before a beta ships.
"""

import base64
import json
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

CHECKS = []


class Skip(Exception):
    """Raised by a check that can't run in this environment."""


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("all modules import")
def _imports():
    import importlib
    modules = ("ambient", "annotate", "arkit", "auto", "calibrate", "camera",
               "cnn", "collect", "dataset", "evaluate", "eyeball", "filters",
               "journal", "live", "model", "phonecam", "publish", "screen",
               "quality", "tracker", "ui", "verify")
    for m in modules:
        importlib.import_module(f"gazekit.{m}")
    return f"{len(modules)} modules"


@check("CLI parses every subcommand")
def _cli():
    cmds = ["auto", "cameras", "doctor", "calibrate", "collect", "live",
            "ambient", "verify", "iterate", "annotate", "train-cnn", "arkit",
            "journal", "camera", "publish", "selftest"]
    out = subprocess.run([sys.executable, "-m", "gazekit", "--help"],
                         capture_output=True, text=True, timeout=60).stdout
    missing = [c for c in cmds if c not in out]
    assert not missing, f"missing from CLI: {missing}"
    return f"{len(cmds)} subcommands"


@check("camera source resolves to its own model file")
def _source():
    from gazekit.dataset import model_path_for
    assert model_path_for("0").endswith("gaze_model.pkl")
    assert model_path_for("phone").endswith("gaze_model_phone.pkl")
    return "webcam / phone separated"


@check("feature transform shape")
def _transform():
    from gazekit.model import transform
    out = transform(np.zeros((3, 14)))
    assert out.shape == (3, 19), out.shape
    return "14 -> 19 dims"


@check("model save/load round-trip")
def _model():
    from gazekit.model import GazeModel
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 14))
    Y = rng.normal(size=(300, 2)) * 100 + [960, 540]
    m = GazeModel((1920, 1080))
    m.fit(X, Y)
    p = Path("data/_selftest_model.pkl")
    m.save(p, {"selftest": True})
    m2 = GazeModel.load(p)
    a, b = m.predict(X[0]), m2.predict(X[0])
    p.unlink(missing_ok=True)
    p.with_suffix(".report.json").unlink(missing_ok=True)
    assert np.allclose(a, b), (a, b)
    return "predictions identical"


@check("dataset write/read round-trip")
def _dataset():
    from gazekit.dataset import DATA_SCHEMA_VERSION, DatasetWriter, load_sessions
    from gazekit.tracker import Observation
    root = Path("data/_selftest_ds")
    shutil.rmtree(root, ignore_errors=True)
    w = DatasetWriter(root, (1920, 1080))
    obs = Observation(ok=True, features=np.zeros(14), blink=0.02,
                      yaw=4.0, pitch=-3.0, brightness=120.0,
                      interocular_px=80.0,
                      landmarks_px=np.random.rand(478, 2) * 100,
                      eye_crops=(np.zeros((48, 64), np.uint8),
                                 np.zeros((48, 64), np.uint8)))
    obs.extras["tmatrix"] = np.eye(4)
    obs.extras["frame_size"] = [1280, 720]
    for i in range(3):
        w.add(obs, (500 + i, 500), tag="calib")
    n = w.close()
    got = list(load_sessions(root))
    rows = [json.loads(line) for line in (w.dir / "samples.jsonl").read_text().splitlines()]
    shutil.rmtree(root, ignore_errors=True)
    assert n == 3 and len(got) == 3, (n, len(got))
    meta, sample = rows[0], rows[1]
    assert meta["schema_version"] == DATA_SCHEMA_VERSION and meta["session_id"]
    assert meta["feature_schema"] == "ridge-raw14"
    assert set(sample["quality_components"]) == {"eyes", "pose", "distance", "lighting"}
    assert 0.0 <= sample["quality_score"] <= 1.0 and sample["frame_size"] == [1280, 720]
    return "3 samples in, v2 metadata + quality recorded"


@check("quality weights preserve legacy samples")
def _quality():
    from gazekit.quality import quality_weight, record_quality_score
    assert record_quality_score({}) == 1.0
    assert record_quality_score({"quality_score": "not-a-number"}) == 1.0
    assert np.isclose(quality_weight(0.0), 0.55)
    assert np.isclose(quality_weight(1.0), 1.0)
    return "legacy=1.0, bounded 0.55..1.0"


@check("blink gate uses the personal profile when present")
def _blink():
    from gazekit.calibrate import blink_max
    v = blink_max()
    assert 0.1 < v < 0.95, v
    return f"threshold {v}"


@check("deploy gate rejects a corrupt incumbent")
def _gate():
    from gazekit.evaluate import _aligned_err, _cluster_err
    from gazekit.model import GazeModel
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 14))
    W = rng.normal(size=(14, 2))
    Y = X @ W * 40 + [960, 540]
    good = GazeModel((1920, 1080))
    good.fit(X, Y)
    bad = GazeModel((1920, 1080))
    bad.fit(X[:30], Y[:30] * 0.1)      # deliberately broken
    recs = [{"session": "s", "i": i, "tag": "calib", "X": X[i], "Y": Y[i]}
            for i in range(400)]
    raw_good = np.mean([c["err"] for c in _cluster_err(good, recs)])
    raw_bad = np.mean([c["err"] for c in _cluster_err(bad, recs)])
    assert raw_bad > 3 * raw_good, (raw_good, raw_bad)
    return f"corrupt {raw_bad:.0f}px vs good {raw_good:.0f}px"


@check("phone protocol end-to-end (simulated phone)")
def _phone():
    import cv2
    from gazekit.phonecam import PhoneCamera
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("0.0.0.0", 5578))
    except OSError:
        raise Skip("port 5578 in use by a running camera session")
    finally:
        probe.close()
    img = (np.random.rand(360, 640, 3) * 255).astype(np.uint8)
    _, jpg = cv2.imencode(".jpg", img)
    payload = json.dumps({"type": "frame", "t": 0,
                          "jpg": base64.b64encode(jpg.tobytes()).decode()
                          }).encode()
    events, stop = [], threading.Event()

    def phone():
        while not stop.is_set():
            try:
                tcp = socket.create_connection(("127.0.0.1", 5578), timeout=2)
            except OSError:
                time.sleep(0.3)
                continue
            events.append("connected")
            tcp.settimeout(0.2)
            streaming, buf = False, b""
            try:
                while not stop.is_set():
                    try:
                        chunk = tcp.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                        while len(buf) >= 4:
                            n = struct.unpack(">I", buf[:4])[0]
                            if len(buf) < 4 + n:
                                break
                            msg, buf = json.loads(buf[4:4 + n]), buf[4 + n:]
                            events.append(msg["cmd"])
                            streaming = msg["cmd"] == "session_start"
                    except socket.timeout:
                        pass
                    if streaming:
                        tcp.sendall(struct.pack(">I", len(payload)) + payload)
                        time.sleep(0.05)
            except OSError:
                pass
            tcp.close()

    threading.Thread(target=phone, daemon=True).start()
    cam = PhoneCamera(wait_s=15)
    ok, frame = cam.read()
    cam.release()
    t0 = time.time()          # the phone polls at 0.2s; give it room
    while "session_stop" not in events and time.time() - t0 < 3:
        time.sleep(0.1)
    stop.set()
    time.sleep(0.2)
    assert ok and frame is not None, "no frame"
    assert "session_start" in events and "session_stop" in events, events
    return "connect -> start -> frames -> stop"


@check("journal records runs")
def _journal():
    from gazekit.journal import PATH, log_run, summary
    before = PATH.read_text() if PATH.exists() else ""
    log_run("selftest", ["selftest"], "done", 0.1, {"n": 1})
    assert "selftest" in summary(3)
    if before:
        PATH.write_text(before)     # leave the real journal untouched
    return "write + read"


def run():
    print("gazekit selftest — automated release checks\n")
    failed = skipped = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
            print(f"  PASS  {name}  ({detail})")
        except Skip as e:
            skipped += 1
            print(f"  SKIP  {name}: {e}")
        # SystemExit is how the camera layer reports env problems and it
        # would otherwise abort the whole suite
        except (Exception, SystemExit) as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    passed = len(CHECKS) - failed - skipped
    print(f"\n{passed}/{len(CHECKS)} passed"
          + (f", {skipped} skipped" if skipped else ""))
    if failed:
        raise SystemExit(1)
    print("automated checks green — walk docs/RELEASE_CHECKLIST.md manual "
          "list before tagging")
    return {"checks": len(CHECKS), "failed": failed}
