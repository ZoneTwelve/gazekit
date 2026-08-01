"""PhoneCamera: the iPhone (GazeTeacher app) as gazekit's camera.

Accepts the app's TCP frame stream (:5578, length-prefixed JSON with
base64 JPEG) and exposes a cv2.VideoCapture-compatible read() so every
existing flow works with `--camera phone`. Simultaneously logs the app's
UDP gaze packets (:5577) to data/arkit/stream_*.jsonl with t_recv, so a
normal calibrate/collect run doubles as ARKit-teacher pairing data.

Sensor frames arrive in landscape orientation; on the first frames a
one-shot MediaPipe probe tries the four rotations and locks onto the one
where a face is found.
"""

import base64
import json
import socket
import struct
import threading
import time
from pathlib import Path  # noqa: F401  (CONFIG below)

import cv2
import numpy as np

FRAME_PORT = 5578
GAZE_PORT = 5577
CONFIG = Path("data/config.json")
STATUS = Path("data/phone_status.json")


def write_status(**kw):
    """Heartbeat any owner of the phone ports writes, so `camera status`
    can report while a collection run holds the sockets."""
    try:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps({"t": time.time(), **kw}))
    except OSError:
        pass


def read_status(max_age=8.0):
    try:
        s = json.loads(STATUS.read_text())
        return s if time.time() - s.get("t", 0) < max_age else None
    except (OSError, json.JSONDecodeError):
        return None


def default_camera():
    """Configured default source ('phone' or an index string)."""
    try:
        return json.loads(CONFIG.read_text()).get("camera", "0")
    except (OSError, json.JSONDecodeError):
        return "0"


def phone_control(action, wait_s=30):
    """`gazekit camera <action>` — detection & remote control per
    docs/PHONE_PROTOCOL.md. The phone auto-reconnects, so we just listen
    on :5578 and it comes to us."""
    if action in ("app", "cam"):
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(
            {"camera": "phone" if action == "app" else "0"}))
        print(f"default camera -> "
              f"{'phone (GazeTeacher)' if action == 'app' else 'webcam 0'}")
        return

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", FRAME_PORT))
    except OSError:
        # a collection run owns the sockets — report from its heartbeat
        # instead of failing (status must work while the server runs)
        st = read_status()
        if action == "status" and st:
            age = time.time() - st["t"]
            print(f"status (via running {st.get('owner','?')} process, "
                  f"{age:.1f}s ago): connected={st.get('connected')} "
                  f"frames={st.get('frames')} gaze={st.get('gaze')}"
                  + (f" waiting={st['waiting']}s" if "waiting" in st else ""))
            return
        raise SystemExit(f"port {FRAME_PORT} busy — a `--camera phone` "
                         "process owns the phone; `camera status` can still "
                         "report if that process is heartbeating")
    srv.listen(1)
    srv.settimeout(wait_s)
    from .arkit import _local_ip
    print(f"waiting for the phone (app should point at {_local_ip()}, "
          f"auto-reconnects every 2s)...")
    try:
        conn, addr = srv.accept()
    except socket.timeout:
        raise SystemExit("phone never connected — is GazeTeacher open and "
                         "pointed at this Mac's IP?")
    print(f"phone connected from {addr[0]}")

    def send(cmd):
        body = json.dumps({"cmd": cmd}).encode()
        conn.sendall(struct.pack(">I", len(body)) + body)

    if action == "status":
        send("session_start")
        gsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        gsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        gaze_n = 0
        try:
            gsock.bind(("0.0.0.0", GAZE_PORT))
            gsock.settimeout(0.1)
        except OSError:
            gsock = None
        conn.settimeout(0.1)
        frames = 0
        t0 = time.time()
        while time.time() - t0 < 3.0:
            try:
                if conn.recv(65536):
                    frames += 1   # rough: chunks, not exact frames
            except socket.timeout:
                pass
            if gsock:
                try:
                    gsock.recvfrom(4096)
                    gaze_n += 1
                except socket.timeout:
                    pass
        print(f"status: connected; ~{frames / 3:.0f} frame-chunks/s, "
              f"{gaze_n / 3:.0f} gaze pkt/s over 3s")
    else:
        cmd = {"start": "session_start", "stop": "session_stop",
               "on": "stream_on", "off": "stream_off"}[action]
        send(cmd)
        print(f"sent {cmd}")
        time.sleep(0.3)
    conn.close()
    srv.close()
ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE,
             cv2.ROTATE_180]


class PhoneCamera:
    def __init__(self, frame_port=FRAME_PORT, gaze_port=GAZE_PORT,
                 wait_s=None, landmarker="models/face_landmarker.task",
                 on_wait=None):
        """on_wait(elapsed_s) is called ~5x/s while waiting for the phone —
        callers with a window MUST pass it, otherwise their UI freezes
        unpainted (macOS only paints while the event loop is pumped)."""
        import os
        wait_s = wait_s or float(os.environ.get("GAZEKIT_PHONE_WAIT", 60))
        self._frame = None
        self._fresh = threading.Event()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._rotation = None
        self._conn = None
        self._gaze_n = 0
        self._frames_n = 0
        self.gaze_path = None

        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._srv.bind(("0.0.0.0", frame_port))
        except OSError:
            raise SystemExit(f"port {frame_port} in use — another "
                             "`--camera phone` process is running")
        self._srv.listen(1)
        self._srv.settimeout(1.0)
        threading.Thread(target=self._frame_loop, daemon=True).start()
        threading.Thread(target=self._gaze_loop, args=(gaze_port,),
                         daemon=True).start()

        print(f"waiting for GazeTeacher frames on tcp:{frame_port} "
              "(app auto-reconnects; keep it in the foreground)...",
              flush=True)
        t0 = time.time()
        while not self._fresh.wait(0.2):
            elapsed = time.time() - t0
            if on_wait is not None:
                on_wait(elapsed)
            if int(elapsed) % 5 == 0 and elapsed - int(elapsed) < 0.2:
                print(f"  ...{elapsed:.0f}s  tcp-connected="
                      f"{self._conn is not None}  gaze="
                      f"{self._gaze_n}", flush=True)
            write_status(owner="camera", connected=self._conn is not None,
                         frames=0, gaze=self._gaze_n, waiting=round(elapsed))
            if elapsed > wait_s:
                self.release()
                raise SystemExit(
                    "no frames from the phone. Checks: app in the "
                    "FOREGROUND (ARKit stops in background), phone not "
                    "locked, link line shows 'connected'.")
        self._auto_orient(landmarker)

    # --- network loops -------------------------------------------------
    def _frame_loop(self):
        conn = None
        buf = b""
        while not self._stop.is_set():
            if conn is None:
                try:
                    conn, addr = self._srv.accept()
                    conn.settimeout(2.0)
                    self._conn = conn
                    print(f"phone connected from {addr[0]}")
                    buf = b""
                    # the phone is armed but idle until told — drive it
                    self.send_cmd("session_start")
                except socket.timeout:
                    continue
            try:
                chunk = conn.recv(65536)
                if not chunk:
                    conn.close(); conn = None
                    continue
                buf += chunk
                while len(buf) >= 4:
                    n = struct.unpack(">I", buf[:4])[0]
                    if n == 0:            # heartbeat ping from the phone
                        buf = buf[4:]
                        continue
                    if n > 8_000_000:     # desync guard
                        buf = b""
                        break
                    if len(buf) < 4 + n:
                        break
                    msg, buf = buf[4:4 + n], buf[4 + n:]
                    try:
                        rec = json.loads(msg)
                        if rec.get("type") == "frame":
                            raw = base64.b64decode(rec["jpg"])
                            img = cv2.imdecode(
                                np.frombuffer(raw, np.uint8),
                                cv2.IMREAD_COLOR)
                            if img is not None:
                                with self._lock:
                                    self._frame = img
                                self._fresh.set()
                            self._frames_n += 1
                            if self._frames_n % 10 == 0:
                                write_status(owner="camera", connected=True,
                                             frames=self._frames_n,
                                             gaze=self._gaze_n)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
            except socket.timeout:
                continue
            except OSError:
                if conn is not None:
                    conn.close(); conn = None

    def _gaze_loop(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            print(f"(gaze port {port} busy — teacher stream not recorded "
                  "by this process)")
            return
        sock.settimeout(1.0)
        Path("data/arkit").mkdir(parents=True, exist_ok=True)
        self.gaze_path = Path("data/arkit") / \
            f"stream_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        n = 0
        with open(self.gaze_path, "w") as f:
            while not self._stop.is_set():
                try:
                    data, _ = sock.recvfrom(4096)
                    pkt = json.loads(data)
                    pkt["t_recv"] = time.time()
                    f.write(json.dumps(pkt) + "\n")
                    n += 1
                    self._gaze_n = n
                    if n % 15 == 0:   # keep the file tailable (monitor/fit)
                        f.flush()
                except (socket.timeout, json.JSONDecodeError):
                    continue
                except OSError:
                    break
        sock.close()

    # --- orientation ---------------------------------------------------
    def _auto_orient(self, landmarker):
        from .tracker import FaceTracker
        probe = FaceTracker(landmarker)
        try:
            for rot in ROTATIONS:
                hits = 0
                for _ in range(6):
                    self._fresh.clear()
                    if not self._fresh.wait(2.0):
                        break
                    with self._lock:
                        img = self._frame.copy()
                    if rot is not None:
                        img = cv2.rotate(img, rot)
                    if probe.process(img).ok:
                        hits += 1
                    if hits >= 2:
                        self._rotation = rot
                        name = {None: "none",
                                cv2.ROTATE_90_CLOCKWISE: "90cw",
                                cv2.ROTATE_90_COUNTERCLOCKWISE: "90ccw",
                                cv2.ROTATE_180: "180"}[rot]
                        print(f"orientation locked: rotate={name}")
                        return
            print("warning: no face found in any rotation — using frames "
                  "as-is (fix framing and restart if tracking fails)")
        finally:
            probe.close()

    # --- cv2.VideoCapture-compatible surface ---------------------------
    def isOpened(self):
        return not self._stop.is_set()

    def read(self):
        self._fresh.clear()
        if not self._fresh.wait(1.0):
            return False, None
        with self._lock:
            img = self._frame.copy()
        if self._rotation is not None:
            img = cv2.rotate(img, self._rotation)
        return True, img

    def grab(self):
        return True

    def set(self, *a):
        return True

    def send_cmd(self, cmd: str):
        """Length-prefixed control message to the phone (PHONE_PROTOCOL)."""
        conn = self._conn
        if conn is None:
            return False
        try:
            body = json.dumps({"cmd": cmd}).encode()
            conn.sendall(struct.pack(">I", len(body)) + body)
            return True
        except OSError:
            return False

    def release(self):
        self.send_cmd("session_stop")
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass
