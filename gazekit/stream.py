"""Headless gaze stream for other apps (e.g. GOZ).

After the usual 3-point quick alignment (a brief fullscreen window), the
window closes and gaze is published as one UDP JSON datagram per camera
frame to 127.0.0.1:5590 (see docs/STREAM_PROTOCOL.md):

  {"t": <unix s>, "x": <screen pt>, "y": <screen pt>, "sw": W, "sh": H,
   "valid": bool, "blink": bool, "blink_score": float,
   "yaw": deg, "pitch": deg, "roll": deg, "face": bool}

x/y are in macOS screen points (top-left origin), the same space a browser
reports for window.screenX + element rects at 100% zoom. Blinks freeze the
point (valid=false) exactly like `live`.
"""

import json
import socket
import time

import cv2

from . import ui
from .camera import open_camera, read_mirrored
from .live import BlinkGate, _quick_align, build_predictor
from .tracker import FaceTracker

STREAM_PORT = 5590


def run(camera_index=0, backend="ridge", model_path=None, host="127.0.0.1",
        port=STREAM_PORT, align=True,
        landmarker="models/face_landmarker.task", screen=None):
    from .filters import GazeSmoother
    from .screen import screen_size
    sw, sh = screen or screen_size()
    predict, _, _ = build_predictor(backend, model_path, sw, sh)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = open_camera(camera_index)
    tracker = FaceTracker(landmarker)
    smoother, gate = GazeSmoother(), BlinkGate()
    ax, bx, ay, by = (1.0, 0.0, 1.0, 0.0)
    sent = valid = blinks = 0
    was_blink = False
    t_start = last_report = time.monotonic()
    try:
        if align:
            win = ui.FullscreenWindow("gazekit-stream-align", (sw, sh))
            ax, bx, ay, by = _quick_align(win, cap, tracker, predict)
            win.close()
            cv2.waitKey(1)
            print(f"aligned: gain=({ax:.2f},{ay:.2f}) "
                  f"offset=({bx:.0f},{by:.0f})")
        print(f"streaming gaze to udp://{host}:{port} — Ctrl+C to stop")
        last_xy = (sw / 2, sh / 2)
        while True:
            frame = read_mirrored(cap)
            if frame is None:
                continue
            obs = tracker.process(frame)
            now = time.monotonic()
            frozen = gate.update(obs)
            ok = False
            if not frozen:
                p = predict(obs)
                if p is not None:
                    px = min(max(ax * float(p[0]) + bx, 0.0), sw - 1.0)
                    py = min(max(ay * float(p[1]) + by, 0.0), sh - 1.0)
                    last_xy = smoother.apply(px, py, now)
                    ok = True
            is_blink = obs.ok and frozen
            blinks += is_blink and not was_blink
            was_blink = is_blink
            msg = dict(t=time.time(), x=round(last_xy[0], 1),
                       y=round(last_xy[1], 1), sw=sw, sh=sh, valid=ok,
                       blink=is_blink, blink_score=round(obs.blink, 3),
                       yaw=round(obs.yaw, 1), pitch=round(obs.pitch, 1),
                       roll=round(obs.roll, 1), face=obs.ok)
            sock.sendto(json.dumps(msg).encode(), (host, port))
            sent += 1
            valid += ok
            if now - last_report > 5:
                el = now - t_start
                print(f"  {sent / el:.0f} Hz  valid {100 * valid / max(sent, 1):.0f}%"
                      f"  blinks {blinks}  ({el:.0f}s)")
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        cap.release()
        sock.close()
        cv2.destroyAllWindows()
    return {"samples": sent, "blinks": blinks,
            "valid_frac": round(valid / max(sent, 1), 3)}
