"""OpenCV fullscreen UI helpers: window, animated targets, HUD text."""

import cv2
import numpy as np

BG = (18, 18, 18)
ACCENT = (255, 180, 40)   # BGR
GOOD = (90, 200, 90)
BAD = (60, 60, 230)
WHITE = (235, 235, 235)


def say(text: str):
    """Non-blocking voice prompt (macOS `say`) — for moments when the user
    can't or shouldn't read the screen."""
    import subprocess
    try:
        subprocess.Popen(["say", text], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception:
        pass


class FullscreenWindow:
    def __init__(self, name: str, size: tuple[int, int]):
        self.name = name
        self.w, self.h = size
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def canvas(self) -> np.ndarray:
        img = np.empty((self.h, self.w, 3), dtype=np.uint8)
        img[:] = BG
        return img

    def show(self, img: np.ndarray) -> int:
        cv2.imshow(self.name, img)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyWindow(self.name)


def draw_target(img, x, y, phase: float, color=ACCENT):
    """Animated calibration target. phase 0->1: ring shrinks onto the dot."""
    x, y = int(round(x)), int(round(y))
    r_out = int(34 - 22 * min(phase, 1.0))
    cv2.circle(img, (x, y), r_out, color, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 4, WHITE, -1, cv2.LINE_AA)


def draw_gaze_dot(img, x, y, color=ACCENT, frozen=False):
    x, y = int(round(x)), int(round(y))
    c = (120, 120, 120) if frozen else color
    cv2.circle(img, (x, y), 14, c, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 5, c, -1, cv2.LINE_AA)


def center_text(img, text, y, scale=0.9, color=WHITE, thickness=2):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(img, text, ((img.shape[1] - tw) // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def progress_bar(img, frac, y, w=420, color=GOOD):
    x0 = (img.shape[1] - w) // 2
    cv2.rectangle(img, (x0, y), (x0 + w, y + 14), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y), (x0 + int(w * max(0.0, min(frac, 1.0))), y + 14),
                  color, -1, cv2.LINE_AA)


def thumbnail(img, frame, obs=None, size=240):
    """Small camera preview in the bottom-left corner with landmark overlay."""
    h, w = frame.shape[:2]
    tw, th = size, int(size * h / w)
    small = cv2.resize(frame, (tw, th))
    if obs is not None and obs.ok and obs.landmarks_px is not None:
        sx, sy = tw / w, th / h
        for i in (468, 473):
            p = obs.landmarks_px[i]
            cv2.circle(small, (int(p[0] * sx), int(p[1] * sy)), 2, ACCENT, -1)
    y0 = img.shape[0] - th - 16
    img[y0:y0 + th, 16:16 + tw] = small
    cv2.rectangle(img, (16, y0), (16 + tw, y0 + th), (90, 90, 90), 1)
