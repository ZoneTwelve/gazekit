"""Face/eye tracking: MediaPipe FaceLandmarker -> gaze features + eye crops."""

from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# Landmark indices (mediapipe 478-point face mesh, image coordinates).
# "Right"/"left" are the subject's eyes as seen in the (mirrored) frame.
R_IRIS = list(range(468, 473))  # center is 468
L_IRIS = list(range(473, 478))  # center is 473
R_CORNERS = (33, 133)   # outer, inner
L_CORNERS = (362, 263)  # inner, outer
R_LIDS = (159, 145)     # top, bottom
L_LIDS = (386, 374)

EYE_CROP_W, EYE_CROP_H = 64, 48


@dataclass
class Observation:
    ok: bool = False
    features: np.ndarray | None = None   # gaze feature vector for the ridge model
    blink: float = 1.0                   # max(eyeBlinkLeft, eyeBlinkRight) blendshape
    yaw: float = 0.0                     # degrees
    pitch: float = 0.0
    roll: float = 0.0
    interocular_px: float = 0.0          # distance between iris centers, pixels
    brightness: float = 0.0              # mean gray value of the face box
    landmarks_px: np.ndarray | None = None
    eye_crops: tuple[np.ndarray, np.ndarray] | None = None  # (right, left) grayscale
    extras: dict = field(default_factory=dict)


def _rotation_to_euler(m: np.ndarray) -> tuple[float, float, float]:
    """Yaw/pitch/roll in degrees from a 3x3 rotation matrix."""
    sy = np.hypot(m[0, 0], m[1, 0])
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(m[2, 1], m[2, 2]))
        yaw = np.degrees(np.arctan2(-m[2, 0], sy))
        roll = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-m[1, 2], m[1, 1]))
        yaw = np.degrees(np.arctan2(-m[2, 0], sy))
        roll = 0.0
    return float(yaw), float(pitch), float(roll)


def _eye_frame_features(pts: np.ndarray, corners, lids, iris_idx) -> tuple[np.ndarray, float]:
    """Iris position in an eye-local frame that cancels head translation/roll/scale.

    Returns ([iris_x, iris_y, lid_rel_y, openness], corner_dist).
    """
    c1, c2 = pts[corners[0]], pts[corners[1]]
    origin = (c1 + c2) / 2.0
    axis = c2 - c1
    dist = float(np.linalg.norm(axis))
    if dist < 1e-6:
        return np.zeros(4), 0.0
    ux = axis / dist
    uy = np.array([-ux[1], ux[0]])
    iris = pts[iris_idx].mean(axis=0)  # mean over 5 iris landmarks denoises
    rel = iris - origin
    ix = float(rel @ ux) / dist
    iy = float(rel @ uy) / dist
    top, bot = pts[lids[0]], pts[lids[1]]
    lid_mid = (top + bot) / 2.0
    lid_rel = float((iris - lid_mid) @ uy) / dist
    openness = float(np.linalg.norm(top - bot)) / dist
    return np.array([ix, iy, lid_rel, openness]), dist


def _crop_eye(gray: np.ndarray, pts: np.ndarray, corners) -> np.ndarray:
    """Fixed-size grayscale crop around one eye, roll-normalized, for the CNN."""
    c1, c2 = pts[corners[0]], pts[corners[1]]
    center = (c1 + c2) / 2.0
    dist = float(np.linalg.norm(c2 - c1))
    angle = float(np.degrees(np.arctan2((c2 - c1)[1], (c2 - c1)[0])))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    scale = EYE_CROP_W / max(dist * 1.7, 1e-6)
    rot = cv2.getRotationMatrix2D(tuple(center), angle, scale)
    rot[0, 2] += EYE_CROP_W / 2 - center[0]
    rot[1, 2] += EYE_CROP_H / 2 - center[1]
    return cv2.warpAffine(gray, rot, (EYE_CROP_W, EYE_CROP_H),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class FaceTracker:
    """Wraps FaceLandmarker (VIDEO mode) and turns frames into Observations."""

    def __init__(self, model_path: str):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._ts_ms = 0

    def close(self):
        self._landmarker.close()

    def process(self, frame_bgr: np.ndarray, want_crops: bool = False) -> Observation:
        """frame_bgr must already be mirrored consistently across the whole app."""
        self._ts_ms += 33
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect_for_video(mp_img, self._ts_ms)

        obs = Observation()
        if not res.face_landmarks:
            return obs

        h, w = frame_bgr.shape[:2]
        pts = np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]],
                       dtype=np.float64)
        obs.landmarks_px = pts

        blink = 0.0
        if res.face_blendshapes:
            scores = {b.category_name: b.score for b in res.face_blendshapes[0]}
            blink = max(scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0))
        obs.blink = blink

        yaw = pitch = roll = 0.0
        tvec = np.zeros(3)
        if res.facial_transformation_matrixes:
            m = np.array(res.facial_transformation_matrixes[0])
            yaw, pitch, roll = _rotation_to_euler(m[:3, :3])
            tvec = m[:3, 3]
            obs.extras["tmatrix"] = m
        obs.yaw, obs.pitch, obs.roll = yaw, pitch, roll

        r_feat, _ = _eye_frame_features(pts, R_CORNERS, R_LIDS, R_IRIS)
        l_feat, _ = _eye_frame_features(pts, L_CORNERS, L_LIDS, L_IRIS)
        obs.interocular_px = float(np.linalg.norm(pts[R_IRIS[0]] - pts[L_IRIS[0]]))

        face_box = pts[np.array([10, 152, 234, 454])]
        x0, y0 = np.clip(face_box.min(axis=0).astype(int), 0, [w - 1, h - 1])
        x1, y1 = np.clip(face_box.max(axis=0).astype(int), 1, [w, h])
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if x1 > x0 and y1 > y0:
            obs.brightness = float(gray[y0:y1, x0:x1].mean())

        obs.features = np.concatenate([
            r_feat, l_feat,
            np.radians([yaw, pitch, roll]),
            tvec / 30.0,
        ])

        if want_crops:
            obs.eye_crops = (_crop_eye(gray, pts, R_CORNERS),
                             _crop_eye(gray, pts, L_CORNERS))

        obs.ok = True
        return obs
