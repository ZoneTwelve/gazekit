"""Explainable, non-destructive tracker-quality scoring for schema-v2 rows.

Collection gates decide whether a frame is valid. This module only describes
how reliable an already-accepted signal looks, so training can softly weight
it without deleting useful pose-variation data.
"""

from __future__ import annotations

from math import hypot
from typing import Any


QUALITY_WEIGHT_FLOOR = 0.55


def clamp01(value: Any, default: float = 0.0) -> float:
    """Return a finite numeric value constrained to the unit interval."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN
        return default
    return max(0.0, min(1.0, value))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value else default


def record_quality_score(record: dict) -> float:
    """Read a row's quality safely; legacy rows intentionally mean 1.0."""
    if "quality_score" not in record:
        return 1.0
    return clamp01(record.get("quality_score"), default=1.0)


def quality_weight(score: float) -> float:
    """Bounded soft training weight specified by ``docs/DATA_STANDARD.md``."""
    return QUALITY_WEIGHT_FLOOR + (1.0 - QUALITY_WEIGHT_FLOOR) * clamp01(
        score, default=1.0)


def _lighting_score(brightness: Any) -> float:
    """Prefer the same usable brightness band as the environment gate."""
    try:
        value = float(brightness)
    except (TypeError, ValueError):
        return 0.0
    if value <= 15.0 or value >= 250.0:
        return 0.0
    if value < 40.0:
        return (value - 15.0) / 25.0
    if value <= 220.0:
        return 1.0
    return (250.0 - value) / 30.0


def _frame_size(obs) -> list[int] | None:
    extras = getattr(obs, "extras", {})
    raw = extras.get("frame_size") if isinstance(extras, dict) else None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        width, height = int(raw[0]), int(raw[1])
    except (TypeError, ValueError):
        return None
    return [width, height] if width > 0 and height > 0 else None


def observation_quality(obs) -> dict:
    """Return the serializable v2 quality fields for an observation.

    Deliberate head movement is valuable training coverage, so pose has a
    deliberately shallow penalty. The resulting score is advisory only; its
    smallest possible training contribution is enforced by ``quality_weight``.
    """
    frame_size = _frame_size(obs)
    blink = clamp01(getattr(obs, "blink", 1.0))
    eyes = 1.0 - clamp01(blink / 0.35)

    pose_mag = hypot(_number(getattr(obs, "yaw", 0.0)),
                      _number(getattr(obs, "pitch", 0.0)))
    pose = 1.0 - 0.40 * min(max(pose_mag, 0.0) / 45.0, 1.0)

    interocular = max(_number(getattr(obs, "interocular_px", 0.0)), 0.0)
    if frame_size is not None:
        ratio = interocular / frame_size[0]
        distance = clamp01((ratio - 0.02) / 0.04)
    else:
        # Synthetic/legacy-style observations have no frame geometry. This
        # fallback is intentionally broad and only affects newly written rows.
        distance = clamp01((interocular - 20.0) / 60.0)

    lighting = _lighting_score(getattr(obs, "brightness", 0.0))
    components = {
        "eyes": round(eyes, 4),
        "pose": round(pose, 4),
        "distance": round(distance, 4),
        "lighting": round(lighting, 4),
    }
    score = (0.40 * components["eyes"] + 0.25 * components["pose"]
             + 0.20 * components["distance"]
             + 0.15 * components["lighting"])
    return {
        "quality_score": round(clamp01(score), 4),
        "quality_components": components,
        "frame_size": frame_size,
    }
