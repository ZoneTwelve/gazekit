"""Structured, advisory data-collection feedback from evaluation reports.

This module deliberately never mutates samples, labels, pruning state, or
models. Those operations remain behind the collection and deploy gates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


SCHEMA_VERSION = 1
WEAK_REGION_RATIO = 1.20


def camera_domain(source: Any) -> str:
    """Map the persisted source setting to a stable artifact name."""
    return "phone" if str(source) == "phone" else "webcam"


def feedback_path(output_dir: str | Path, source: Any) -> Path:
    return Path(output_dir) / f"feedback_{camera_domain(source)}.json"


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def _region_feedback(region_map: Any) -> tuple[list[list[float]] | None,
                                                list[dict], list[dict]]:
    """Return bounded collection weights plus weak/unmeasured cells."""
    if not isinstance(region_map, list) or len(region_map) != 3:
        return None, [], []
    cells = []
    for row in region_map:
        if not isinstance(row, list) or len(row) != 3:
            return None, [], []
        cells.extend(_number(value) for value in row)
    measured = [value for value in cells if value is not None]
    if not measured:
        return None, [], [{"row": i, "col": j}
                           for i in range(3) for j in range(3)]

    baseline = median(measured)
    weights, weak, unmeasured = [], [], []
    for row_idx, row in enumerate(region_map):
        weights_row = []
        for col_idx, raw in enumerate(row):
            error = _number(raw)
            if error is None:
                weights_row.append(1.0)
                unmeasured.append({"row": row_idx, "col": col_idx})
                continue
            relative = error / baseline if baseline > 1e-6 else 1.0
            weights_row.append(round(max(0.75, min(1.50, relative)), 2))
            if relative >= WEAK_REGION_RATIO:
                weak.append({
                    "row": row_idx,
                    "col": col_idx,
                    "error_px": round(error, 1),
                    "relative_error": round(relative, 2),
                })
        weights.append(weights_row)
    weak.sort(key=lambda cell: cell["error_px"], reverse=True)
    return weights, weak, unmeasured


def _collect_action(scenario: str, reason: str, metric: str,
                    value: Any, threshold: Any) -> dict:
    return {
        "type": "collect",
        "advisory": True,
        "scenario": scenario,
        "reason": reason,
        "metric": metric,
        "value": value,
        "threshold": threshold,
    }


def build_feedback(report: dict, source: Any) -> dict:
    """Convert an `evaluate()` report into the safe feedback schema."""
    coverage = dict(report.get("coverage") or {})
    quality = dict(report.get("quality") or {})
    weights, weak_regions, unmeasured_regions = _region_feedback(
        report.get("region_map_px"))
    actions = []
    scenarios = []

    days = _number(coverage.get("days"))
    if days is not None and days < 3:
        action = _collect_action("daily", "cross-day lighting coverage",
                                 "coverage.days", int(days), 3)
        actions.append(action)
        scenarios.append(action)

    pose = _number(coverage.get("pose_gt10deg_pct"))
    if pose is not None and pose < 10:
        action = _collect_action("vor", "head-pose coverage",
                                 "coverage.pose_gt10deg_pct", pose, 10)
        actions.append(action)
        scenarios.append(action)

    edges = _number(coverage.get("screen_edge_pct"))
    if edges is not None and edges < 8:
        action = _collect_action("edges", "screen-edge coverage",
                                 "coverage.screen_edge_pct", edges, 8)
        actions.append(action)
        scenarios.append(action)

    if weak_regions and weights is not None:
        actions.append({
            "type": "adjust_region_sampling",
            "advisory": True,
            "reason": "cross-session screen-region error",
            "metric": "region_map_px.relative_to_median",
            "threshold": WEAK_REGION_RATIO,
            "weights": weights,
        })

    low_quality = _number(quality.get("low_lt_0_7_pct"))
    if low_quality is not None and low_quality > 15:
        actions.append({
            "type": "review_quality",
            "advisory": True,
            "reason": "elevated low-quality accepted samples",
            "metric": "quality.low_lt_0_7_pct",
            "value": low_quality,
            "threshold": 15,
        })

    p50 = _number(report.get("loso_p50_px"))
    p95 = _number(report.get("loso_p95_px"))
    if p50 is not None and p95 is not None and p50 > 1e-6 and p95 > 2.2 * p50:
        actions.append({
            "type": "verify",
            "advisory": True,
            "mode": "path",
            "reason": "heavy cross-session error tail",
            "metric": "loso_p95_px / loso_p50_px",
            "value": round(p95 / p50, 2),
            "threshold": 2.2,
        })

    summary_keys = ("samples", "sessions", "loso_aligned_px", "loso_px",
                    "loto_px", "coverage", "quality")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "camera_domain": camera_domain(source),
        "summary": {key: report[key] for key in summary_keys if key in report},
        "actions": actions,
        "collection_suggestions": {
            "weak_regions": weak_regions,
            "unmeasured_regions": unmeasured_regions,
            "region_sampling_weights": weights,
            "scenarios": scenarios,
        },
        "recommendations": list(report.get("recommendations") or []),
    }


def write_feedback(report: dict, source: Any,
                   output_dir: str | Path = "data/eval") -> tuple[Path, dict]:
    """Atomically write the source-specific feedback artifact."""
    path = feedback_path(output_dir, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    feedback = build_feedback(report, source)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(feedback, indent=2, sort_keys=True) + "\n")
    temp.replace(path)
    return path, feedback
