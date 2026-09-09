"""NudeNet-compatible, per-class non-maximum suppression."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

NUDENET_CLASSES = (
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
    "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE",
    "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
)
EXPLICIT_CLASSES = frozenset({"BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
                              "ANUS_EXPOSED", "MALE_GENITALIA_EXPOSED"})
INTIMATE_COVERED_CLASSES = frozenset({"FEMALE_GENITALIA_COVERED", "ANUS_COVERED",
                                      "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED"})


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    overlap_w = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    overlap_h = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    union = lw * lh + rw * rh - overlap_w * overlap_h
    return overlap_w * overlap_h / union if union else 0.0


def _as_rows(output: Any) -> list[list[float]]:
    """Accept a nested model output, including a single leading batch dimension."""
    rows = output.tolist() if hasattr(output, "tolist") else output
    while isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list):
        rows = rows[0]
    if not isinstance(rows, list):
        raise ValueError("NudeNet output must be a nested sequence")
    if rows and isinstance(rows[0], (int, float)):
        rows = [rows]
    expected_columns = 4 + len(NUDENET_CLASSES)
    if rows and (
        (len(rows) == expected_columns and len(rows[0]) != expected_columns)
        or len(rows[0]) < expected_columns <= len(rows)
    ):
        rows = [list(column) for column in zip(*rows)]
    if any(not isinstance(row, list) or len(row) < expected_columns for row in rows):
        raise ValueError("NudeNet output shape is not recognized")
    return [[float(value) for value in row] for row in rows]


def classwise_nms(output: Any, *, score_threshold: float = 0.15, iou_threshold: float = 0.45,
                  canvas_size: int = 320) -> list[dict[str, Any]]:
    """Decode center-format NudeNet boxes and suppress overlaps within each class."""
    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    grouped: dict[str, list[tuple[list[float], float]]] = defaultdict(list)
    for row in _as_rows(output):
        scores = row[4:4 + len(NUDENET_CLASSES)]
        class_index = max(range(len(scores)), key=scores.__getitem__)
        score = scores[class_index]
        if score < score_threshold:
            continue
        cx, cy, width, height = row[:4]
        x, y = max(0.0, cx - width / 2), max(0.0, cy - height / 2)
        width, height = max(0.0, min(width, canvas_size - x)), max(0.0, min(height, canvas_size - y))
        if width * height >= 4:
            grouped[NUDENET_CLASSES[class_index]].append(([x, y, width, height], score))
    detections: list[dict[str, Any]] = []
    for name, candidates in grouped.items():
        kept: list[tuple[list[float], float]] = []
        for box, score in sorted(candidates, key=lambda item: item[1], reverse=True):
            if all(_iou(box, existing[0]) <= iou_threshold for existing in kept):
                kept.append((box, score))
        detections.extend({"class": name, "confidence": round(score, 6),
                           "area_ratio": round(box[2] * box[3] / canvas_size**2, 6),
                           "box": [round(value, 2) for value in box]} for box, score in kept)
    return sorted(detections, key=lambda item: item["confidence"], reverse=True)


def detection_summary(detections: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize detector output without imposing a curation decision."""
    maxima: dict[str, float] = defaultdict(float)
    for detection in detections:
        maxima[str(detection["class"])] = max(maxima[str(detection["class"])], float(detection["confidence"]))
    return {"class_maxima": dict(sorted(maxima.items())),
            "explicit_score": max((maxima[name] for name in EXPLICIT_CLASSES), default=0.0),
            "intimate_covered_score": max((maxima[name] for name in INTIMATE_COVERED_CLASSES), default=0.0),
            "detections": list(detections)[:24]}
