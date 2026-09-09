"""Auditable calibration for explicit, human-labeled vector validation data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .classification import ClassificationConfig, classify_open_set


@dataclass(frozen=True)
class LabeledVector:
    """A vector with an explicit ground-truth label supplied outside file paths."""

    label: str | None
    vector: tuple[float, ...]
    identifier: str = ""


@dataclass(frozen=True)
class CalibrationMetrics:
    """Classifier metrics for one threshold pair."""

    min_similarity: float
    min_margin: float
    total: int
    accepted: int
    coverage: float
    accepted_accuracy: float | None
    unknown_total: int
    unknown_false_accept: int
    unknown_false_accept_rate: float | None
    confusion: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_thresholds(records: Sequence[LabeledVector], references: Mapping[str, Sequence[Sequence[float]]], *,
                        config: ClassificationConfig) -> CalibrationMetrics:
    """Evaluate thresholds against explicit labels; unknown labels are never inferred from paths."""
    if not records:
        raise ValueError("at least one explicitly labeled validation vector is required")
    reference_labels = set(references)
    confusion: dict[str, Counter[str]] = {}
    accepted = correct = unknown_total = unknown_false_accept = 0
    for record in records:
        actual = record.label if record.label in reference_labels else "unknown"
        decision = classify_open_set(record.vector, references, config=config)
        predicted = decision.label if decision.label is not None else "unknown"
        confusion.setdefault(actual, Counter())[predicted] += 1
        if decision.state == "accepted":
            accepted += 1
            correct += predicted == actual
            if actual == "unknown":
                unknown_false_accept += 1
        if actual == "unknown":
            unknown_total += 1
    return CalibrationMetrics(
        config.min_similarity, config.min_margin, len(records), accepted, accepted / len(records),
        correct / accepted if accepted else None, unknown_total, unknown_false_accept,
        unknown_false_accept / unknown_total if unknown_total else None,
        {actual: dict(sorted(predictions.items())) for actual, predictions in sorted(confusion.items())},
    )


def calibrate_thresholds(records: Sequence[LabeledVector], references: Mapping[str, Sequence[Sequence[float]]], *,
                         similarity_thresholds: Sequence[float], margin_thresholds: Sequence[float],
                         top_k: int = 3, min_accepted_accuracy: float = 0.90,
                         max_unknown_false_accept_rate: float = 0.05, min_accepted: int = 5,
                         min_unknown: int = 1) -> dict[str, Any]:
    """Evaluate every explicit threshold pair and select a conservative recommendation.

    A candidate must meet explicit accuracy, unknown false-accept, accepted
    count, and unknown count gates before it can be recommended.  Among those,
    the recommendation favors coverage, then accepted accuracy and the lowest
    unknown false-accept rate.  It is a report for human review, not an
    automatic production-policy change.
    """
    results = [evaluate_thresholds(records, references,
                                   config=ClassificationConfig(similarity, margin, top_k))
               for similarity in sorted(set(similarity_thresholds))
               for margin in sorted(set(margin_thresholds))]
    if not results:
        raise ValueError("at least one similarity and margin threshold are required")
    if not 0 <= min_accepted_accuracy <= 1 or not 0 <= max_unknown_false_accept_rate <= 1:
        raise ValueError("accuracy and false-accept targets must be between 0 and 1")
    if min_accepted < 1 or min_unknown < 0:
        raise ValueError("minimum sample targets must be non-negative (accepted must be positive)")

    def ranking(metrics: CalibrationMetrics) -> tuple[float, float, float, float, float]:
        accuracy = metrics.accepted_accuracy if metrics.accepted_accuracy is not None else -1.0
        false_accept = metrics.unknown_false_accept_rate
        safety = -(false_accept if false_accept is not None else 0.0)
        return metrics.coverage, accuracy, safety, metrics.min_similarity, metrics.min_margin

    eligible = [result for result in results
                if result.accepted >= min_accepted
                and result.unknown_total >= min_unknown
                and result.accepted_accuracy is not None
                and result.accepted_accuracy >= min_accepted_accuracy
                and result.unknown_false_accept_rate is not None
                and result.unknown_false_accept_rate <= max_unknown_false_accept_rate]
    recommended = max(eligible, key=ranking) if eligible else None
    return {
        "label_source": "explicit validation JSON; paths and directory names are not labels",
        "reference_labels": sorted(references),
        "validation_count": len(records),
        "target_gates": {"min_accepted_accuracy": min_accepted_accuracy,
                         "max_unknown_false_accept_rate": max_unknown_false_accept_rate,
                         "min_accepted": min_accepted, "min_unknown": min_unknown},
        "recommended": recommended.as_dict() if recommended else None,
        "recommendation_eligible": bool(recommended),
        "candidates": [result.as_dict() for result in results],
    }
