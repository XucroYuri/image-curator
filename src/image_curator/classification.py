"""Small, dependency-free open-set reference classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any


def normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """Return a unit vector, rejecting malformed or zero vectors."""
    values = tuple(float(value) for value in vector)
    if not values:
        raise ValueError("vector must not be empty")
    magnitude = sqrt(sum(value * value for value in values))
    if magnitude == 0:
        raise ValueError("vector must not be zero")
    return tuple(value / magnitude for value in values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate cosine similarity after validating matching dimensions."""
    if len(left) != len(right):
        raise ValueError(f"vector dimensions differ: {len(left)} and {len(right)}")
    return sum(a * b for a, b in zip(normalize(left), normalize(right), strict=True))


@dataclass(frozen=True)
class ClassificationConfig:
    min_similarity: float = 0.80
    min_margin: float = 0.05
    top_k: int = 3

    def __post_init__(self) -> None:
        if not -1 <= self.min_similarity <= 1:
            raise ValueError("min_similarity must be between -1 and 1")
        if self.min_margin < 0 or self.top_k < 1:
            raise ValueError("min_margin must be non-negative and top_k must be positive")


@dataclass(frozen=True)
class ClassificationDecision:
    label: str | None
    state: str
    similarity: float
    margin: float
    candidates: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidates"] = [{"label": label, "similarity": round(score, 6)}
                                for label, score in self.candidates]
        result["similarity"] = round(self.similarity, 6)
        result["margin"] = round(self.margin, 6)
        return result


def _center(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("reference class must contain at least one vector")
    normalised = [normalize(vector) for vector in vectors]
    dimensions = len(normalised[0])
    if any(len(vector) != dimensions for vector in normalised):
        raise ValueError("reference vectors in a class must have matching dimensions")
    return normalize(tuple(sum(vector[index] for vector in normalised) / len(normalised)
                           for index in range(dimensions)))


def classify_open_set(vector: Sequence[float], references: Mapping[str, Sequence[Sequence[float]]], *,
                      config: ClassificationConfig | None = None) -> ClassificationDecision:
    """Classify against class centers and reject insufficiently distinct matches.

    Each score combines the normalized class-center similarity with an average
    of the class's top-k individual reference similarities.  Both best
    similarity and its gap from the runner-up must
    pass configured thresholds before a label is accepted.
    """
    config = config or ClassificationConfig()
    query = normalize(vector)
    scores: list[tuple[str, float]] = []
    for label, examples in references.items():
        if not label.strip():
            raise ValueError("reference labels must not be blank")
        normalized_examples = [normalize(example) for example in examples]
        if not normalized_examples:
            raise ValueError(f"reference class must contain at least one vector: {label}")
        if any(len(example) != len(query) for example in normalized_examples):
            raise ValueError("reference vector dimension differs from query")
        example_scores = sorted((sum(a * b for a, b in zip(query, example, strict=True))
                                 for example in normalized_examples), reverse=True)
        top_scores = example_scores[:config.top_k]
        reference_aggregate = sum(top_scores) / len(top_scores)
        center_score = sum(a * b for a, b in zip(query, _center(normalized_examples), strict=True))
        scores.append((label, (reference_aggregate + center_score) / 2))
    if not scores:
        raise ValueError("at least one reference class is required")
    ordered = sorted(scores, key=lambda item: (-item[1], item[0]))
    best_label, best_score = ordered[0]
    margin = best_score - (ordered[1][1] if len(ordered) > 1 else -1.0)
    accepted = best_score >= config.min_similarity and margin >= config.min_margin
    return ClassificationDecision(best_label if accepted else None, "accepted" if accepted else "unknown",
                                  best_score, margin, tuple(ordered[:config.top_k]))
