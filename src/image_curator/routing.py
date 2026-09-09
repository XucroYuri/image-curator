"""Conservative staged decision routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .classification import ClassificationDecision


@dataclass(frozen=True)
class RouteDecision:
    state: str
    stage: str
    label: str | None
    reason: str

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def route(*, rule_accepted: bool = False, rule_reason: str = "", classification: ClassificationDecision | None = None) -> RouteDecision:
    """Apply a low-cost rule, then reference classification, else queue review."""
    if rule_accepted:
        return RouteDecision("accepted", "rule", None, rule_reason or "accepted by configured rule")
    if classification is not None and classification.state == "accepted":
        return RouteDecision("accepted", "reference", classification.label, "high-confidence reference match")
    return RouteDecision("review", "review", None, "no rule or reference decision passed thresholds")
