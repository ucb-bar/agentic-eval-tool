"""Structured rubric scoring for aet evaluations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RubricCriterion:
    name: str
    weight: float
    description: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"RubricCriterion weight must be in [0.0, 1.0], got {self.weight}")


def validate_rubric(criteria: list[RubricCriterion]) -> None:
    """Raise ValueError if weights don't sum to 1.0 (within 1e-6 tolerance)."""
    total = sum(c.weight for c in criteria)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"RubricCriterion weights must sum to 1.0, got {total:.6f}")


def compute_weighted_score(scores: dict[str, float], criteria: list[RubricCriterion]) -> float:
    """Weighted sum of scores. Missing criteria count as 0. Does NOT validate weight sum."""
    return round(sum(scores.get(c.name, 0.0) * c.weight for c in criteria), 6)
