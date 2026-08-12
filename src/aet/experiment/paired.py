"""Paired analysis across arms — differencing within a seed, not pooling across them.

``aet compare`` runs an unpaired Welch t-test over methods pooled by seed. For a design where every
seed is run under every arm that discards the pairing, and between-seed variance is usually the
largest term. This module keeps seeds matched and reports the paired difference, its bootstrap
interval, and — for time-to-X metrics — a survival estimate that handles runs which never got there.

Nothing here invents a number a pilot cannot support: :func:`compare_arms` carries
``underpowered`` straight through from the signed-rank test, and reports how many seeds it actually
paired rather than how many were requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from aet.core.metrics import kaplan_meier, paired_bootstrap_ci, wilcoxon_signed_rank


@dataclass
class ArmComparison:
    """The paired result for one metric between two arms."""

    metric: str
    arm_a: str
    arm_b: str
    seeds: list[int] = field(default_factory=list)
    a_values: list[Any] = field(default_factory=list)
    b_values: list[Any] = field(default_factory=list)
    n_paired: int = 0
    n_unpaired: int = 0
    mean_delta: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    median_delta: float | None = None
    p_value: float | None = None
    underpowered: bool = True
    min_achievable_p: float | None = None
    survival: dict | None = None

    def summary(self) -> str:
        if self.n_paired == 0:
            return f"{self.metric}: no paired seeds ({self.n_unpaired} unpaired)"
        ci = ("" if self.ci_lower is None
              else f" [{self.ci_lower:g}, {self.ci_upper:g}]")
        p = "" if self.p_value is None else f", p={self.p_value:g}"
        warn = " (UNDERPOWERED: no result at this n could reach p<0.05)" if self.underpowered else ""
        return (f"{self.metric}: {self.arm_a} - {self.arm_b} = "
                f"{self.mean_delta:g}{ci} over {self.n_paired} paired seed(s){p}{warn}")


def pair_by_seed(a_by_seed: Mapping[int, Any], b_by_seed: Mapping[int, Any]
                 ) -> tuple[list[int], list[Any], list[Any], int]:
    """Seeds present in BOTH mappings, with their values, plus how many were dropped.

    A seed present in one arm and not the other cannot be differenced. Dropping it silently would
    make a partially-completed grid look like a smaller but complete one, so the count comes back.
    """
    shared = sorted(set(a_by_seed) & set(b_by_seed))
    dropped = len(set(a_by_seed) ^ set(b_by_seed))
    return shared, [a_by_seed[s] for s in shared], [b_by_seed[s] for s in shared], dropped


def compare_arms(metric: str, arm_a: str, arm_b: str,
                 a_by_seed: Mapping[int, Any], b_by_seed: Mapping[int, Any],
                 *, confidence: float = 0.95, seed: int = 0,
                 censored_a: Mapping[int, bool] | None = None,
                 censored_b: Mapping[int, bool] | None = None) -> ArmComparison:
    """Paired comparison of one metric between two arms.

    ``censored_*`` marks runs that hit their budget without reaching the endpoint. When supplied, a
    Kaplan-Meier estimate is attached for each arm — the only correct way to summarise a
    time-to-success metric where some runs never succeeded. It is attached rather than folded into
    ``mean_delta`` on purpose: a mean over censored durations is not a mean of anything.
    """
    seeds, a_vals, b_vals, dropped = pair_by_seed(a_by_seed, b_by_seed)
    cmp_ = ArmComparison(metric=metric, arm_a=arm_a, arm_b=arm_b, seeds=seeds,
                         a_values=a_vals, b_values=b_vals, n_unpaired=dropped)

    boot = paired_bootstrap_ci(a_vals, b_vals, confidence=confidence, seed=seed)
    cmp_.n_paired = boot["n_pairs"]
    cmp_.mean_delta = boot["mean_delta"]
    cmp_.ci_lower, cmp_.ci_upper = boot["lower"], boot["upper"]

    w = wilcoxon_signed_rank(a_vals, b_vals)
    cmp_.median_delta = w["median_delta"]
    cmp_.p_value = w["p_value"]
    cmp_.underpowered = w["underpowered"]
    cmp_.min_achievable_p = w["min_achievable_p"]

    if censored_a is not None or censored_b is not None:
        cmp_.survival = {}
        for name, vals, cens in ((arm_a, a_vals, censored_a), (arm_b, b_vals, censored_b)):
            if cens is None:
                continue
            events = [0 if cens.get(s, False) else 1 for s in seeds]
            cmp_.survival[name] = kaplan_meier(vals, events)
    return cmp_


def compare_all(metric: str, by_arm: Mapping[str, Mapping[int, Any]],
                baseline: str, **kw) -> list[ArmComparison]:
    """Every arm against one baseline, for one metric."""
    return [compare_arms(metric, arm, baseline, vals, by_arm[baseline], **kw)
            for arm, vals in by_arm.items() if arm != baseline]
