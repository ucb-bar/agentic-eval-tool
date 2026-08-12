"""The arm × seed grid, its execution order, and the fairness check.

Three things a controlled comparison needs that a driver script does not give you for free:

* **Pairing.** Every seed run under every arm, so the analysis can difference within a seed instead
  of pooling across them. Between-seed variance is usually the largest term; pooling throws it away
  and then reports the resulting noise as an absence of effect.
* **Order randomization.** Arms run in a shuffled order so machine load and time-of-day do not
  correlate with arm. Seeded, so the order is reproducible and auditable rather than merely random.
* **A fairness check that fails.** :meth:`ArmMatrix.validate` compares realized arm configurations
  and reports every difference that was not declared as a treatment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from aet.experiment.arm import DEFAULT_HELD_CONSTANT, Arm, ArmDiff, diff_arms


@dataclass(frozen=True)
class Cell:
    """One unit of work: this arm, this seed."""

    arm: str
    seed: int
    order_index: int = 0

    @property
    def run_id(self) -> str:
        return f"{self.arm}_seed{self.seed:03d}"


@dataclass
class MatrixFinding:
    code: str            # ARM001 undeclared difference | ARM002 ladder | ARM003 grid
    detail: str
    arms: tuple[str, ...] = ()


@dataclass
class ArmMatrix:
    """A set of arms crossed with a set of paired seeds."""

    arms: list[Arm]
    seeds: list[int] = field(default_factory=list)
    held_constant: tuple[str, ...] = DEFAULT_HELD_CONSTANT
    #: Arms that are declared ablations. Excluded from the ladder check and reported separately —
    #: an ablation is *supposed* to differ from its base by more than one dimension.
    ablations: tuple[str, ...] = ()

    # ------------------------------------------------------------------ grid

    def cells(self) -> list[Cell]:
        """Every (arm, seed) pair, in declaration order."""
        return [Cell(a.id, s) for a in self.arms for s in self.seeds]

    def execution_order(self, seed: int = 0) -> list[Cell]:
        """Cells shuffled so arm does not correlate with wall-clock position.

        Deterministic given ``seed`` — an unreproducible execution order cannot be audited, and the
        order is exactly the thing a reviewer would want to check when a result looks load-dependent.
        """
        cells = self.cells()
        random.Random(seed).shuffle(cells)
        return [Cell(c.arm, c.seed, i) for i, c in enumerate(cells)]

    def arm(self, arm_id: str) -> Arm:
        for a in self.arms:
            if a.id == arm_id:
                return a
        raise KeyError(f"no arm {arm_id!r} in this matrix")

    # ------------------------------------------------------------------ validation

    def undeclared_differences(self) -> list[tuple[str, str, ArmDiff]]:
        """Every pairwise difference between non-ablation arms that was not declared."""
        out: list[tuple[str, str, ArmDiff]] = []
        live = [a for a in self.arms if a.id not in self.ablations]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                for d in diff_arms(a, b, held_constant=self.held_constant):
                    if d.undeclared:
                        out.append((a.id, b.id, d))
        return out

    def ladder_steps(self) -> list[tuple[str, str, tuple[str, ...]]]:
        """For each consecutive pair of non-ablation arms, which treatment dimensions changed."""
        live = [a for a in self.arms if a.id not in self.ablations]
        steps = []
        for a, b in zip(live, live[1:]):
            changed = tuple(sorted(
                k for k in set(a.treatment) | set(b.treatment)
                if a.treatment.get(k) != b.treatment.get(k)))
            steps.append((a.id, b.id, changed))
        return steps

    def validate(self, require_ladder: bool = True) -> list[MatrixFinding]:
        """Findings that must be empty before a measured run.

        ``require_ladder`` asserts each consecutive pair differs in exactly ONE treatment dimension.
        That is what makes a difference attributable: if two things change at once, the delta between
        those arms measures their sum and nothing isolates either.
        """
        out: list[MatrixFinding] = []

        for a_id, b_id, d in self.undeclared_differences():
            out.append(MatrixFinding(
                "ARM001",
                f"{a_id} vs {b_id}: {d.field} differs ({d.a_value!r} vs {d.b_value!r}) — {d.reason}",
                (a_id, b_id)))

        if require_ladder:
            for a_id, b_id, changed in self.ladder_steps():
                if len(changed) != 1:
                    out.append(MatrixFinding(
                        "ARM002",
                        f"{a_id} -> {b_id} changes {len(changed)} treatment dimension(s) "
                        f"{changed or '()'} — a delta across two changes isolates neither",
                        (a_id, b_id)))

        if len(self.seeds) != len(set(self.seeds)):
            out.append(MatrixFinding("ARM003", f"duplicate seeds: {self.seeds}"))
        if not self.seeds:
            out.append(MatrixFinding("ARM003", "no seeds — nothing to pair"))
        if len({a.id for a in self.arms}) != len(self.arms):
            out.append(MatrixFinding("ARM003", "duplicate arm ids"))

        return out

    @property
    def valid(self) -> bool:
        return not self.validate()


def format_matrix(m: ArmMatrix) -> str:
    L = [
        "Arm matrix",
        "=" * 72,
        f"arms: {len(m.arms)}   seeds: {m.seeds}   cells: {len(m.cells())}"
        + (f"   ablations: {list(m.ablations)}" if m.ablations else ""),
    ]
    for a_id, b_id, changed in m.ladder_steps():
        L.append(f"  {a_id:<20} -> {b_id:<20} changes {changed or '()'}")
    findings = m.validate()
    L.append("-" * 72)
    L.append(f"validation: {'ok' if not findings else str(len(findings)) + ' finding(s)'}")
    for f in findings:
        L.append(f"  {f.code}  {f.detail}")
    return "\n".join(L)
