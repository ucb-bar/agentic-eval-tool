"""One experimental condition, and what it is allowed to differ in.

An :class:`Arm` bundles the treatment (what this condition is *for*) with the realized configuration
that delivers it (what the agent can read, which tools it has, what it is charged). Keeping both on
one object is the point: the fairness check compares realized configurations and asks whether every
difference it finds was declared as a treatment.

The failure this prevents is mundane and fatal. Someone adds a tool to arm 4 while debugging, forgets
to remove it, and the A3→A4 delta now measures that tool plus the treatment. Nothing errors. The
result is wrong in a direction that flatters the hypothesis, and no downstream check can see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

#: Fields that describe the TREATMENT — arms are expected to differ here.
_TREATMENT_FIELDS = ("treatment", "granted", "denied", "tools", "prompt_id")

#: Fields that must be identical across every arm in a matrix unless the matrix says otherwise.
#: Not a style preference: each one, if it varied, would confound the comparison on its own.
DEFAULT_HELD_CONSTANT = (
    "model",
    "model_version",
    "agent",
    "max_tokens",
    "max_cost_usd",
    "max_wall_seconds",
    "budget_enforcement",
    "public_suite",
    "hidden_eval",
    "candidate_contract",
    "sandbox_isolation",
    "network_isolated",
    "env_cleared",
    "git_masked",
)


@dataclass(frozen=True)
class Arm:
    """One condition in a controlled comparison."""

    id: str
    label: str = ""

    # --- the treatment: what this arm exists to test -------------------------------------------
    treatment: Mapping[str, str] = field(default_factory=dict)
    granted: tuple[str, ...] = ()
    denied: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    prompt_id: str = ""

    # --- held constant: varying any of these would confound the comparison ----------------------
    model: str = ""
    model_version: str = ""
    agent: str = ""
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_wall_seconds: float | None = None
    budget_enforcement: str = "preflight_and_midrun"
    public_suite: str = ""
    hidden_eval: str = ""
    candidate_contract: str = ""
    sandbox_isolation: str = "bwrap"
    network_isolated: bool = True
    env_cleared: bool = True
    git_masked: bool = True

    #: Free-form, never compared. For anything that genuinely does not affect the measurement.
    notes: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ views

    def fingerprint(self) -> dict:
        """Every comparable field. ``notes`` is excluded by construction."""
        out: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            if name in ("id", "label", "notes"):
                continue
            val = getattr(self, name)
            out[name] = dict(val) if isinstance(val, Mapping) else val
        return out

    def treatment_dims(self) -> tuple[str, ...]:
        return tuple(sorted(self.treatment))

    def sandbox_kwargs(self, workspace: Path) -> dict:
        """Arguments for :class:`aet.isolation.SandboxSpec`.

        Returned as a dict rather than a ``SandboxSpec`` so this module stays import-light and
        usable where bwrap is irrelevant (analysis, plotting, validation).
        """
        return {
            "workspace": Path(workspace),
            "allow": [Path(p) for p in self.granted],
            "deny": [Path(p) for p in self.denied],
            "unshare_net": self.network_isolated,
            "clearenv": self.env_cleared,
            "mask_git": self.git_masked,
        }

    def with_(self, **changes) -> "Arm":
        """A copy with fields replaced — for building a ladder from a base arm."""
        return replace(self, **changes)


@dataclass
class ArmDiff:
    """A field on which two arms differ."""

    field: str
    a_value: Any
    b_value: Any
    declared: bool          # True when this field is part of the treatment
    reason: str = ""

    @property
    def undeclared(self) -> bool:
        return not self.declared


def diff_arms(a: Arm, b: Arm, held_constant: tuple[str, ...] = DEFAULT_HELD_CONSTANT
              ) -> list[ArmDiff]:
    """Every field on which ``a`` and ``b`` differ, each marked declared or not.

    A difference is *declared* when the field is part of the treatment AND the two arms actually
    declare different treatments. Two arms with identical ``treatment`` dicts that nonetheless grant
    different files are not expressing a treatment — they are expressing a mistake, and this reports
    it as undeclared even though ``granted`` is nominally a treatment field.
    """
    fa, fb = a.fingerprint(), b.fingerprint()
    same_treatment = fa.get("treatment") == fb.get("treatment")
    out: list[ArmDiff] = []
    for name in sorted(set(fa) | set(fb)):
        va, vb = fa.get(name), fb.get(name)
        if va == vb:
            continue
        if name in held_constant:
            out.append(ArmDiff(name, va, vb, declared=False,
                               reason="held constant across arms; a difference here confounds the "
                                      "comparison"))
        elif name in _TREATMENT_FIELDS and not same_treatment:
            out.append(ArmDiff(name, va, vb, declared=True,
                               reason="differs, and the arms declare different treatments"))
        elif name in _TREATMENT_FIELDS:
            out.append(ArmDiff(name, va, vb, declared=False,
                               reason="a treatment field differs, but the declared treatments are "
                                      "identical — so this difference is undeclared"))
        else:
            out.append(ArmDiff(name, va, vb, declared=False,
                               reason="not a declared treatment dimension"))
    return out
