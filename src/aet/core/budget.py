"""Budget enforcement — before a run starts, and while it is running.

``aet spend --budget-usd`` is post-hoc: it tells you afterwards that a grid cost more than intended.
That is useful for accounting and useless as a control, because an arm that overran its cap is no
longer comparable to one that did not, and by the time the number exists the money is gone.

Two enforcement points, both of which have to exist for a cap to mean anything:

* **Pre-flight.** Estimate the whole grid before spending anything and refuse if it does not fit.
  A refusal here costs nothing; the same refusal after cell 9 of 12 has wasted nine runs and left a
  grid that cannot be analysed.
* **Mid-run.** Stop a single run that has exceeded its own cap. Without it, one pathological run can
  consume the budget for every run that was supposed to follow it.

Honesty rule inherited from :class:`aet.trajectory.pricing.PriceTable`: an unpriced model yields
``None``, never ``0.0``. A guard that treats "cost unknown" as "cost zero" will happily authorise an
unbounded grid, which is the exact opposite of what it is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """A cap was reached. Carries the numbers so the caller can report rather than re-derive."""

    def __init__(self, message: str, *, kind: str, limit: float, actual: float):
        super().__init__(message)
        self.kind = kind          # tokens | cost_usd | wall_seconds
        self.limit = limit
        self.actual = actual


class BudgetUnknown(RuntimeError):
    """A pre-flight check could not price the grid.

    Distinct from :class:`BudgetExceeded` on purpose. "This costs more than the cap" and "nobody can
    say what this costs" call for different decisions, and collapsing them into one refusal teaches
    the operator to bypass both with the same flag.
    """


@dataclass
class Budget:
    """Per-run caps. ``None`` means unbounded for that dimension."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_wall_seconds: float | None = None
    max_active_seconds: float | None = None
    enforcement: str = "preflight_and_midrun"   # preflight_and_midrun | preflight_only | posthoc_only

    def is_bounded(self) -> bool:
        return any(v is not None for v in
                   (self.max_tokens, self.max_cost_usd, self.max_wall_seconds,
                    self.max_active_seconds))


@dataclass
class Usage:
    """What one run has consumed so far."""

    tokens: int = 0
    cost_usd: float | None = 0.0
    wall_seconds: float = 0.0
    active_seconds: float = 0.0


@dataclass
class BudgetGuard:
    """Enforces one :class:`Budget` for one run.

    ``check`` is called as usage accrues and raises the first time a cap is crossed. It does not
    kill anything itself — the caller owns the process — but it converts "over budget" from a
    number somebody might notice into an exception somebody must handle.
    """

    budget: Budget
    run_id: str = ""
    breaches: list[str] = field(default_factory=list)

    def check(self, usage: Usage, *, raise_on_breach: bool = True) -> str | None:
        """Return the breached dimension, or ``None``. Raises unless told not to."""
        b = self.budget
        for kind, limit, actual in (
            ("tokens", b.max_tokens, usage.tokens),
            ("cost_usd", b.max_cost_usd, usage.cost_usd),
            ("wall_seconds", b.max_wall_seconds, usage.wall_seconds),
            ("active_seconds", b.max_active_seconds, usage.active_seconds),
        ):
            if limit is None or actual is None:
                # `actual is None` is an UNPRICED model, not a free one. Skipping is right — the
                # pre-flight check is where an unpriceable grid is refused, and raising here would
                # kill a run for a pricing-table gap rather than for spending.
                continue
            if actual > limit:
                msg = (f"run {self.run_id or '<unnamed>'} exceeded {kind}: "
                       f"{actual:g} > {limit:g}")
                self.breaches.append(msg)
                if raise_on_breach:
                    raise BudgetExceeded(msg, kind=kind, limit=float(limit), actual=float(actual))
                return kind
        return None

    @property
    def clean(self) -> bool:
        return not self.breaches


def preflight(n_runs: int, per_run: Budget, *, total_cost_cap_usd: float | None,
              already_spent_usd: float = 0.0, estimated_cost_per_run: float | None = None
              ) -> dict:
    """Refuse a grid that cannot fit its total cap, BEFORE anything is spent.

    ``estimated_cost_per_run`` falls back to ``per_run.max_cost_usd`` — the worst case, which is the
    right default for a go/no-go: authorising on an optimistic estimate and discovering the truth at
    cell 9 is how a budget gets blown.

    Raises :class:`BudgetUnknown` when no estimate is available and a cap exists, and
    :class:`BudgetExceeded` when the estimate does not fit. Returns the arithmetic otherwise, so a
    caller can print what it authorised.
    """
    per = estimated_cost_per_run if estimated_cost_per_run is not None else per_run.max_cost_usd
    if total_cost_cap_usd is None:
        return {"authorised": True, "n_runs": n_runs, "per_run_usd": per,
                "estimated_total_usd": None if per is None else per * n_runs,
                "cap_usd": None, "already_spent_usd": already_spent_usd,
                "headroom_usd": None, "note": "no total cap configured"}
    if per is None:
        raise BudgetUnknown(
            f"cannot price {n_runs} run(s): no per-run cost estimate and no per-run cost cap. "
            f"An unpriced model reports None, never $0 — set Budget.max_cost_usd or pass "
            f"estimated_cost_per_run.")
    estimated = per * n_runs
    headroom = total_cost_cap_usd - already_spent_usd
    if estimated > headroom:
        raise BudgetExceeded(
            f"grid of {n_runs} run(s) at ${per:g} each = ${estimated:g}, but only "
            f"${headroom:g} remains of a ${total_cost_cap_usd:g} cap "
            f"(${already_spent_usd:g} already spent)",
            kind="cost_usd", limit=float(headroom), actual=float(estimated))
    return {"authorised": True, "n_runs": n_runs, "per_run_usd": per,
            "estimated_total_usd": estimated, "cap_usd": total_cost_cap_usd,
            "already_spent_usd": already_spent_usd, "headroom_usd": headroom - estimated,
            "note": ""}
