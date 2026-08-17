"""Reconciliation reports — prove the imported numbers add up, and surface what could not be priced.

Four checks the Codex×AET contract requires (handoff "AET changes" #10):

1. **raw events vs imported events** — every raw JSONL line is accounted for (dispatched, kept as
   an unknown event, or kept as an ``[UNPARSED]`` line). Nothing silently dropped.
2. **token-ledger vs trajectory** — the per-turn token ledger sums to the trajectory's final
   buckets (input/output/cache-read/cache-write/reasoning), with subset invariants intact.
3. **calculated per-run cost vs campaign admin cost** — a signed delta between the price-calculated
   figure and an externally-reconciled (invoice/usage-API) figure, so drift is visible and stored.
4. **missing / unpriced fields** — which turns reported a null bucket, and whether the run priced.

Everything is returned as plain dicts (JSON-serializable) so a report lands in the run bundle.
"""
from __future__ import annotations

from aet.trajectory.codex import CodexRun
from aet.trajectory.model import RunTrajectory

_BUCKETS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
            "output_tokens", "reasoning_output_tokens")


def reconcile_raw_events(run: CodexRun) -> dict:
    """Every raw line is either a dispatched JSON event, a kept-unknown, or a kept-unparsed line."""
    accounted = run.normalized_event_count + len(run.unparsed_lines)
    # unknown_events are a SUBSET of normalized_event_count (they were parsed then kept), so they
    # are not added again — reconciliation is (dispatched JSON) + (non-JSON unparsed) == raw lines.
    return {
        "raw_event_count": run.raw_event_count,
        "normalized_event_count": run.normalized_event_count,
        "unknown_event_count": len(run.unknown_events),
        "unparsed_line_count": len(run.unparsed_lines),
        "accounted": accounted,
        "reconciled": accounted == run.raw_event_count,
    }


def reconcile_token_ledger(run: CodexRun, traj: RunTrajectory) -> dict:
    """The per-turn token ledger vs the trajectory finals.

    The trajectory keeps buckets non-overlapping (``final_input`` is the *uncached* input, cache
    read/write split out, reasoning beside output), so we compare the ledger's derived quantities
    to the trajectory finals, and separately assert the subset invariants on the raw ledger.
    """
    totals = run.totals()
    unc = totals.get("uncached_input_tokens")
    cr = totals.get("cached_input_tokens") or 0
    cw = totals.get("cache_write_input_tokens") or 0
    out = totals.get("output_tokens") or 0
    reason = totals.get("reasoning_output_tokens") or 0

    checks = {
        "uncached_input": {"ledger": unc, "trajectory": traj.final_input_tokens,
                           "match": (unc or 0) == traj.final_input_tokens},
        "cache_read": {"ledger": cr, "trajectory": traj.final_cache_read_tokens,
                       "match": cr == traj.final_cache_read_tokens},
        "cache_write": {"ledger": cw, "trajectory": traj.final_cache_creation_tokens,
                        "match": cw == traj.final_cache_creation_tokens},
        "output": {"ledger": out, "trajectory": traj.final_output_tokens,
                   "match": out == traj.final_output_tokens},
        "reasoning": {"ledger": reason, "trajectory": traj.final_reasoning_tokens,
                      "match": reason == traj.final_reasoning_tokens},
    }
    # subset invariants (per turn): cached ⊆ input, cache_write ⊆ input, reasoning ⊆ output
    subset_ok = True
    for t in run.turns:
        if t.input_tokens is not None:
            if (t.cached_input_tokens or 0) + (t.cache_write_input_tokens or 0) > t.input_tokens:
                subset_ok = False
        if t.output_tokens is not None and (t.reasoning_output_tokens or 0) > t.output_tokens:
            subset_ok = False
    return {
        "checks": checks,
        "all_match": all(c["match"] for c in checks.values()),
        "subset_invariants_hold": subset_ok,
        "num_turns": len(run.turns),
    }


def reconcile_missing_fields(run: CodexRun, traj: RunTrajectory) -> dict:
    """Which turns reported a null bucket, and whether the run priced at all."""
    per_turn = []
    for i, t in enumerate(run.turns):
        missing = [b for b in _BUCKETS if getattr(t, b) is None]
        if missing:
            per_turn.append({"turn": i, "missing": missing})
    cost = traj.cost or {}
    return {
        "turns_with_missing_buckets": per_turn,
        "any_missing": bool(per_turn),
        "cost_unpriced": traj.final_cost_usd is None,
        "cost_kind": cost.get("kind"),
        "price_table_id": cost.get("price_table_id"),
    }


def reconcile_cost_vs_admin(calculated_usd: float | None, admin_usd: float | None) -> dict:
    """Signed delta between the price-calculated cost and an externally-reconciled admin cost.

    ``None`` on either side means that figure is unavailable — the delta is then ``None`` (never
    silently 0), and ``reconciled`` is False so an unpriced/unbilled run is flagged, not hidden.
    """
    delta = (None if calculated_usd is None or admin_usd is None
             else round(float(admin_usd) - float(calculated_usd), 6))
    return {
        "calculated_usd": calculated_usd,
        "admin_usd": admin_usd,
        "delta_usd": delta,
        "reconciled": delta is not None,
    }


def reconcile_codex(run: CodexRun, traj: RunTrajectory, *,
                    admin_usd: float | None = None) -> dict:
    """The full reconciliation report for one Codex run (all four checks)."""
    calc = traj.final_cost_usd
    report = {
        "raw_events": reconcile_raw_events(run),
        "token_ledger": reconcile_token_ledger(run, traj),
        "missing_fields": reconcile_missing_fields(run, traj),
        "cost_vs_admin": reconcile_cost_vs_admin(calc, admin_usd),
    }
    report["ok"] = (report["raw_events"]["reconciled"]
                    and report["token_ledger"]["all_match"]
                    and report["token_ledger"]["subset_invariants_hold"])
    return report


# ------------------------------------------------------------------ ledgers (for the run bundle)
def token_ledger_rows(run: CodexRun) -> list[dict]:
    """One JSON row per turn — the append-only token ledger (``metrics/token_ledger.jsonl``)."""
    rows = []
    for i, t in enumerate(run.turns):
        rows.append({"turn": i, "source_event": t.source_event, **t.to_dict()})
    return rows


def tool_ledger_rows(run: CodexRun) -> list[dict]:
    """One JSON row per tool span (``agent/tools.jsonl``)."""
    return [t.to_dict() for t in run.tools]
