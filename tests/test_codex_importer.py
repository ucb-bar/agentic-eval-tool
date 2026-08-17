"""Codex importer → RunTrajectory: idempotency, reconciliation, nullable cost, ground truth."""
from pathlib import Path

from aet.trajectory.importers import get_importer
from aet.trajectory.importers.codex import import_codex, import_codex_run
from aet.trajectory.model import RunTrajectory
from aet.trajectory.reconcile import reconcile_codex

FIX = Path(__file__).parent / "fixtures" / "codex"


def test_registry_wires_codex():
    assert get_importer("codex") is import_codex


def test_import_real_simple_tokens_nonoverlapping():
    traj = import_codex(FIX / "real_simple_agent_message.jsonl")
    # buckets kept non-overlapping: input holds the uncached portion, cache split out
    assert traj.final_input_tokens == 17713 - 9984
    assert traj.final_cache_read_tokens == 9984
    assert traj.final_cache_creation_tokens == 0
    assert traj.final_output_tokens == 6
    assert traj.final_reasoning_tokens == 0
    # cum_total never double-counts a subset
    p = traj.points[-1]
    assert p.cum_total_tokens == p.cum_input_tokens + p.cum_output_tokens + p.cum_cache_tokens


def test_import_synthetic_rounds_and_split():
    traj = import_codex(FIX / "synthetic_full.jsonl")
    assert traj.num_rounds == 2
    assert traj.final_reasoning_tokens == 120
    assert traj.final_cache_creation_tokens == 1000
    # per-round split is populated (deliverable 2)
    r0 = traj.rounds[0]
    assert r0.cache_read_tokens == 4000
    assert r0.cache_creation_tokens == 1000
    assert r0.reasoning_tokens == 120


def test_activity_bands_from_structured_items():
    traj = import_codex(FIX / "synthetic_full.jsonl")
    cats = {b.category for b in traj.bands}
    names = {b.tool_name for b in traj.bands}
    assert "write" in cats                       # file_change → write
    assert "command_execution" in names          # command → its own span, not string-parsed
    # a file_change registered the first_file checkpoint
    assert traj.time_to("first_file") is not None


def test_idempotent_reimport_identical():
    a = import_codex(FIX / "synthetic_full.jsonl").to_dict()
    b = import_codex(FIX / "synthetic_full.jsonl").to_dict()
    assert a == b


def test_roundtrip_json_stable(tmp_path):
    traj = import_codex(FIX / "synthetic_full.jsonl")
    p = traj.to_json(tmp_path / "trajectory.json")
    reloaded = RunTrajectory.from_json(p)
    assert reloaded.to_dict() == traj.to_dict()
    assert reloaded.schema_version == "1.2"


# ---------------------------------------------------------------- cost: nullable + provenance
def test_cost_is_priced_with_provenance():
    traj = import_codex(FIX / "synthetic_full.jsonl", model="gpt-5-codex")
    assert traj.final_cost_usd is not None and traj.final_cost_usd > 0
    assert traj.cost["kind"] == "metered"        # openai provider → metered
    assert traj.cost["source"] == "price_calculated"
    assert traj.cost["price_table_id"] == "openai-2026-08-17"
    assert traj.cost["price_table_sha256"]
    assert traj.cost["value_usd"] == traj.final_cost_usd


def test_unpriced_model_is_none_not_zero():
    traj = import_codex(FIX / "synthetic_full.jsonl", model="totally-unknown-model")
    assert traj.final_cost_usd is None           # unpriced ≠ $0
    assert traj.cost["kind"] == "unpriced"
    assert traj.cost["value_usd"] is None


def test_subscription_mode_is_notional():
    traj = import_codex(FIX / "synthetic_full.jsonl", model="gpt-5-codex",
                        billing_mode="subscription")
    assert traj.cost["kind"] == "subscription_notional"


# ---------------------------------------------------------------- reconciliation (deliverable 10)
def test_reconcile_report_ok():
    traj, run = import_codex_run(FIX / "synthetic_full.jsonl")
    rep = reconcile_codex(run, traj, admin_usd=0.02)
    assert rep["raw_events"]["reconciled"] is True
    assert rep["token_ledger"]["all_match"] is True
    assert rep["token_ledger"]["subset_invariants_hold"] is True
    assert rep["missing_fields"]["any_missing"] is True   # turn 2 cache_write null
    assert rep["cost_vs_admin"]["delta_usd"] is not None
    assert rep["ok"] is True


def test_reconcile_cost_vs_admin_unavailable_is_none():
    traj, run = import_codex_run(FIX / "synthetic_full.jsonl", model="unknown")
    rep = reconcile_codex(run, traj, admin_usd=None)
    assert rep["cost_vs_admin"]["reconciled"] is False
    assert rep["cost_vs_admin"]["delta_usd"] is None


# ---------------------------------------------------------------- resume dir import
def test_import_directory_concatenates_resume(tmp_path):
    (tmp_path / "events.0.jsonl").write_text((FIX / "synthetic_full.jsonl").read_text())
    (tmp_path / "events.1.jsonl").write_text((FIX / "synthetic_resume.jsonl").read_text())
    traj = import_codex(tmp_path)
    assert traj.num_rounds == 3                   # 2 + 1 resumed turn, one continued thread
