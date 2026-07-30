"""Cross-experiment spend rollup: aggregate cost/tokens over synthetic run dirs, order the
cumulative series by calendar time, keep cost-unavailable runs honest, and enforce a budget."""
import json
import subprocess
import sys

import pytest

from aet.trajectory.rollup import read_run_spend, rollup_runs


def _make_run(root, suite, run_id, *, model, cost, cin=0, cout=0, cread=0, ccreate=0,
              created_at="2026-07-01T00:00:00Z", with_cost=True):
    """A minimal aet run dir: run_record.json + logs/metrics.jsonl (the `aet runs` contract)."""
    rd = root / suite / run_id
    (rd / "logs").mkdir(parents=True)
    rd.joinpath("run_record.json").write_text(json.dumps({
        "run_id": run_id, "suite": suite, "model": model, "created_at": created_at,
    }))
    lines = [
        {"name": "gen_ai.usage.input_tokens", "value": cin},
        {"name": "gen_ai.usage.output_tokens", "value": cout},
        {"name": "gen_ai.usage.cache_read.input_tokens", "value": cread},
        {"name": "gen_ai.usage.cache_creation.input_tokens", "value": ccreate},
    ]
    if with_cost:
        lines.insert(0, {"name": "aet.agent.cost_usd", "value": cost})
    metrics = rd / "logs" / "metrics.jsonl"
    metrics.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return rd


def test_rollup_aggregates_cost_tokens_and_per_model(tmp_path):
    root = tmp_path / "runs"
    _make_run(root, "suiteA", "r1", model="claude-opus-4-8", cost=1.50, cin=1000, cout=200,
              cread=50, ccreate=10, created_at="2026-07-01T00:00:00Z")
    _make_run(root, "suiteA", "r2", model="claude-opus-4-8", cost=2.25, cin=2000, cout=400,
              created_at="2026-07-02T00:00:00Z")
    _make_run(root, "suiteB", "r3", model="amazon.nova-pro-v1", cost=0.30, cin=500, cout=100,
              created_at="2026-07-03T00:00:00Z")

    roll = rollup_runs([root])
    assert roll.n_runs == 3
    assert roll.total_cost_usd == pytest.approx(4.05)
    assert roll.tokens.input == 3500
    assert roll.tokens.output == 700
    assert roll.tokens.cache_total == 60
    assert roll.unpriced_runs == 0

    assert set(roll.per_model) == {"claude-opus-4-8", "amazon.nova-pro-v1"}
    assert roll.per_model["claude-opus-4-8"].cost_usd == pytest.approx(3.75)
    assert roll.per_model["claude-opus-4-8"].n_runs == 2

    # cumulative series is calendar-ordered and ends at the total
    cum = roll.cumulative
    assert [c["run_id"] for c in cum] == ["r1", "r2", "r3"]
    assert cum[-1]["cumulative_usd"] == pytest.approx(4.05)


def _make_multimodel_run(root, suite, run_id, *, identity_model, per_model,
                         created_at="2026-07-01T00:00:00Z"):
    """A run that recorded a within-run `per_model.<safe>.*` split (an orchestrator + sub-agents).

    ``per_model`` maps safe-model-name -> dict(cost, input, output, cache_read, cache_creation).
    The run's top-level ``aet.agent.cost_usd`` is the sum of the per-model costs (the authoritative
    whole-run total), and the top-level token counts are the per-model sums.
    """
    rd = root / suite / run_id
    (rd / "logs").mkdir(parents=True)
    rd.joinpath("run_record.json").write_text(json.dumps({
        "run_id": run_id, "suite": suite, "model": identity_model, "created_at": created_at,
    }))
    total_cost = sum(d["cost"] for d in per_model.values())
    tin = sum(d.get("input", 0) for d in per_model.values())
    tout = sum(d.get("output", 0) for d in per_model.values())
    tread = sum(d.get("cache_read", 0) for d in per_model.values())
    tcreate = sum(d.get("cache_creation", 0) for d in per_model.values())
    lines = [
        {"name": "aet.agent.cost_usd", "value": total_cost},
        {"name": "gen_ai.usage.input_tokens", "value": tin},
        {"name": "gen_ai.usage.output_tokens", "value": tout},
        {"name": "gen_ai.usage.cache_read.input_tokens", "value": tread},
        {"name": "gen_ai.usage.cache_creation.input_tokens", "value": tcreate},
    ]
    for safe, d in per_model.items():
        billed = (d.get("input", 0) + d.get("output", 0)
                  + d.get("cache_read", 0) + d.get("cache_creation", 0))
        lines += [
            {"name": f"per_model.{safe}.cost_usd", "value": d["cost"]},
            {"name": f"per_model.{safe}.input_tokens_raw", "value": d.get("input", 0)},
            {"name": f"per_model.{safe}.output_tokens", "value": d.get("output", 0)},
            {"name": f"per_model.{safe}.cache_read_tokens", "value": d.get("cache_read", 0)},
            {"name": f"per_model.{safe}.cache_creation_tokens", "value": d.get("cache_creation", 0)},
            {"name": f"per_model.{safe}.total_billed_tokens", "value": billed},
        ]
    (rd / "logs" / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return rd


def test_rollup_splits_within_run_per_model(tmp_path):
    # Two runs, EACH an Opus orchestrator + a Sonnet sub-agent (a truncated bedrock run shape).
    root = tmp_path / "runs"
    _make_multimodel_run(
        root, "capsule", "r1", identity_model="us.anthropic.claude-opus-4-6-v1",
        per_model={
            "claude_opus_4_6": {"cost": 9.55, "input": 1000, "output": 2000, "cache_read": 4000},
            "claude_sonnet_4_6": {"cost": 0.13, "input": 100, "output": 200, "cache_read": 300},
        }, created_at="2026-07-01T00:00:00Z")
    _make_multimodel_run(
        root, "capsule", "r2", identity_model="us.anthropic.claude-opus-4-6-v1",
        per_model={
            "claude_opus_4_6": {"cost": 4.00, "input": 500, "output": 1000, "cache_read": 2000},
            "claude_sonnet_4_6": {"cost": 0.07, "input": 50, "output": 100, "cache_read": 150},
        }, created_at="2026-07-02T00:00:00Z")

    roll = rollup_runs([root])
    assert roll.n_runs == 2
    # whole-run totals unchanged (authoritative per-run cost counted once)
    assert roll.total_cost_usd == pytest.approx(9.55 + 0.13 + 4.00 + 0.07)

    # the per-model split shows BOTH models distinctly, each spanning both runs
    assert set(roll.per_model) == {"claude_opus_4_6", "claude_sonnet_4_6"}
    opus = roll.per_model["claude_opus_4_6"]
    sonnet = roll.per_model["claude_sonnet_4_6"]
    assert opus.cost_usd == pytest.approx(13.55)
    assert sonnet.cost_usd == pytest.approx(0.20)
    assert opus.n_runs == 2 and sonnet.n_runs == 2
    assert opus.tokens.input == 1500 and sonnet.tokens.input == 150

    # activity shares are token fractions and sum to 1 across models
    assert abs(opus.activity_share + sonnet.activity_share - 1.0) < 1e-9
    assert opus.activity_share > sonnet.activity_share
    total_billed = roll.tokens.total
    assert abs(opus.activity_share - opus.tokens.total / total_billed) < 1e-9


def test_rollup_without_per_model_attributes_to_identity_model(tmp_path):
    # Back-compat: a run with no per_model.* metrics is still attributed to its single model.
    root = tmp_path / "runs"
    _make_run(root, "s", "r1", model="claude-opus-4-8", cost=3.0, cin=100, cout=50)
    roll = rollup_runs([root])
    assert set(roll.per_model) == {"claude-opus-4-8"}
    assert roll.per_model["claude-opus-4-8"].cost_usd == pytest.approx(3.0)


def test_unavailable_cost_is_not_counted_as_zero(tmp_path):
    root = tmp_path / "runs"
    _make_run(root, "s", "priced", model="claude-opus-4-8", cost=5.0)
    _make_run(root, "s", "noprice", model="mystery-model", cost=0.0, with_cost=False)

    roll = rollup_runs([root])
    assert roll.n_runs == 2
    assert roll.unpriced_runs == 1
    assert roll.total_cost_usd == pytest.approx(5.0)  # unpriced run NOT folded in as $0
    spend = read_run_spend(root / "s" / "noprice")
    assert spend.cost_usd is None and spend.cost_available is False


def test_budget_ceiling_over_and_under(tmp_path):
    root = tmp_path / "runs"
    _make_run(root, "s", "r1", model="claude-opus-4-8", cost=120.0,
              created_at="2026-07-01T00:00:00Z")
    _make_run(root, "s", "r2", model="claude-opus-4-8", cost=210.0,
              created_at="2026-07-02T00:00:00Z")

    over = rollup_runs([root], budget_usd=300.0)
    assert over.total_cost_usd == pytest.approx(330.0)
    assert over.over_budget is True
    assert over.headroom_usd == pytest.approx(-30.0)

    under = rollup_runs([root], budget_usd=400.0)
    assert under.over_budget is False
    assert under.headroom_usd == pytest.approx(70.0)  # remaining headroom


def test_cli_spend_json_and_budget_exit(tmp_path):
    root = tmp_path / "runs"
    _make_run(root, "s", "r1", model="claude-opus-4-8", cost=200.0)
    _make_run(root, "s", "r2", model="claude-opus-4-8", cost=150.0)

    cp = subprocess.run(
        [sys.executable, "-m", "aet.cli.main", "spend", str(root), "--json",
         "--budget-usd", "300"],
        capture_output=True, text=True)
    assert cp.returncode == 2, cp.stderr          # over budget → non-zero exit
    data = json.loads(cp.stdout)
    assert data["over_budget"] is True
    assert data["total_cost_usd"] == pytest.approx(350.0)
    assert data["headroom_usd"] == pytest.approx(-50.0)

    cp2 = subprocess.run(
        [sys.executable, "-m", "aet.cli.main", "spend", str(root), "--budget-usd", "500"],
        capture_output=True, text=True)
    assert cp2.returncode == 0, cp2.stderr
    assert "within budget" in cp2.stdout
