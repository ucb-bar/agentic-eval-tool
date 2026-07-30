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
