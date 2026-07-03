"""Control tests for full-fidelity OTel import (per-turn tokens + real-duration activity).

Ground truth: the sum of ``claude_code.api_request`` token fields == billed totals, and activity
bands come from REAL ``duration_ms`` spans (no interpolation). Validates tokens, cost, distinct
in/out shapes, and a well-formed activity decomposition.
"""
from __future__ import annotations

import json

from aet.trajectory.importers.otel import (
    build_from_otel_events, parse_otel_logs, import_otel, activity_breakdown,
)
from aet.viz.trajectory_plot import activity_share, ACTS


def _req(seq, t_ns, inp, out, cr, cc, cost, dur_ms, model="claude-opus-4-8"):
    return {"seq": seq, "t_ns": t_ns, "name": "claude_code.api_request",
            "attrs": {"input_tokens": inp, "output_tokens": out, "cache_read_tokens": cr,
                      "cache_creation_tokens": cc, "cost_usd": cost, "duration_ms": dur_ms,
                      "model": model}}


def _tool(seq, t_ns, name, dur_ms):
    return {"seq": seq, "t_ns": t_ns, "name": "claude_code.tool_result",
            "attrs": {"tool_name": name, "duration_ms": dur_ms, "success": "true"}}


# A controlled run: 3 API turns + 2 tools. timeUnixNano is the COMPLETION time; duration_ms is how
# long the work took → real span = [completion − duration, completion]. Chosen to tile perfectly
# (no overlap, no gap): think [0,2]+[2.5,7.5]+[19.5,21]=8.5s, read [2,2.5]=0.5s, tool [7.5,19.5]=12s.
def _events():
    B = 1_000_000_000_000_000_000  # base ns
    S = 1_000_000_000              # 1s in ns
    return [
        _req(2,  B + 2 * S,           inp=500, out=100, cr=0,     cc=0,   cost=0.10, dur_ms=2000),
        _tool(3, B + 2 * S + 500_000_000, "Read", 500),
        _req(4,  B + 7 * S + 500_000_000, inp=5, out=800, cr=10000, cc=200, cost=0.50, dur_ms=5000),
        _tool(5, B + 19 * S + 500_000_000, "Bash", 12000),  # 12s Bash → tool-wait
        _req(6,  B + 21 * S,          inp=5,   out=300, cr=20000, cc=0,   cost=0.30, dur_ms=1500),
    ]


AUTH = {"input": 710, "output": 1200, "cache": 30000, "cost": 0.90}  # sums of the api_request fields


class TestOtelTokens:
    def test_totals_are_exact_sums(self):
        pts, bands, totals, dur, ms = build_from_otel_events(_events(), n_passed=8, n_total=8)
        assert totals == AUTH

    def test_curves_cumulative_and_end_at_totals(self):
        pts, *_ = build_from_otel_events(_events())
        assert round(pts[-1].cum_input_tokens) == 710
        assert round(pts[-1].cum_output_tokens) == 1200
        assert round(pts[-1].cum_cache_tokens) == 30000
        assert abs(pts[-1].cum_cost_usd - 0.90) < 1e-6
        for a, b in zip(pts, pts[1:]):
            assert b.cum_input_tokens >= a.cum_input_tokens
            assert b.cum_output_tokens >= a.cum_output_tokens
            assert b.t_s >= a.t_s

    def test_in_out_distinct(self):
        # turn 1 is input-heavy (500/100), turn 2 output-heavy (5/800) → ratio must vary
        pts, *_ = build_from_otel_events(_events())
        ratios = [p.cum_output_tokens / p.cum_input_tokens for p in pts if p.cum_input_tokens > 0]
        assert max(ratios) - min(ratios) > 0.1

    def test_times_are_real_not_interpolated(self):
        # points sit at the real api_request COMPLETION wall-times (2s, 7.5s, 21s from the first start)
        pts, *_ = build_from_otel_events(_events())
        ts = [round(p.t_s, 1) for p in pts]
        assert ts[0] == 2.0 and 7.4 < ts[1] < 7.6 and 20.9 < ts[2] < 21.1


class TestOtelActivity:
    def test_breakdown_is_exact_sum_of_otel_durations(self):
        # GROUND TRUTH: activity seconds per category == Σ OTel duration_ms for that category.
        # think = 2+5+1.5 = 8.5s; tool = 12s Bash; read = 0.5s. No heuristic, no layout.
        bd = activity_breakdown(_events())
        assert abs(bd["think"] - 8.5) < 1e-6
        assert abs(bd["tool"] - 12.0) < 1e-6
        assert abs(bd["read"] - 0.5) < 1e-6

    def test_bands_partition_no_overlap(self):
        # the fix for the real bug: bands are a contiguous NON-OVERLAPPING partition
        _, bands, _, dur, _ = build_from_otel_events(_events())
        segs = sorted((b.t0_s, b.t1_s) for b in bands)
        for (a0, a1), (b0, b1) in zip(segs, segs[1:]):
            assert b0 >= a1 - 1e-9, "bands must not overlap"

    def test_bands_cover_wall_gaplessly(self):
        # NO empty space: the bands must tile [0, duration] with zero gap and zero overlap
        _, bands, _, dur, _ = build_from_otel_events(_events())
        segs = sorted((b.t0_s, b.t1_s) for b in bands)
        assert abs(segs[0][0]) < 1e-9, "first band starts at 0"
        assert abs(segs[-1][1] - dur) < 1e-9, "last band ends at duration"
        for (a0, a1), (b0, b1) in zip(segs, segs[1:]):
            assert abs(b0 - a1) < 1e-9, f"gap/overlap between bands at {a1}..{b0}"
        covered = sum(b1 - b0 for b0, b1 in segs)
        assert abs(covered - dur) < 1e-9, "bands cover 100% of wall"

    def test_bands_reproduce_breakdown(self):
        # each category's total BAND time equals the ground-truth breakdown (durations preserved)
        _, bands, _, _, _ = build_from_otel_events(_events())
        bd = activity_breakdown(_events())
        from collections import defaultdict
        got = defaultdict(float)
        for b in bands:
            got[b.category] += b.t1_s - b.t0_s
        for cat, secs in bd.items():
            assert abs(got[cat] - secs) < 1e-6, f"{cat}: bands {got[cat]} vs breakdown {secs}"

    def test_bash_classified_by_command_when_available(self):
        # precise classification: a verilator ./run.sh Bash is tool-wait even if short; a plain
        # ls Bash is ordinary shell even if (spuriously) long
        ev = [_req(2, 1_000_000_000_000_000_000, 5, 5, 0, 0, 0.01, 500),
              {"seq": 3, "t_ns": 1_000_000_000_500_000_000, "name": "claude_code.tool_result",
               "attrs": {"tool_name": "Bash", "duration_ms": 200, "tool_use_id": "t1"}},
              {"seq": 4, "t_ns": 1_000_000_001_000_000_000, "name": "claude_code.tool_result",
               "attrs": {"tool_name": "Bash", "duration_ms": 9000, "tool_use_id": "t2"}}]
        cmds = {"t1": "./run.sh 2>&1 | tail", "t2": "ls -la /tmp"}
        bd = activity_breakdown(ev, tool_cmds=cmds)
        assert abs(bd.get("tool", 0) - 0.2) < 1e-6   # the short run.sh → tool-wait
        assert abs(bd.get("bash", 0) - 9.0) < 1e-6   # the long ls → ordinary shell

    def test_activity_share_wellformed(self):
        traj = import_otel.__wrapped__ if hasattr(import_otel, "__wrapped__") else None
        pts, bands, totals, dur, ms = build_from_otel_events(_events())
        from aet.trajectory.model import RunTrajectory
        tr = RunTrajectory(run_id="t", source="test")
        tr.points, tr.bands, tr.milestones = pts, bands, ms
        from aet.trajectory.model import RoundBoundary
        tr.rounds = [RoundBoundary(index=0, t_start_s=0.0, t_end_s=dur)]
        tr.duration_s = dur
        g, sh = activity_share(tr)
        import numpy as np
        tot = sum(sh[a] for a in ACTS)
        assert np.allclose(tot[tot > 0], 1.0, atol=1e-6)   # shares sum to 1 where covered
        # tool-wait should hold a meaningful fraction (the 12s Bash dominates a ~21s run)
        assert sh["tool"].max() > 0.3


class TestOtelParse:
    def test_parse_otlp_envelope(self, tmp_path):
        # a realistic OTLP/JSON logs envelope (as the sink writes it) parses into ordered events
        rec = {"timeUnixNano": "1000000000000000000",
               "body": {"stringValue": "claude_code.api_request"},
               "attributes": [
                   {"key": "event.sequence", "value": {"intValue": 2}},
                   {"key": "input_tokens", "value": {"intValue": 500}},
                   {"key": "output_tokens", "value": {"intValue": 100}},
                   {"key": "cost_usd", "value": {"doubleValue": 0.1}},
                   {"key": "duration_ms", "value": {"intValue": 2000}}]}
        env = {"kind": "logs", "payload": {"resourceLogs": [{"scopeLogs": [{"logRecords": [rec]}]}]}}
        f = tmp_path / "otel.jsonl"
        f.write_text(json.dumps(env) + "\n")
        ev = parse_otel_logs(f)
        assert len(ev) == 1 and ev[0]["name"] == "claude_code.api_request"
        assert ev[0]["attrs"]["input_tokens"] == 500 and ev[0]["seq"] == 2

    def test_empty_returns_none(self):
        assert build_from_otel_events([]) is None


class TestCostReconciliation:
    """Cost accumulation is verified THREE independent ways that must agree: the sum of per-turn
    ``api_request.cost_usd`` (the cumulative curve), claude's own ``claude_code.cost.usage`` metric
    counter, and (on a real run) the transcript ``result.total_cost_usd`` billed total."""

    def _sum_cost_metric(self, payload) -> float:
        tot = 0.0
        for rm in payload.get("resourceMetrics", []):
            for sm in rm.get("scopeMetrics", []):
                for m in sm.get("metrics", []):
                    if m.get("name") == "claude_code.cost.usage":
                        for dp in m.get("sum", {}).get("dataPoints", []):
                            tot += float(dp.get("asDouble") or dp.get("asInt") or 0)
        return tot

    @staticmethod
    def _apireq_record(t_ns, seq, cost):
        attrs = [{"key": "event.sequence", "value": {"intValue": seq}},
                 {"key": "cost_usd", "value": {"doubleValue": cost}},
                 {"key": "input_tokens", "value": {"intValue": 5}},
                 {"key": "output_tokens", "value": {"intValue": 5}},
                 {"key": "duration_ms", "value": {"intValue": 100}}]
        return {"timeUnixNano": str(t_ns), "body": {"stringValue": "claude_code.api_request"},
                "attributes": attrs}

    @staticmethod
    def _logs_env(records):
        return {"kind": "logs",
                "payload": {"resourceLogs": [{"scopeLogs": [{"logRecords": records}]}]}}

    @staticmethod
    def _metrics_env(costs):
        dps = [{"asDouble": c} for c in costs]
        metric = {"name": "claude_code.cost.usage", "sum": {"dataPoints": dps}}
        return {"kind": "metrics",
                "payload": {"resourceMetrics": [{"scopeMetrics": [{"metrics": [metric]}]}]}}

    def test_per_turn_cost_sum_equals_cost_metric(self, tmp_path):
        # a capture with BOTH api_request logs and the cost.usage metric → the per-turn sum (what the
        # cumulative curve integrates) must equal claude's independent cost counter.
        costs = [0.10, 0.40]
        logs = self._logs_env([self._apireq_record(1_000_000_000_000_000_000, 2, costs[0]),
                               self._apireq_record(1_000_000_001_000_000_000, 3, costs[1])])
        metrics = self._metrics_env(costs)
        f = tmp_path / "otel.jsonl"
        f.write_text(json.dumps(logs) + "\n" + json.dumps(metrics) + "\n")
        _, _, totals, _, _ = build_from_otel_events(parse_otel_logs(f))
        metric_total = self._sum_cost_metric(metrics["payload"])
        assert abs(totals["cost"] - 0.50) < 1e-9          # cumulative curve total
        assert abs(metric_total - 0.50) < 1e-9            # independent cost counter
        assert abs(totals["cost"] - metric_total) < 1e-9  # they RECONCILE
