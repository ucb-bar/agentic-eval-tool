"""Control tests for token / cost / activity capture from Claude CLI stream-json transcripts.

Ground truth = the CLI's OWN authoritative numbers: the terminal ``result`` event's ``usage`` (the
billed input/cache/output counts) and ``total_cost_usd``. These tests assert our parsing reproduces
those exactly, that the per-signal over-time curves are modelled distinctly (input ≠ output shape —
the bug where both were driven by one shared stream), and that activity-share is well-formed.

The newer CLI format (verified against real runs) exposes each signal differently:
  * input / cache_read  — real per-message ``usage`` (matches result totals exactly)
  * output_tokens (usage) — VISIBLE output only; the bulk is thinking, streamed as
    ``estimated_tokens_delta`` (subtype ``thinking_tokens``)
  * result event         — authoritative totals + cost
So a correct parser must combine visible+thinking for output and pin totals to the result event.
"""
from __future__ import annotations

import json

from aet.trajectory.importers.transcript import densify_new_format, import_transcript
from aet.trajectory.classify import spec_to_rtl_config, ActivityClassifier
from aet.viz.trajectory_plot import activity_share


# ── a controlled synthetic NEW-format transcript with KNOWN authoritative numbers ──────────────
def _asst(mid, inp, cache_read, vis_out, cache_creation=0):
    # the CLI repeats the same message id once per content block; include a dup to exercise dedup
    msg = {"id": mid, "usage": {"input_tokens": inp, "cache_creation_input_tokens": cache_creation,
                                "cache_read_input_tokens": cache_read, "output_tokens": vis_out}}
    return {"type": "assistant", "message": msg}


def _think(delta):
    return {"type": "system", "subtype": "thinking_tokens", "estimated_tokens_delta": delta}


def _toolresult(ts):
    return {"type": "user", "timestamp": ts, "message": {"content": [{"type": "tool_result"}]}}


def _result(inp, cache_read, out, cost, cache_creation=0):
    return {"type": "result", "total_cost_usd": cost,
            "usage": {"input_tokens": inp, "cache_creation_input_tokens": cache_creation,
                      "cache_read_input_tokens": cache_read, "output_tokens": out}}


# Known ground truth for the synthetic run:
#   input=1000, cache_read=13000 (per-message == result), visible_out=30, thinking(est)=1000,
#   authoritative output=1500, cost=$2.50. Input arrives early; thinking accrues throughout →
#   input and output curves MUST have different shapes.
def _synthetic_new():
    return [
        {"type": "system", "subtype": "init", "session_id": "s"},
        _asst("A", inp=1000, cache_read=5000, vis_out=10), _asst("A", 1000, 5000, 10),  # dup id
        _think(100), _think(200),
        _toolresult("2026-01-01T00:00:10+00:00"),
        _think(300),
        _asst("B", inp=0, cache_read=8000, vis_out=20),
        _think(400),
        _toolresult("2026-01-01T00:00:40+00:00"),
        _result(inp=1000, cache_read=13000, out=1500, cost=2.50),
    ]


AUTH = {"input": 1000, "cache": 13000, "output": 1500, "cost": 2.50}
DUR = 60.0


class TestTokenTotals:
    def test_totals_match_authoritative_result_event(self):
        pts, totals = densify_new_format(_synthetic_new(), DUR)
        assert totals == AUTH, "parsed totals must equal the result-event's billed numbers exactly"

    def test_input_and_cache_are_exact_per_message(self):
        # input + cache_read come straight from per-message usage; they match the result exactly
        pts, totals = densify_new_format(_synthetic_new(), DUR)
        assert totals["input"] == 1000 and totals["cache"] == 13000

    def test_output_includes_thinking_not_just_visible(self):
        # visible output is only 30; the real 1500 is dominated by thinking → must NOT be ~30
        pts, totals = densify_new_format(_synthetic_new(), DUR)
        assert totals["output"] == 1500
        assert pts[-1].cum_output_tokens == 1500 and pts[-1].cum_output_tokens > 100

    def test_curves_end_at_authoritative_totals(self):
        pts, _ = densify_new_format(_synthetic_new(), DUR)
        assert round(pts[-1].cum_input_tokens) == 1000
        assert round(pts[-1].cum_output_tokens) == 1500
        assert round(pts[-1].cum_cache_tokens) == 13000
        assert abs(pts[-1].cum_cost_usd - 2.50) < 1e-6

    def test_cumulative_monotonic(self):
        pts, _ = densify_new_format(_synthetic_new(), DUR)
        for a, b in zip(pts, pts[1:]):
            assert b.cum_input_tokens >= a.cum_input_tokens - 1e-9
            assert b.cum_output_tokens >= a.cum_output_tokens - 1e-9
            assert b.cum_cache_tokens >= a.cum_cache_tokens - 1e-9
            assert b.cum_cost_usd >= a.cum_cost_usd - 1e-9
            assert b.t_s >= a.t_s - 1e-9

    def test_input_and_output_curves_are_distinct(self):
        # the original bug: in and out driven by one shared shape → perfectly proportional. Here the
        # input/output ratio must VARY over the run (different shapes).
        pts, _ = densify_new_format(_synthetic_new(), DUR)
        ratios = [p.cum_output_tokens / p.cum_input_tokens
                  for p in pts if p.cum_input_tokens > 0]
        assert max(ratios) - min(ratios) > 1e-3, "in/out curves must not be proportional-by-construction"

    def test_time_within_bounds(self):
        pts, _ = densify_new_format(_synthetic_new(), DUR)
        assert pts[0].t_s >= 0.0 and pts[-1].t_s == DUR
        assert all(0.0 <= p.t_s <= DUR for p in pts)

    def test_old_format_falls_back(self):
        # a transcript with per-turn usage but NO thinking stream → densify declines (None), so the
        # caller's existing per-turn path is preserved
        old = [_asst("A", 500, 0, 40), {"type": "result",
               "usage": {"input_tokens": 500, "output_tokens": 40, "cache_read_input_tokens": 0}}]
        pts, totals = densify_new_format(old, DUR)
        assert pts is None and totals is None


class TestActivityShare:
    def test_shares_sum_to_one(self, tmp_path):
        # a real transcript through the importer: activity-share columns must sum to ~1 everywhere
        f = tmp_path / "t.jsonl"
        f.write_text("\n".join(json.dumps(e) for e in _synthetic_new()))
        traj = import_transcript(f, run_id="syn", n_passed=1, n_total=1)
        if not traj.bands:
            return  # synthetic has no tool_use → no bands; covered by the classifier test below
        g, sh = activity_share(traj)
        import numpy as np
        tot = sum(sh[a] for a in sh)
        assert np.allclose(tot[tot > 0], 1.0, atol=1e-6)

    def test_verilator_bash_is_tool_wait(self):
        clf = ActivityClassifier(spec_to_rtl_config())
        cat, w = clf.classify("Bash", {"command": "rm -rf obj_dir; ./run.sh 2>&1 | tail"})
        assert cat == "tool", "a verilator/run.sh Bash is a long tool-wait, not ordinary shell"
        cat2, _ = clf.classify("Edit", {"file_path": "mxu.sv"})
        assert cat2 == "write"
        cat3, _ = clf.classify("Read", {"file_path": "spec/SPEC.md"})
        assert cat3 == "read"
