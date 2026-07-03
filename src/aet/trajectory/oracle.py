"""Extract a test-pass *progression* from an agent's own oracle invocations in a transcript.

Many agentic tasks give the agent a runnable oracle it invokes repeatedly during a run (e.g.
abc-testing's ``./run.sh`` → a Verilator testbench). Each invocation's result is already in the
transcript as a tool result. Reading those in time order recovers the *climb* — the same
tests-passing-over-time signal a periodic grader would produce — with **no harness change** and
**retroactively** on existing runs.

The parser is generic: it looks for a marker command (``run.sh``/``verilator``/…) on a Bash tool
call, then reads the following tool result for a pass verdict. It understands the common Verilator
testbench summary forms:

  * ``*** PASSED *** <dut> replay: <N> cases``        → N / N  (full pass, N is the suite size)
  * ``*** FAILED *** … <M> (lane )?mismatches``        → 0 / N  (a failed attempt)
  * a bare ``RESULT: PASS`` / ``RESULT: FAIL``          → pass/fail (count from a hint or carried N)

A per-invocation ``k/N`` is emitted at the tool call's wall time; ``n_total`` is taken from the
first count seen (or a caller hint), so a run that fails then passes climbs ``0/N → N/N`` at the
exact iteration it started passing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

# summary lines emitted by the replay testbenches
_PASS_CASES = re.compile(r"\*\*\*\s*PASSED\s*\*\*\*.*?(\d+)\s*cases", re.I | re.S)
_FAIL_MM = re.compile(r"\*\*\*\s*FAILED\s*\*\*\*.*?(\d+)\s*(?:lane\s+)?mismatch", re.I | re.S)
_RESULT = re.compile(r"RESULT:\s*(PASS|FAIL)", re.I)
_PASSED_ANY = re.compile(r"\*\*\*\s*PASSED\s*\*\*\*", re.I)
_FAILED_ANY = re.compile(r"\*\*\*\s*FAILED\s*\*\*\*", re.I)

# The task's self-check ENTRYPOINT only — e.g. abc-testing's ``./run.sh`` (which runs the DUT's
# replay testbench). Deliberately NOT bare ``run_test.py``/``verilator``: the agent also runs those
# directly on sub-modules / ad-hoc unit tests, whose case counts are not the grading oracle.
DEFAULT_ORACLE_MARKERS = ("run.sh",)


@dataclass
class OracleReading:
    t_s: float            # wall seconds from the round/segment start
    n_passed: int
    n_total: int
    passed: bool


def parse_oracle_result(text: str, *, n_total_hint: int | None = None,
                        carried_total: int | None = None) -> tuple[int, int, bool, bool] | None:
    """Parse one tool-result blob → (n_passed, n_total, passed, explicit_count), or None.

    ``explicit_count`` is True when the suite size came from the text ("N cases") rather than a
    hint — the caller uses it to reject readings from a *different* testbench (a count that doesn't
    match the known suite size)."""
    if not text:
        return None
    m_pass = _PASS_CASES.search(text)
    if m_pass:
        n = int(m_pass.group(1))
        return n, n, True, True
    passed = bool(_PASSED_ANY.search(text)) or bool(_RESULT.search(text) and
                                                     _RESULT.search(text).group(1).upper() == "PASS")
    failed = bool(_FAILED_ANY.search(text)) or bool(_RESULT.search(text) and
                                                    _RESULT.search(text).group(1).upper() == "FAIL")
    if not (passed or failed):
        return None
    total = carried_total or n_total_hint or 1
    return (total if passed else 0), total, passed, False


def _matches(cmd: str, markers) -> bool:
    return any(mk in cmd for mk in markers)


def extract_oracle_progression(seg_events: "list[tuple[float, str]]", *,
                               markers=DEFAULT_ORACLE_MARKERS,
                               n_total_hint: int | None = None) -> list[OracleReading]:
    """Walk one segment's ``(abs_ts, json_line)`` events → oracle readings, times relative to the
    segment start. Only Bash tool calls whose command matches a marker are considered."""
    if not seg_events:
        return []
    base = seg_events[0][0]
    marker_ids: dict[str, float] = {}     # tool_use_id -> abs_ts of the oracle call
    results: list[OracleReading] = []
    # first pass: which tool_use ids are oracle calls + their time
    result_texts: dict[str, tuple[float, str]] = {}
    for ts, line in seg_events:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        t = ev.get("type")
        if t == "assistant":
            for b in (ev.get("message", {}) or {}).get("content", []) or []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Bash":
                    try:
                        cmd = json.dumps(b.get("input", {}) or {})
                    except Exception:
                        cmd = str(b.get("input", ""))
                    if _matches(cmd, markers):
                        marker_ids[b.get("id", "")] = ts
        elif t == "user":
            for b in (ev.get("message", {}) or {}).get("content", []) or []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = b.get("tool_use_id", "")
                    c = b.get("content")
                    txt = c if isinstance(c, str) else " ".join(
                        x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else ""
                    result_texts[tid] = (ts, txt)

    carried = n_total_hint
    for tid, call_ts in sorted(marker_ids.items(), key=lambda kv: kv[1]):
        res_ts, txt = result_texts.get(tid, (call_ts, ""))
        parsed = parse_oracle_result(txt, n_total_hint=n_total_hint, carried_total=carried)
        if parsed is None:
            continue
        n_passed, n_total, passed, explicit = parsed
        # reject a reading whose explicit suite size differs from the known one — that's the agent
        # running a DIFFERENT testbench (its own sub-module unit tests), not the grading oracle.
        if explicit and n_total_hint and n_total != n_total_hint:
            continue
        carried = max(carried or 0, n_total)     # remember the suite size for later failed runs
        results.append(OracleReading(t_s=res_ts - base, n_passed=n_passed,
                                     n_total=n_total, passed=passed))
    return results
