"""Full-fidelity trajectory import from Claude Code OTel telemetry (OTLP/JSON logs).

The stream-json transcript lost per-turn timing + real output counts in the newer CLI. The CLI's
OTel telemetry does NOT: it emits, with real wall-clock timestamps and durations,

  * ``claude_code.api_request``  — one per API turn: input/output/cache_read/cache_creation tokens,
    ``cost_usd``, ``duration_ms`` (the model-active/decode span), ``model``, ``query_source``.
  * ``claude_code.tool_result``  — one per tool call: ``tool_name``, ``duration_ms`` (real execution
    time), ``success``.

So tokens/cost are EXACT per turn (summed → the billed totals) and the activity timeline is built
from REAL spans (model-compute from api_request, tool execution from tool_result) — no interpolation,
no heuristic timeline. This is the capture the runner enables with ``ABT_TRACKING=full`` (an OTLP
sink per run; see bench/otel_sink.py).

Capture with :func:`bench.otel_sink` → an OTLP/JSON logs file (one envelope per line). Point
:func:`import_otel` at it.
"""
from __future__ import annotations

import json
from pathlib import Path

from aet.trajectory.model import ActivityBand, RoundBoundary, RunTrajectory, TrajectoryPoint, TestMilestone


# tool_name → activity category. Bash is split by duration: a long Bash is a tool-wait (in the
# checker-service flow the agent's ./run.sh blocks on the out-of-process verilator compile, so its
# duration IS the tool-wait), a quick Bash (ls/cat) is ordinary shell.
_READ = {"Read", "Glob", "Grep", "NotebookRead"}
_WRITE = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
_BASH_TOOLWAIT_MS = 3000


def _av(a: dict):
    v = a.get("value", {}) or {}
    for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if k in v:
            return v[k]
    return None


def parse_otel_logs(path: str | Path) -> list[dict]:
    """Flatten an OTLP/JSON logs capture into ordered normalized events.

    Returns dicts ``{seq, t_ns, name, attrs}`` sorted by (event.sequence, timestamp)."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for ln in p.read_text(errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            env = json.loads(ln)
        except Exception:
            continue
        if env.get("kind") != "logs":
            continue
        for rl in env.get("payload", {}).get("resourceLogs", []) or []:
            for sl in rl.get("scopeLogs", []) or []:
                for r in sl.get("logRecords", []) or []:
                    b = r.get("body")
                    name = b if isinstance(b, str) else (b or {}).get("stringValue")
                    attrs = {a["key"]: _av(a) for a in r.get("attributes", []) or []}
                    try:
                        t_ns = int(r.get("timeUnixNano") or attrs.get("event.timestamp_ns") or 0)
                    except (TypeError, ValueError):
                        t_ns = 0
                    try:
                        seq = int(attrs.get("event.sequence"))
                    except (TypeError, ValueError):
                        seq = 10 ** 9
                    out.append({"seq": seq, "t_ns": t_ns, "name": name, "attrs": attrs})
    out.sort(key=lambda e: (e["seq"], e["t_ns"]))
    return out


_TOOLWAIT_MARKERS = ("run.sh", "run_test.py", "verilator", "obj_dir", "sim_build", "make ", "cmake")


def _tool_category(tool_name: str, duration_ms: float, command: str | None = None) -> str:
    if tool_name in _READ:
        return "read"
    if tool_name in _WRITE:
        return "write"
    if tool_name == "Bash":
        # precise when the command is known (cross-ref'd from the transcript): a
        # verilator/run.sh/build command is a tool-wait, everything else is ordinary shell. Fall
        # back to a duration threshold only when the command is unavailable.
        if command is not None:
            return "tool" if any(m in command for m in _TOOLWAIT_MARKERS) else "bash"
        return "tool" if duration_ms >= _BASH_TOOLWAIT_MS else "bash"
    return "bash"


def activity_breakdown(events: list[dict], tool_cmds: dict | None = None) -> dict:
    """Ground-truth activity seconds per category, straight from OTel's own ``duration_ms`` fields.

    This is the VERIFIABLE decomposition: ``think`` = Σ ``api_request.duration_ms`` (model-compute),
    and each tool category = Σ ``tool_result.duration_ms`` for that tool. No layout, no heuristic
    timeline — just claude's reported durations summed by category. The plotted bands must reproduce
    exactly these totals (see tests)."""
    tool_cmds = tool_cmds or {}
    out: dict[str, float] = {}
    for e in events:
        a = e["attrs"]
        try:
            d = float(a.get("duration_ms") or 0) / 1000.0
        except (TypeError, ValueError):
            d = 0.0
        if d <= 0:
            continue
        if e["name"] == "claude_code.api_request":
            cat = "think"
        elif e["name"] == "claude_code.tool_result":
            cat = _tool_category(str(a.get("tool_name") or ""), float(a.get("duration_ms") or 0),
                                 tool_cmds.get(a.get("tool_use_id")))
        else:
            continue
        out[cat] = out.get(cat, 0.0) + d
    return out


def build_from_otel_events(events: list[dict], *, n_passed=None, n_total=1, tool_cmds=None):
    """Build (points, bands, totals, duration_s, milestones) from normalized OTel events.

    Pure + testable. Tokens/cost from ``api_request`` (exact per turn). Activity bands are a
    CONTIGUOUS, NON-OVERLAPPING partition: events are laid out in time order with a cursor
    (``start = max(real_start, prev_end)``), so each band's duration EXACTLY equals the event's
    OTel-reported ``duration_ms`` (the share is ground-truth), overlaps are impossible, and any
    residual wall-time is explicit idle. ``tool_cmds`` (tool_use_id→command, from the transcript)
    makes Bash classification precise (verilator/run.sh → tool-wait)."""
    tool_cmds = tool_cmds or {}
    reqs = [e for e in events if e["name"] == "claude_code.api_request"]
    tools = [e for e in events if e["name"] == "claude_code.tool_result"]
    if not reqs:
        return None

    def _i(a, k):
        try:
            return int(a.get(k) or 0)
        except (TypeError, ValueError):
            return 0

    def _f(a, k):
        try:
            return float(a.get(k) or 0)
        except (TypeError, ValueError):
            return 0.0

    # t0 = earliest activity START = min(completion − duration): a turn's work precedes its
    # completion timestamp, so anchoring at the first completion would drop the first turn's span.
    t0 = min((e["t_ns"] - int(_f(e["attrs"], "duration_ms") * 1e6))
             for e in events if e["t_ns"] > 0)

    def sec(e):
        return max(0.0, (e["t_ns"] - t0) / 1e9)

    # CRITICAL: a log event's ``timeUnixNano`` is the event's COMPLETION time, and ``duration_ms`` is
    # how long that api call / tool ran. So each event's real span is [t − duration, t] (the work
    # happened BEFORE the completion timestamp). A 14-min extended-thinking turn shows up as one
    # api_request completing at T with duration≈840s → span [T−840, T]. The wall duration is the last
    # completion time, NOT max(start+duration) — durations laid forward would overlap/overshoot.
    pts = []
    cin = cout = ccache = ccost = 0.0
    for e in sorted(reqs, key=lambda e: (e["t_ns"], e["seq"])):
        a = e["attrs"]
        cin += _i(a, "input_tokens") + _i(a, "cache_creation_tokens")
        cout += _i(a, "output_tokens")
        ccache += _i(a, "cache_read_tokens")
        ccost += _f(a, "cost_usd")
        pts.append(TrajectoryPoint(t_s=sec(e), cum_input_tokens=cin, cum_output_tokens=cout,
                                   cum_cache_tokens=ccache, cum_cost_usd=round(ccost, 6)))
    dur_s = max(0.001, max(sec(e) for e in events))     # last completion ≈ subprocess wall

    # activity bands: completion-anchored spans [t−duration, t], laid out as a NON-OVERLAPPING
    # partition. Sort by completion time; clamp each span's start up to the previous end so
    # sequential turns never overlap (and a stray long/parallel duration can't overshoot). Per-
    # category band time == the OTel duration sum (activity_breakdown) — the share is ground-truth;
    # residual wall-time between spans is honest idle.
    spans = []
    for e in reqs:
        d = _f(e["attrs"], "duration_ms") / 1000.0
        spans.append((sec(e), "think", d))
    for e in tools:
        cat = _tool_category(str(e["attrs"].get("tool_name") or ""), _f(e["attrs"], "duration_ms"),
                             tool_cmds.get(e["attrs"].get("tool_use_id")))
        spans.append((sec(e), cat, _f(e["attrs"], "duration_ms") / 1000.0))
    spans.sort(key=lambda s: s[0])                       # by completion time
    bands = []
    prev_end = 0.0
    for end_s, cat, d in spans:
        if d <= 0:
            continue
        s0 = max(end_s - d, prev_end)                    # span ends at completion; clamp start ≥ prev
        s1 = max(end_s, s0)
        if s1 - s0 < 1e-9:
            continue
        bands.append(ActivityBand(category=cat, t0_s=s0, t1_s=s1))
        prev_end = s1
    dur_s = max(dur_s, prev_end)

    totals = {"input": int(cin), "output": int(cout), "cache": int(ccache), "cost": round(ccost, 6)}
    milestones = []
    if n_passed is not None:
        milestones = [TestMilestone(t_s=dur_s, n_passed=int(n_passed), n_total=int(n_total),
                                    scope="all", source="terminal_verdict")]
    return pts, bands, totals, dur_s, milestones


def _tool_cmds_from_transcript(transcript_path) -> dict:
    """Map tool_use_id → command (Bash) / file_path, from the stream-json transcript, so OTel tool
    spans (which carry only tool_name) can be classified precisely (verilator/run.sh → tool-wait)."""
    out: dict = {}
    p = Path(transcript_path)
    if not p.exists():
        return out
    for ln in p.read_text(errors="ignore").splitlines():
        ln = ln.strip()
        if '"tool_use"' not in ln:
            continue
        try:
            o = json.loads(ln)
        except Exception:
            continue
        for b in (o.get("message", {}) or {}).get("content", []) or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                inp = b.get("input") or {}
                out[b.get("id")] = inp.get("command") or inp.get("file_path") or ""
    return out


def import_otel(logs_path: str | Path, *, run_id: str = "", n_passed=None, n_total: int = 1,
                transcript_path=None, **_ignored) -> RunTrajectory | None:
    """Build a full-fidelity :class:`RunTrajectory` from an OTLP logs capture, or None if unusable.

    ``transcript_path`` (defaults to a sibling ``transcript.jsonl``) is used only to map tool_use_id
    → command for precise Bash tool-wait classification."""
    events = parse_otel_logs(logs_path)
    if transcript_path is None:
        _sib = Path(logs_path).parent / "transcript.jsonl"
        transcript_path = _sib if _sib.exists() else None
    tool_cmds = _tool_cmds_from_transcript(transcript_path) if transcript_path else {}
    built = build_from_otel_events(events, n_passed=n_passed, n_total=n_total, tool_cmds=tool_cmds)
    if built is None:
        return None
    pts, bands, totals, dur_s, milestones = built
    traj = RunTrajectory(run_id=run_id or "otel", source="import:otel")
    traj.points = pts
    traj.bands = bands
    traj.milestones = milestones
    traj.rounds = [RoundBoundary(index=0, t_start_s=0.0, t_end_s=dur_s,
                                 input_tokens=totals["input"], output_tokens=totals["output"],
                                 cache_tokens=totals["cache"], cost_usd=totals["cost"])]
    traj.duration_s = dur_s          # plain field (active wall); consumers read it directly
    traj.num_rounds = 1
    traj.final_input_tokens = totals["input"]
    traj.final_output_tokens = totals["output"]
    traj.final_cache_tokens = totals["cache"]
    traj.final_cost_usd = totals["cost"]
    return traj
