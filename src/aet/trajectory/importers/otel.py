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

from aet.trajectory.model import (
    ActivityBand, InferenceRecord, RoundBoundary, RunTrajectory, TrajectoryPoint, TestMilestone,
)


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
            if env.get("kind") == "traces":
                for rs in env.get("payload", {}).get("resourceSpans", []) or []:
                    for ss in rs.get("scopeSpans", []) or []:
                        for span in ss.get("spans", []) or []:
                            attrs = {a["key"]: _av(a) for a in span.get("attributes", []) or []}
                            name = span.get("name") or ""
                            # Normalize an inference span to the same event vocabulary
                            # as Claude's logs path; unknown spans remain available for
                            # hierarchy/activity consumers without inventing usage.
                            if name in ("claude_code.api_request", "gen_ai.client.operation"):
                                normalized = "claude_code.api_request"
                            elif name in ("claude_code.tool_result", "gen_ai.tool.call"):
                                normalized = "claude_code.tool_result"
                            else:
                                normalized = name
                            try:
                                t_ns = int(span.get("endTimeUnixNano") or 0)
                            except (TypeError, ValueError):
                                t_ns = 0
                            if span.get("startTimeUnixNano") and t_ns:
                                try:
                                    attrs.setdefault("duration_ms", (
                                        t_ns - int(span["startTimeUnixNano"])) / 1e6)
                                except (TypeError, ValueError):
                                    pass
                            attrs.setdefault("trace_id", span.get("traceId", ""))
                            attrs.setdefault("span_id", span.get("spanId", ""))
                            attrs.setdefault("parent_span_id", span.get("parentSpanId", ""))
                            out.append({"seq": len(out), "t_ns": t_ns, "signal": "trace",
                                        "name": normalized, "attrs": attrs})
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
                    out.append({"seq": seq, "t_ns": t_ns, "signal": "log",
                                "name": name, "attrs": attrs})
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


def build_from_otel_events(events: list[dict], *, n_passed=None, n_total=1, tool_cmds=None,
                           think_cum=None):
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
    cin = cout = ccache_read = ccache_write = creason = ccost = 0.0
    for e in sorted(reqs, key=lambda e: (e["t_ns"], e["seq"])):
        a = e["attrs"]
        # These are distinct priced input classes. Cache creation used to be
        # added into fresh input and then lost from cache, overstating fresh
        # tokens and understating total cache activity.
        cin += _i(a, "input_tokens")
        cout += _i(a, "output_tokens")
        ccache_read += _i(a, "cache_read_tokens")
        ccache_write += _i(a, "cache_creation_tokens")
        creason += _i(a, "reasoning_tokens")
        ccost += _f(a, "cost_usd")
        pts.append(TrajectoryPoint(t_s=sec(e), cum_input_tokens=cin, cum_output_tokens=cout,
                                   cum_cache_tokens=ccache_read + ccache_write,
                                   cum_cache_read_tokens=ccache_read,
                                   cum_cache_creation_tokens=ccache_write,
                                   cum_reasoning_tokens=creason,
                                   cum_cost_usd=round(ccost, 6)))
    dur_s = max(0.001, max(sec(e) for e in events))     # last completion ≈ subprocess wall

    # SUB-TURN density (optional): OTel gives one exact point per turn, so a long extended-thinking
    # turn is a single flat point. The transcript's dense estimated_tokens (thinking) stream is the
    # real sub-turn progression — distribute its samples across each turn's [start, completion] window
    # so the output curve CLIMBS through a long think instead of jumping once. Scaled so the endpoint
    # still equals the exact per-turn output total (anchors preserved). Input/cache are prompt-side
    # (they step at turn boundaries) so they stay per-turn.
    if think_cum and len(think_cum) >= 4 and pts:
        req_spans = []           # (start_s, end_s, cum_out_at_turn_end, cum_in, cum_cache, cum_cost)
        cin2 = cout2 = cca_read2 = cca_write2 = ccst2 = 0.0
        for e in sorted(reqs, key=lambda e: (e["t_ns"], e["seq"])):
            a = e["attrs"]
            d = _f(a, "duration_ms") / 1000.0
            end = sec(e)
            cin2 += _i(a, "input_tokens")
            cout2 += _i(a, "output_tokens")
            cca_read2 += _i(a, "cache_read_tokens")
            cca_write2 += _i(a, "cache_creation_tokens")
            ccst2 += _f(a, "cost_usd")
            req_spans.append((max(0.0, end - d), end, cout2, cin2,
                              cca_read2, cca_write2, round(ccst2, 6)))
        think_total = think_cum[-1] or 1.0
        out_total = cout2 or 1.0
        # even-distribute the N think samples across the concatenated think windows, in order
        n = len(think_cum)
        win_durs = [max(1e-6, s[1] - s[0]) for s in req_spans]
        win_tot = sum(win_durs) or 1.0
        dense = []
        acc = 0
        for wi, (s0, s1, co, ci, cr, cw, ccost) in enumerate(req_spans):
            k = max(1, round(n * win_durs[wi] / win_tot))
            for j in range(k):
                if acc >= n:
                    break
                frac_in_win = (j + 1) / k
                t = s0 + frac_in_win * (s1 - s0)
                cthink = think_cum[min(acc, n - 1)]
                acc += 1
                # output scaled from the thinking progression to the exact grand total
                dense.append(TrajectoryPoint(
                    t_s=t, cum_output_tokens=(cthink / think_total) * out_total,
                    cum_input_tokens=ci, cum_cache_tokens=cr + cw,
                    cum_cache_read_tokens=cr, cum_cache_creation_tokens=cw,
                    cum_cost_usd=ccost))
        if dense:
            dense[-1] = pts[-1]                          # pin the exact final totals
            pts = sorted(dense, key=lambda p: p.t_s)

    # activity bands: a GAPLESS wall-time partition. Events are ordered by completion time and each
    # interval [previous completion, this completion] is attributed to the activity that COMPLETES at
    # its right edge — i.e. the thing that was running during that interval (a model turn completing
    # at T was active since the prior event; a tool completing at T ran since the prior event). This
    # covers 100% of the wall with no gaps and no overlap, using only real event timestamps, so the
    # share = fraction of WALL TIME per activity (inter-turn overhead is attributed to the turn it
    # belongs to, not dropped). ``activity_breakdown`` (Σ duration_ms) is the complementary
    # active-compute view; both are exposed and tested.
    marked = [(sec(e), "think") for e in reqs]
    for e in tools:
        cat = _tool_category(str(e["attrs"].get("tool_name") or ""), _f(e["attrs"], "duration_ms"),
                             tool_cmds.get(e["attrs"].get("tool_use_id")))
        marked.append((sec(e), cat))
    marked.sort(key=lambda m: m[0])                      # by completion time
    bands = []
    prev = 0.0
    for end_s, cat in marked:
        if end_s > prev + 1e-9:
            bands.append(ActivityBand(category=cat, t0_s=prev, t1_s=end_s))
            prev = end_s
    dur_s = max(dur_s, prev)

    totals = {"input": int(cin), "output": int(cout),
              "cache": int(ccache_read + ccache_write),
              "cache_read": int(ccache_read), "cache_creation": int(ccache_write),
              "cost": round(ccost, 6)}
    milestones = []
    if n_passed is not None:
        milestones = [TestMilestone(t_s=dur_s, n_passed=int(n_passed), n_total=int(n_total),
                                    scope="all", source="terminal_verdict")]
    return pts, bands, totals, dur_s, milestones


def inference_records_from_otel(events: list[dict]) -> list[InferenceRecord]:
    """Extract de-duplicated inference attempts from OTLP logs and spans.

    Logs are preferred when a CLI exports both signals. A request/span identity
    is the de-duplication key; missing identities fall back to timestamp+model.
    Cache TTL expiry is only labelled ``probable_expiry`` when an idle gap
    exceeds a declared TTL and a formerly cached session subsequently reports
    no cache reads.
    """
    requests = [event for event in events if event["name"] == "claude_code.api_request"]
    if not requests:
        return []
    starts = []
    for event in requests:
        try:
            duration_ns = int(float(event["attrs"].get("duration_ms") or 0) * 1e6)
        except (TypeError, ValueError):
            duration_ns = 0
        starts.append(max(0, event["t_ns"] - duration_ns))
    origin_ns = min(starts) if starts else 0

    def pick(attrs, *keys, default=None):
        for key in keys:
            value = attrs.get(key)
            if value not in (None, ""):
                return value
        return default

    def integer(attrs, *keys) -> int:
        try:
            return int(pick(attrs, *keys, default=0) or 0)
        except (TypeError, ValueError):
            return 0

    records: list[InferenceRecord] = []
    seen: set[str] = set()
    # Prefer log records when both signals describe the same request: Claude's
    # logs carry its native cache/cost vocabulary, while generic spans may omit it.
    ordered = sorted(requests, key=lambda e: (
        0 if e.get("signal") == "log" else 1, e["t_ns"], e["seq"],
    ))
    for index, event in enumerate(ordered):
        attrs = event["attrs"]
        request_id = str(pick(attrs, "request_id", "gen_ai.request.id", "span_id",
                              default=f"otel-{event['t_ns']}-{index}"))
        identities = {str(value) for value in (
            attrs.get("span_id"), attrs.get("gen_ai.request.id"), attrs.get("request_id"),
            request_id,
        ) if value not in (None, "")}
        if seen.intersection(identities):
            continue
        seen.update(identities)
        try:
            duration_s = max(0.0, float(attrs.get("duration_ms") or 0) / 1000.0)
        except (TypeError, ValueError):
            duration_s = 0.0
        end_s = max(0.0, (event["t_ns"] - origin_ns) / 1e9)
        try:
            cost = float(pick(attrs, "cost_usd", "gen_ai.usage.cost", default=None))
        except (TypeError, ValueError):
            cost = None
        context_window = integer(attrs, "context_window_tokens", "gen_ai.context_window.tokens") or None
        fresh = integer(attrs, "input_tokens", "gen_ai.usage.input_tokens")
        read = integer(attrs, "cache_read_tokens", "cache_read_input_tokens",
                       "gen_ai.usage.cache_read_tokens")
        write = integer(attrs, "cache_creation_tokens", "cache_write_tokens",
                        "gen_ai.usage.cache_write_tokens")
        try:
            ttl = float(pick(attrs, "cache_ttl_s", "gen_ai.cache.ttl_s", default=None))
        except (TypeError, ValueError):
            ttl = None
        records.append(InferenceRecord(
            request_id=request_id, t_start_s=max(0.0, end_s - duration_s), t_end_s=end_s,
            agent_id=str(pick(attrs, "agent_id", "gen_ai.agent.id", default="")),
            parent_agent_id=str(pick(attrs, "parent_agent_id", "gen_ai.agent.parent_id", default="")),
            trace_id=str(attrs.get("trace_id") or ""), span_id=str(attrs.get("span_id") or ""),
            call_id=str(attrs.get("call_id") or ""), session_id=str(pick(
                attrs, "session_id", "conversation_id", "gen_ai.conversation.id", default="")),
            attempt=integer(attrs, "attempt", "gen_ai.request.attempt") or 1,
            provider=str(pick(attrs, "provider", "gen_ai.provider.name", default="anthropic")),
            model=str(pick(attrs, "model", "gen_ai.response.model", default="")),
            input_tokens=fresh,
            output_tokens=integer(attrs, "output_tokens", "gen_ai.usage.output_tokens"),
            cache_read_tokens=read, cache_write_tokens=write,
            reasoning_tokens=integer(attrs, "reasoning_tokens", "gen_ai.usage.reasoning_tokens"),
            status=str(pick(attrs, "status", "gen_ai.response.finish_reasons", default="completed")),
            retry=bool(integer(attrs, "retry") or (integer(attrs, "attempt") > 1)),
            cost_usd=cost, cost_source="billed" if cost is not None else "unavailable",
            billing_mode=str(attrs.get("billing_mode") or "per_token"),
            context_window_tokens=context_window,
            estimated_context_tokens=fresh + read + write,
            cache_ttl_s=ttl,
        ))

    records.sort(key=lambda record: (record.t_end_s, record.request_id))
    previous: dict[str, InferenceRecord] = {}
    for record in records:
        key = record.session_id or record.agent_id or "unattributed"
        prior = previous.get(key)
        if (prior and record.cache_ttl_s and prior.cache_read_tokens > 0
                and record.cache_read_tokens == 0
                and record.t_start_s - prior.t_end_s > record.cache_ttl_s):
            record.ttl_inference = "probable_expiry"
        elif record.cache_ttl_s is not None:
            record.ttl_inference = "no_expiry_signal"
        previous[key] = record
    return records


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


def _think_cum_from_transcript(transcript_path) -> list:
    """Cumulative thinking-token progression from the transcript's dense estimated_tokens stream —
    the sub-turn resolution used to draw a climbing (not flat) rate through long thinking turns."""
    p = Path(transcript_path)
    if not p.exists():
        return []
    cum = 0.0
    out = []
    for ln in p.read_text(errors="ignore").splitlines():
        if "estimated_tokens_delta" not in ln:
            continue
        try:
            d = json.loads(ln).get("estimated_tokens_delta") or 0
        except Exception:
            continue
        cum += float(d)
        out.append(cum)
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
    think_cum = _think_cum_from_transcript(transcript_path) if transcript_path else None
    built = build_from_otel_events(events, n_passed=n_passed, n_total=n_total, tool_cmds=tool_cmds,
                                   think_cum=think_cum)
    if built is None:
        return None
    pts, bands, totals, dur_s, milestones = built
    traj = RunTrajectory(run_id=run_id or "otel", source="import:otel")
    traj.points = pts
    traj.bands = bands
    traj.milestones = milestones
    traj.rounds = [RoundBoundary(index=0, t_start_s=0.0, t_end_s=dur_s,
                                 input_tokens=totals["input"], output_tokens=totals["output"],
                                 cache_tokens=totals["cache"],
                                 cache_read_tokens=totals["cache_read"],
                                 cache_creation_tokens=totals["cache_creation"],
                                 cost_usd=totals["cost"])]
    traj.inferences = inference_records_from_otel(events)
    traj.duration_s = dur_s          # plain field (active wall); consumers read it directly
    traj.num_rounds = 1
    traj.final_input_tokens = totals["input"]
    traj.final_output_tokens = totals["output"]
    traj.final_cache_tokens = totals["cache"]
    traj.final_cache_read_tokens = totals["cache_read"]
    traj.final_cache_creation_tokens = totals["cache_creation"]
    traj.final_cost_usd = totals["cost"]
    return traj
