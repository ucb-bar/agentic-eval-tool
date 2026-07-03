"""Generic Claude Code session/transcript importer → a canonical :class:`RunTrajectory`.

This is the **repo-agnostic default** importer: point it at one — or many — Claude Code
``*.jsonl`` transcripts and it produces a single trajectory, with **zero project-specific code**.
It is what lets *any* project that emits stream-json (abc-testing, a bare ``claude --print`` run,
recovered desktop session logs) get the same recording + plots as the bespoke capsule-bench loop.

It handles the two on-disk shapes the ecosystem produces, and degrades gracefully:

  * **CLI ``stream-json``** (``system``/``assistant``/``user``/``result``): the ``result`` event
    carries the authoritative ``total_cost_usd`` → **exact billed cost**. A single file may
    concatenate several invocations (several ``result`` events) — each becomes its own round.
  * **Desktop / app session log** (``assistant`` messages with ``message.usage`` + a ``timestamp``,
    interleaved with ``queue-operation``/``attachment``/``last-prompt``/``ai-title`` — and **no
    ``result`` event**): tokens + activity timing extract fine; there is no billed number, so cost
    is the list-price **provisional** estimate (points flagged ``provisional_cost``).

Multiple files (a directory, or an explicit list) are ordered by their first embedded timestamp and
appended as consecutive rounds on one active-wall axis. With no test-pass signal, ``milestones``
stays empty and the tests-over-time views degrade to the final verdict (or are skipped). An optional
terminal ``pass_bool``/``n_total`` (e.g. abc-testing's ``functional_pass``) is recorded as the last
round's QA verdict so a single-shot pass/fail still shows up.
"""
from __future__ import annotations

import glob
from pathlib import Path

from aet.tracking.claude_stream import parse_stream, parse_timestamped_stream
from aet.trajectory.build import append_round
from aet.trajectory.classify import (
    ActivityClassifier, ActivityConfig, capsule_bench_config,
)
from aet.trajectory.importers.capsule_bench import _round_events, _split_at_results
from aet.trajectory.model import RunTrajectory, TestMilestone
from aet.trajectory.oracle import extract_oracle_progression


def _transcript_files(raw: str | Path) -> list[Path]:
    """Resolve ``raw`` to an ordered list of transcript files.

    A single file → ``[file]``. A directory → every ``*.jsonl`` beneath it (``transcript.jsonl``
    first when present, then the rest), ordered by each file's first embedded ISO timestamp so
    session logs replay in wall-clock order regardless of filename."""
    p = Path(raw)
    if p.is_file():
        return [p]
    if not p.is_dir():
        return []
    files = sorted(Path(x) for x in glob.glob(str(p / "**" / "*.jsonl"), recursive=True))
    files = [f for f in files if f.name != "trajectory.json"]

    def _first_ts(f: Path) -> float:
        ev = _round_events(f)
        return ev[0][0] if ev else float("inf")

    # stable order: by first timestamp, then by name (files with no timestamp sink to the end)
    return sorted(files, key=lambda f: (_first_ts(f), f.name))


def _resolve_classifier(classifier_config, circt: bool | None) -> tuple[ActivityClassifier, dict]:
    """A classifier + its serialisable config. Generic by default; ``circt`` opts into the
    capsule-bench long-wait rules for RTL tooling without importing any project specifics here."""
    if classifier_config is not None:
        cfg = classifier_config
    elif circt:
        cfg = capsule_bench_config(circt=True)
    else:
        cfg = ActivityConfig()
    return ActivityClassifier(cfg), cfg.to_dict()


def import_transcript(raw: str | Path, *,
                      classifier_config: ActivityConfig | None = None,
                      circt: bool | None = None,
                      run_id: str = "",
                      label: str | None = None,
                      pass_bool: bool | None = None,
                      n_passed: int | None = None,
                      n_total: int = 1,
                      oracle_markers: "list[str] | None" = None,
                      milestone_time: str = "proportional",  # accepted for CLI uniformity; unused
                      **_ignored) -> RunTrajectory:
    """Ingest one or many Claude Code transcripts into a canonical :class:`RunTrajectory`.

    Terminal verdict: pass ``pass_bool`` for an all-or-nothing boolean grade, or ``n_passed`` +
    ``n_total`` for a ``k/N`` grade (e.g. abc-testing's ``cases_total`` − failed cases). Either
    records the last round's QA verdict + a single end-of-run milestone."""
    # a k/N grade takes precedence over the boolean when both are given
    if n_passed is not None:
        term_passed, term_total = int(n_passed), int(n_total)
    elif pass_bool is not None:
        term_passed, term_total = (int(n_total) if pass_bool else 0), int(n_total)
    else:
        term_passed = term_total = None
    files = _transcript_files(raw)
    classifier, cfg_dict = _resolve_classifier(classifier_config, circt)

    rid = run_id or label or (Path(raw).stem if Path(raw).is_file() else Path(raw).name)
    traj = RunTrajectory(run_id=rid, source="import:transcript", classifier_config=cfg_dict)

    # Flatten every file into ordered (result, raw_segment) pairs, so a directory of session logs
    # and a single multi-invocation file are the same one code path. The raw segment is kept so the
    # oracle-progression extractor can read the agent's testbench invocations on the same time axis.
    parsed: list = []
    for f in files:
        events = _round_events(f)
        if not events:
            continue
        for seg in _split_at_results(events):
            base = seg[0][0]
            rebased = [(ts - base, line) for ts, line in seg]
            try:
                result = parse_timestamped_stream(rebased)
            except Exception:
                result = parse_stream("\n".join(line for _, line in seg))
            if result.turn_usage:              # skip noise segments with no agent turns
                parsed.append((result, seg))

    climb: list[TestMilestone] = []
    for i, (result, seg) in enumerate(parsed):
        t0 = traj.duration_s                   # this round starts here on the active-wall axis
        # a single terminal k/N (or boolean) grade is attached as the last round's QA verdict
        verdict = None
        if term_passed is not None and i == len(parsed) - 1:
            verdict = {"n_passed": term_passed, "n_total": term_total}
        append_round(traj, result, classifier=classifier, verdict=verdict)
        # oracle progression: each testbench invocation in this segment → a milestone at its wall time
        if oracle_markers is not None:
            hint = term_total if term_total else (n_total or None)
            for r in extract_oracle_progression(seg, markers=tuple(oracle_markers),
                                                n_total_hint=hint):
                climb.append(TestMilestone(t_s=t0 + r.t_s, n_passed=r.n_passed,
                                           n_total=r.n_total, scope="public",
                                           source="oracle_log"))

    if climb:
        # the mined climb wins (multiple over-time milestones — the real curve)
        traj.milestones = climb
    elif term_passed is not None and traj.duration_s > 0:
        # else the terminal grade surfaces as a single end-of-run milestone so tests-over-time views
        # still have something to draw (graceful degradation to 'no intermediate progression').
        traj.milestones = [TestMilestone(
            t_s=traj.duration_s, n_passed=term_passed, n_total=term_total,
            scope="all", source="terminal_verdict")]
    _densify_from_estimated(raw, traj)
    return traj


def densify_new_format(events: list, duration_s: float, final_cost_usd: float = 0.0):
    """Build a dense, per-signal point series from a newer Claude CLI stream (pure fn, testable).

    Returns ``(points, totals)`` or ``(None, None)`` when the stream is the OLD format (no thinking
    stream) or lacks a ``result`` event — caller keeps the per-turn path in that case.

    The newer CLI exposes each signal DIFFERENTLY, so we model them distinctly (not one shared shape):
      * input / cache_read  — REAL per-message ``usage`` (verified to match the result-event totals
        exactly), accumulated at each unique assistant message;
      * output              — ``usage.output_tokens`` counts only *visible* output (tiny under extended
        thinking); the bulk is thinking, which streams as ``estimated_tokens_delta`` (subtype
        ``thinking_tokens``). output curve = visible + thinking, then scaled so its END equals the
        authoritative ``result.usage.output_tokens``;
      * totals / cost       — the ``result`` event (``usage`` + ``total_cost_usd``) is claude's own
        billed truth and is the ground the curves are pinned to.
    Time: assistant/thinking events carry no timestamp, so events are placed by interpolating between
    the real tool-result (``user``) timestamps (endpoints 0 and ``duration_s``)."""
    import datetime as _dt
    from aet.trajectory.model import TrajectoryPoint as _TP

    usage = None
    cost_tot = final_cost_usd or 0.0
    for o in events:
        if o.get("type") == "result":
            if isinstance(o.get("usage"), dict):
                usage = o["usage"]
            if o.get("total_cost_usd") is not None:
                cost_tot = float(o["total_cost_usd"])
    has_think = any("estimated_tokens_delta" in o for o in events)
    if usage is None or not has_think or duration_s <= 0:
        return None, None                       # old format → caller keeps the per-turn path

    seen = set()
    cin = ccache = cvis = cthink = 0.0
    samples = []                                 # (event_index, cum_in, cum_out_raw, cum_cache)
    for i, o in enumerate(events):
        t = o.get("type")
        if t == "assistant":
            m = o.get("message", {}) or {}
            mid = m.get("id")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            u = m.get("usage") or {}
            cin += float(u.get("input_tokens") or 0) + float(u.get("cache_creation_input_tokens") or 0)
            ccache += float(u.get("cache_read_input_tokens") or 0)
            cvis += float(u.get("output_tokens") or 0)
        elif "estimated_tokens_delta" in o:
            cthink += float(o.get("estimated_tokens_delta") or 0)
        else:
            continue
        samples.append((i, cin, cvis + cthink, ccache))
    if len(samples) < 4:
        return None, None

    out_t = float(usage.get("output_tokens") or 0)
    in_t = float(usage.get("input_tokens") or 0) + float(usage.get("cache_creation_input_tokens") or 0)
    ca_t = float(usage.get("cache_read_input_tokens") or 0)
    fin_in, fin_out, fin_ca = samples[-1][1], samples[-1][2], samples[-1][3]
    si = in_t / fin_in if fin_in > 0 else 0.0
    so = out_t / fin_out if fin_out > 0 else 0.0
    scch = ca_t / fin_ca if fin_ca > 0 else 0.0
    tot_final = fin_in + fin_out + fin_ca or 1.0

    anchors = []
    t0 = None
    for i, o in enumerate(events):
        ts = o.get("timestamp")
        if not ts:
            continue
        try:
            dt = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            continue
        if t0 is None:
            t0 = dt
        anchors.append((i, (dt - t0).total_seconds()))
    if anchors and anchors[-1][1] > 0:
        sc = duration_s / anchors[-1][1]
        anchors = [(i, t * sc) for i, t in anchors]
    anchors = [(0, 0.0)] + anchors + [(samples[-1][0], duration_s)]

    def _t(idx):
        prev = (0, 0.0)
        for a in anchors:
            if a[0] >= idx:
                if a[0] == prev[0]:
                    return a[1]
                f = (idx - prev[0]) / (a[0] - prev[0])
                return prev[1] + f * (a[1] - prev[1])
            prev = a
        return duration_s

    pts = []
    for (i, civ, cov, ccv) in samples:
        fcost = (civ + cov + ccv) / tot_final          # cost tracks total-token progress
        pts.append(_TP(t_s=_t(i), cum_input_tokens=civ * si, cum_output_tokens=cov * so,
                       cum_cache_tokens=ccv * scch, cum_cost_usd=round(cost_tot * fcost, 6)))
    if pts and pts[-1].t_s < duration_s:
        pts[-1].t_s = duration_s
    return pts, {"input": int(in_t), "output": int(out_t), "cache": int(ca_t),
                 "cost": round(cost_tot, 6)}


def _densify_from_estimated(raw, traj) -> None:
    """Apply :func:`densify_new_format` to a transcript file/text; no-op on the old format."""
    try:
        import json as _json
        from pathlib import Path as _P
        text = _P(str(raw)).read_text(errors="ignore") if _P(str(raw)).is_file() else str(raw)
        events = []
        for ln in text.splitlines():
            ln = ln.strip()
            if ln.startswith("{"):
                try:
                    events.append(_json.loads(ln))
                except Exception:
                    pass
        pts, totals = densify_new_format(events, traj.duration_s, traj.final_cost_usd or 0.0)
        if pts is None:
            return
        traj.points = pts
        traj.final_input_tokens = totals["input"]
        traj.final_output_tokens = totals["output"]
        traj.final_cache_tokens = totals["cache"]
        traj.final_cost_usd = totals["cost"]
    except Exception:
        return                                            # never break import on the densify path
