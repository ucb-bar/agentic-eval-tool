"""Import a Gemmini capsule-bench run directory into a canonical :class:`RunTrajectory`.

Layout consumed (all optional except ``rounds/``):
  * ``rounds/round_*.transcript.jsonl`` — one ``stream-json`` transcript per agent round, whose
    lines carry a per-event ISO ``timestamp`` (so we recover real per-tool timing).
  * ``qa_history/verdict_round_*.json`` — per-round QA verdict (``n_passed`` / ``n_capsules``).
  * ``selfcheck_log.jsonl`` — full-suite self-check rows; the strictly-increasing all-scope
    ``n_passed`` values are the gold test-pass milestones (e.g. 13 → 17 → 20).

This reproduces the semantics of the oscar-merlin ``load_arm``/``_fine_milestones`` reference,
but uses the canonical parser's real timestamps instead of within-round weighting.
"""
from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path

from aet.tracking.claude_stream import parse_stream, parse_timestamped_stream
from aet.trajectory.build import append_round
from aet.trajectory.classify import ActivityClassifier, ActivityConfig, capsule_bench_config
from aet.trajectory.model import RunTrajectory, TestMilestone


def _iso_to_epoch(ts: str) -> float | None:
    """Parse an ISO-8601 transcript timestamp (``...Z``) to epoch seconds."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _round_events(transcript: Path) -> list[tuple[float, str]]:
    """(epoch_s, raw_line) pairs. Lines missing a ``timestamp`` carry the previous line's time
    forward (and lead-in lines borrow the first real one), so every value stays on the epoch
    scale — never mixing epoch seconds with a line-index fallback (which would explode the axis).
    """
    raw: list[tuple[float | None, str]] = []
    for line in transcript.read_text().splitlines():
        if not line.strip():
            continue
        ts = None
        try:
            ts = _iso_to_epoch(json.loads(line).get("timestamp", ""))
        except Exception:
            ts = None
        raw.append((ts, line))
    if not raw:
        return []
    first_real = next((t for t, _ in raw if t is not None), 0.0)
    events: list[tuple[float, str]] = []
    last = first_real
    for ts, line in raw:
        if ts is not None:
            last = ts
        events.append((last, line))
    return events


def _split_at_results(events: list[tuple[float, str]]) -> list[list[tuple[float, str]]]:
    """Split a transcript into segments, one per ``result`` event. A single transcript file can
    concatenate several ``claude`` invocations (e.g. a resumed round) — each has its own
    authoritative ``result``/cost and must be counted as its own segment, else cost is undercounted.
    A trailing segment without a result (in-flight) is kept too."""
    segments: list[list[tuple[float, str]]] = []
    cur: list[tuple[float, str]] = []
    for ts, line in events:
        cur.append((ts, line))
        try:
            is_result = json.loads(line).get("type") == "result"
        except Exception:
            is_result = False
        if is_result:
            segments.append(cur)
            cur = []
    if cur:
        segments.append(cur)
    return segments


def _load_verdict(run_dir: Path, k: int) -> dict | None:
    for name in (f"verdict_round_{k:02d}.json", f"verdict_round_{k}.json"):
        p = run_dir / "qa_history" / name
        if p.is_file():
            try:
                d = json.loads(p.read_text())
                return {"n_passed": d.get("n_passed"), "n_capsules": d.get("n_capsules")}
            except Exception:
                return None
    return None


def _cumulative_wall(rows: list[dict]) -> list[float]:
    """Reconstruct a monotone global wall from a self-check log whose ``wall_offset_s`` RESETS
    each round (the log is appended per round). A drop signals a new round: carry the prior
    round's span forward so the reconstructed clock never goes backwards."""
    out: list[float] = []
    base = 0.0
    prev = 0.0
    seg_max = 0.0
    for r in rows:
        w = float(r.get("wall_offset_s") or 0.0)
        if w < prev:                      # offset went backwards → a new round started
            base += seg_max
            seg_max = 0.0
        out.append(base + w)
        seg_max = max(seg_max, w)
        prev = w
    return out


def _fine_milestones(run_dir: Path, total_s: float, mode: str) -> list[TestMilestone]:
    """Strictly-increasing all-scope self-check passes → gold milestones, on the active axis.

    Handles per-round ``wall_offset_s`` resets by reconstructing a cumulative clock, so the
    13→17→20 progression stays time-ordered even across rounds."""
    p = run_dir / "selfcheck_log.jsonl"
    if not p.is_file() or total_s <= 0:
        return []
    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    if not rows:
        return []
    walls = _cumulative_wall(rows)
    max_wall = max(walls) or 1.0
    best = 0
    out: list[TestMilestone] = []
    for r, gwall in zip(rows, walls):
        if str(r.get("capsules")) != "all":
            continue
        n_total = r.get("n_capsules") or 0
        n_pass = r.get("n_passed") or 0
        if n_total < 20 or n_pass <= best:
            continue
        best = n_pass
        t_s = gwall if mode == "wallclock" else (gwall / max_wall * total_s)
        out.append(TestMilestone(t_s=t_s, n_passed=n_pass, n_total=n_total,
                                 scope="all", source="selfcheck_log"))
    return out


def import_run(raw_run_dir: str | Path, *,
               classifier_config: ActivityConfig | None = None,
               circt: bool | None = None,
               milestone_time: str = "proportional",
               run_id: str = "") -> RunTrajectory:
    """Read a capsule-bench run directory → canonical :class:`RunTrajectory`."""
    run_dir = Path(raw_run_dir)
    if circt is None:
        circt = "circt" in run_dir.name.lower()
    cfg = classifier_config or capsule_bench_config(circt=circt)
    classifier = ActivityClassifier(cfg)

    traj = RunTrajectory(
        run_id=run_id or run_dir.name,
        source="import:capsule-bench",
        classifier_config=cfg.to_dict(),
    )

    transcripts = sorted(glob.glob(str(run_dir / "rounds" / "round_*.transcript.jsonl")))
    for k, tp in enumerate(transcripts):
        events = _round_events(Path(tp))
        if not events:
            continue
        # one file may concatenate several invocations → split so each result/cost is counted;
        # the file's QA verdict describes the round outcome, so attach it to the last segment.
        segments = _split_at_results(events)
        parsed = []
        for seg in segments:
            base = seg[0][0]
            rebased = [(ts - base, line) for ts, line in seg]
            try:
                result = parse_timestamped_stream(rebased)
            except Exception:
                result = parse_stream("\n".join(line for _, line in seg))
            if result.turn_usage:              # drop trailing/noise segments with no agent turns
                parsed.append(result)
        for si, result in enumerate(parsed):
            verdict = _load_verdict(run_dir, k) if si == len(parsed) - 1 else None
            append_round(traj, result, classifier=classifier, verdict=verdict)

    traj.milestones = _fine_milestones(run_dir, traj.duration_s, milestone_time)
    return traj
