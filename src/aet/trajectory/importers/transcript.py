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

    # Flatten every file into ordered (result, is_terminal_segment) segments, so a directory of
    # session logs and a single multi-invocation file are the same one code path.
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
                parsed.append(result)

    for i, result in enumerate(parsed):
        # a single terminal k/N (or boolean) grade is attached as the last round's QA verdict
        verdict = None
        if term_passed is not None and i == len(parsed) - 1:
            verdict = {"n_passed": term_passed, "n_total": term_total}
        append_round(traj, result, classifier=classifier, verdict=verdict)

    # The terminal grade also surfaces as a single end-of-run milestone so tests-over-time views have
    # something to draw (they otherwise degrade to empty — no intermediate progression exists here).
    if term_passed is not None and traj.duration_s > 0:
        traj.milestones = [TestMilestone(
            t_s=traj.duration_s, n_passed=term_passed, n_total=term_total,
            scope="all", source="terminal_verdict")]
    return traj
