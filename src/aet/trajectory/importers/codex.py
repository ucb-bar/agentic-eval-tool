"""Codex-CLI JSONL importer → a canonical :class:`RunTrajectory`.

Ingests a ``codex exec --json`` stdout stream (raw ``events.raw.jsonl``, or a directory of them —
consecutive same-thread streams are appended as continued rounds, which is how a
``codex exec resume`` capture is imported). Uses the lossless structural normalizer in
``trajectory/codex.py`` (no regex, unknown events kept verbatim), then builds the trajectory:

  * one **round + point per ``turn.completed``**, on a pseudo-time axis derived from event ordinals
    when the raw stream carries no wall clock (the live recorder uses real per-line timestamps);
  * token buckets kept **non-overlapping** so ``cum_total`` never double-counts a subset —
    ``cum_input`` holds the *uncached* input, ``cache_read`` the cached-input subset, ``cache_creation``
    the cache-write subset, ``reasoning`` the reasoning-output subset (reported beside output, not
    added to it);
  * **activity bands / tool spans mapped from structured item fields** (``command_execution`` → a
    Bash/tool band, ``file_change`` → a write band) — never by string-parsing the command;
  * a nullable, fully-provenanced :class:`~aet.trajectory.cost.CostRecord` (unpriced ≠ $0).

**Idempotent:** the import is a pure function of the input bytes (cost provenance uses a fixed,
caller-supplied ``calculated_at`` — empty by default — so re-import yields a byte-identical
trajectory). **Reconcilable:** every raw line is accounted for (dispatched, kept as an unknown
event, or kept as an ``[UNPARSED]`` line); see :func:`aet.trajectory.reconcile.reconcile_codex`.
"""
from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path

from aet.trajectory.classify import ActivityClassifier, ActivityConfig
from aet.trajectory.codex import CodexNormalizer, CodexRun
from aet.trajectory.cost import CostRecord
from aet.trajectory.model import (
    ActivityBand, Checkpoint, RoundBoundary, RunTrajectory, TrajectoryPoint,
)
from aet.trajectory.price_snapshot import PriceSnapshot, cost_record_for

# Pseudo-seconds per event ordinal when the raw stream has no wall clock. A positive step keeps
# bands non-degenerate and preserves event ordering; it is NOT a claim about real latency.
_PSEUDO_DT_S = 1.0


def _codex_files(raw: str | Path) -> list[Path]:
    p = Path(raw)
    if p.is_file():
        return [p]
    if not p.is_dir():
        return []
    # sorted by name: a resume capture named e.g. events.0.jsonl / events.1.jsonl replays in order
    return sorted(Path(x) for x in glob.glob(str(p / "**" / "*.jsonl"), recursive=True))


def _timestamped_files(raw: str | Path | None) -> list[Path]:
    """Resolve timestamp sidecars without ever mixing them into the raw-event file set."""
    if raw is None:
        return []
    p = Path(raw)
    if p.is_file():
        return [p]
    if not p.is_dir():
        return []
    return sorted(Path(x) for x in glob.glob(str(p / "**" / "*.jsonl"), recursive=True))


def _normalize(files: list[Path], timestamped: str | Path | None = None) -> CodexRun:
    """Normalize raw files, optionally taking wall offsets from lossless timestamp sidecars.

    A sidecar record is ``{"ts": <ISO-8601>, "line": <raw line>}``.  Sidecars are a timing
    annotation for the raw stream, not another event stream: feeding both would double every turn
    and token bucket.  Fail closed on malformed or byte-divergent sidecars so a purported real-time
    trajectory can never quietly fall back to pseudo-time or reconcile against different bytes.
    """
    sidecars = _timestamped_files(timestamped)
    if not sidecars:
        norm = CodexNormalizer()
        for path in files:
            norm.feed_text(path.read_text(errors="ignore"))
        return norm.result()
    if len(sidecars) != len(files):
        raise ValueError(
            f"timestamp sidecar count ({len(sidecars)}) does not match raw file count ({len(files)})")

    norm = CodexNormalizer()
    origin: datetime | None = None
    for raw_path, sidecar_path in zip(files, sidecars, strict=True):
        raw_lines = raw_path.read_text(errors="ignore").splitlines()
        records = [json.loads(line) for line in sidecar_path.read_text(errors="ignore").splitlines()
                   if line.strip()]
        annotated = [record.get("line") for record in records]
        if annotated != raw_lines:
            raise ValueError(f"timestamp sidecar does not match raw events: {sidecar_path}")
        for record in records:
            when = datetime.fromisoformat(str(record["ts"]))
            origin = when if origin is None else origin
            norm.feed_line(str(record["line"]), t_s=max((when - origin).total_seconds(), 0.0))
    return norm.result()


def _classifier(classifier_config, circt):
    if classifier_config is not None:
        cfg = classifier_config
    else:
        cfg = ActivityConfig()
    return ActivityClassifier(cfg), cfg.to_dict()


def _tool_category(tc, classifier: ActivityClassifier) -> tuple[str, float, str, bool]:
    """Structured item → (category, weight, tool_name, is_error). No command string parsing."""
    if tc.kind == "file_change":
        return "write", classifier.weight_for("write"), "file_change", False
    if tc.kind == "command_execution":
        # a shell command: classify by the classifier's tool rules (a long-wait rule may promote
        # it to `tool`); the classifier keys on the tool name, so present it as a Bash call.
        cat, w = classifier.classify("Bash", {"command": tc.command or ""})
        return cat, w, "command_execution", tc.is_error
    if tc.kind in ("mcp_tool_call", "web_search"):
        return "tool", classifier.weight_for("tool"), tc.kind, False
    return "bash", classifier.weight_for("bash"), tc.kind, False


def build_trajectory_from_run(
        run: CodexRun, *,
        run_id: str,
        classifier: ActivityClassifier,
        classifier_cfg: dict,
        model: str = "",
        snapshot: PriceSnapshot | None = None,
        billing_row: dict | None = None,
        calculated_at: str = "",
        source: str = "import:codex") -> RunTrajectory:
    """Assemble a :class:`RunTrajectory` from an already-normalized :class:`CodexRun` (pure)."""
    snapshot = snapshot or PriceSnapshot.default_openai()
    traj = RunTrajectory(run_id=run_id, source=source, model=model,
                         classifier_config=classifier_cfg)

    # ---- cost (whole-run), nullable + provenanced -------------------------------------------
    totals = run.totals()
    cost: CostRecord = cost_record_for(
        totals, snapshot=snapshot, model_requested=model, model_resolved=model,
        billing_row=billing_row, calculated_at=calculated_at)
    traj.cost = cost.to_dict()

    # a per-token unit cost we can spread across turns (None-safe): output+uncached weighted
    have_price = not cost.is_unpriced

    # ---- rounds + points (one per completed turn) -------------------------------------------
    cum_unc = cum_out = cum_cr = cum_cw = cum_reason = 0.0
    cum_cost = 0.0
    for i, tu in enumerate(run.turns):
        unc = tu.uncached_input_tokens or 0
        cr = tu.cached_input_tokens or 0
        cw = tu.cache_write_input_tokens or 0
        out = tu.output_tokens or 0
        reason = tu.reasoning_output_tokens or 0
        cum_unc += unc
        cum_out += out
        cum_cr += cr
        cum_cw += cw
        cum_reason += reason

        # per-turn cost from the snapshot (subset semantics), accumulated; 0 while unpriced
        turn_cost = 0.0
        if have_price:
            rec = cost_record_for(
                {"input_tokens": (tu.input_tokens or 0),
                 "uncached_input_tokens": unc,
                 "cached_input_tokens": cr,
                 "cache_write_input_tokens": cw,
                 "output_tokens": out},
                snapshot=snapshot, model_requested=model, model_resolved=model,
                billing_row=billing_row, calculated_at=calculated_at)
            turn_cost = rec.value_usd or 0.0
        cum_cost += turn_cost

        # real wall offset when the stream was timestamped (the live recorder); else pseudo-time
        if tu.t_s is not None:
            t_s = tu.t_s
        else:
            idx = tu.event_index if tu.event_index is not None else (i + 1)
            t_s = idx * _PSEUDO_DT_S
        traj.points.append(TrajectoryPoint(
            t_s=t_s,
            cum_input_tokens=cum_unc,
            cum_output_tokens=cum_out,
            cum_cache_read_tokens=cum_cr,
            cum_cache_creation_tokens=cum_cw,
            cum_cache_tokens=cum_cr + cum_cw,
            cum_reasoning_tokens=cum_reason,
            cum_cost_usd=round(cum_cost, 6),
            round_index=i,
            provisional_cost=not have_price,
        ))
        if i > 0:
            prev = run.turns[i - 1]
            t_start = prev.t_s if prev.t_s is not None else (
                (prev.event_index or 0) * _PSEUDO_DT_S)
        else:
            t_start = 0.0
        traj.rounds.append(RoundBoundary(
            index=i, t_start_s=t_start, t_end_s=t_s,
            cost_usd=round(turn_cost, 6),
            input_tokens=int(unc), output_tokens=int(out),
            cache_tokens=int(cr + cw), cache_read_tokens=int(cr),
            cache_creation_tokens=int(cw), reasoning_tokens=int(reason),
            session_id=run.thread_id or ""))

    # ---- activity bands / tool spans from structured items ----------------------------------
    for tc in run.tools:
        cat, w, tname, is_err = _tool_category(tc, classifier)
        start_idx = tc.started_index if tc.started_index is not None else tc.completed_index
        end_idx = tc.completed_index if tc.completed_index is not None else start_idx
        if start_idx is None:
            continue
        t0 = (tc.t_start_s if tc.t_start_s is not None else start_idx * _PSEUDO_DT_S)
        t1 = (tc.t_end_s if tc.t_end_s is not None else max(end_idx, start_idx) * _PSEUDO_DT_S)
        if t1 <= t0:
            t1 = t0 + _PSEUDO_DT_S
        traj.bands.append(ActivityBand(
            t0_s=t0, t1_s=t1, category=cat, tool_name=tname, weight=w,
            round_index=0, is_error=is_err))
        # a file_change is the agent's first written artifact → a `first_file` checkpoint
        if tc.kind == "file_change" and not any(c.kind == "first_file" for c in traj.checkpoints):
            traj.checkpoints.append(Checkpoint(
                t_s=t0, kind="first_file", scope="all", source="codex_file_change"))

    # ---- totals / axis ----------------------------------------------------------------------
    traj.final_input_tokens = int(cum_unc)
    traj.final_output_tokens = int(cum_out)
    traj.final_cache_read_tokens = int(cum_cr)
    traj.final_cache_creation_tokens = int(cum_cw)
    traj.final_cache_tokens = int(cum_cr + cum_cw)
    traj.final_reasoning_tokens = int(cum_reason)
    traj.final_cost_usd = cost.value_usd     # None when unpriced (NOT 0.0)
    traj.duration_s = traj.points[-1].t_s if traj.points else 0.0
    traj.num_rounds = len(run.turns)
    traj.provisional = False
    if model:
        traj.model = model
    return traj


def import_codex(raw: str | Path, *,
                 classifier_config: ActivityConfig | None = None,
                 circt: bool | None = None,
                 run_id: str = "",
                 label: str | None = None,
                 model: str = "gpt-5-codex",
                 price_snapshot: str | Path | None = None,
                 billing_mode: str | None = None,
                 provider: str = "openai",
                 calculated_at: str = "",
                 timestamped: str | Path | None = None,
                 milestone_time: str = "proportional",   # accepted for CLI uniformity; unused
                 **_ignored) -> RunTrajectory:
    """Import a Codex ``events.raw.jsonl`` (or a directory of them) into a :class:`RunTrajectory`.

    ``model`` is the requested model id (Codex does not echo it in the stream); pricing resolves it
    against the OpenAI price snapshot. Pass ``price_snapshot`` to pin a specific snapshot file;
    ``billing_mode`` (``per_token``/``subscription``) overrides the provider-derived classification.
    """
    files = _codex_files(raw)
    classifier, cfg_dict = _classifier(classifier_config, circt)
    rid = run_id or label or (Path(raw).stem if Path(raw).is_file() else Path(raw).name)

    # one normalizer across all files → a resume capture is a single continued thread
    run = _normalize(files, timestamped)

    snapshot = (PriceSnapshot.load(price_snapshot) if price_snapshot
                else PriceSnapshot.default_openai())
    billing_row: dict = {"provider": provider}
    if billing_mode:
        billing_row["billing_mode"] = billing_mode

    traj = build_trajectory_from_run(
        run, run_id=rid, classifier=classifier, classifier_cfg=cfg_dict, model=model,
        snapshot=snapshot, billing_row=billing_row, calculated_at=calculated_at)
    return traj


def import_codex_run(raw: str | Path, **kwargs) -> "tuple[RunTrajectory, CodexRun]":
    """Like :func:`import_codex` but also returns the raw :class:`CodexRun` (for ledgers/reconcile)."""
    files = _codex_files(raw)
    run = _normalize(files, kwargs.get("timestamped"))
    traj = import_codex(raw, **kwargs)
    return traj, run
