"""Emit a :class:`RunTrajectory` to canonical logs, and reconstruct one back.

Native aet runs record the trajectory through the ordinary :class:`EvalRunLogger` primitives, so
it lives in the same ``logs/`` a run already produces — plus a ``metrics/trajectory.json``
fast-path artifact. ``materialize_run`` turns an imported trajectory into a full aet run dir so
existing runs become indistinguishable from native ones (``aet runs``/``aet show``/``aet plot``).
"""
from __future__ import annotations

import json
from pathlib import Path

from aet.trajectory.model import (
    RunTrajectory, TrajectoryPoint, TestMilestone, RoundBoundary,
)


# --------------------------------------------------------------------- emit
def emit_trajectory(traj: RunTrajectory, logger, run_path: str | Path) -> Path:
    """Record ``traj`` through ``logger`` (points/milestones/rounds) + write the fast-path JSON."""
    run_path = Path(run_path)
    logger.log_param("aet.traj.classifier_config", traj.classifier_config)
    logger.log_param("aet.traj.summary", {
        "run_id": traj.run_id, "source": traj.source, "model": traj.model,
        "duration_s": traj.duration_s, "num_rounds": traj.num_rounds,
        "final_cost_usd": traj.final_cost_usd,
        "final_input_tokens": traj.final_input_tokens,
        "final_output_tokens": traj.final_output_tokens,
        "final_cache_tokens": traj.final_cache_tokens,
    })
    for rb in traj.rounds:
        logger.log_round_boundary(rb)
    for i, p in enumerate(traj.points):
        logger.log_trajectory_point(
            i, p.t_s, p.cum_input_tokens, p.cum_output_tokens, p.cum_cache_tokens,
            p.cum_cost_usd, p.round_index, p.provisional_cost)
    for m in traj.milestones:
        logger.log_test_milestone(m.t_s, m.n_passed, m.n_total, m.round_index, m.scope)

    artifact = traj.to_json(run_path / "metrics" / "trajectory.json")
    logger.log_artifact(artifact, "trajectory.json")
    return artifact


# --------------------------------------------------------------------- reconstruct
def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def trajectory_from_logs(run_path: str | Path) -> RunTrajectory:
    """Rebuild a trajectory from canonical ``logs/`` when the fast-path artifact is absent.

    Points come from the ``aet.traj.*`` step-metric families; rounds/milestones from
    ``aet.traj.round``/``aet.traj.milestone`` events; totals from the ``aet.traj.summary`` param.
    Activity bands are not stored in logs (they belong to tool events) and come back empty here —
    prefer ``metrics/trajectory.json`` (via :meth:`RunTrajectory.from_run_dir`) for full fidelity.
    """
    run_path = Path(run_path)
    logs = run_path / "logs"

    # points: group step-metric rows by step
    by_step: dict[int, dict] = {}
    for row in _read_jsonl(logs / "metrics.jsonl"):
        name = row.get("name", "")
        if not name.startswith("aet.traj."):
            continue
        step = row.get("step")
        if step is None:
            continue
        by_step.setdefault(int(step), {})[name] = row.get("value")
    points: list[TrajectoryPoint] = []
    for step in sorted(by_step):
        d = by_step[step]
        if "aet.traj.cum_total_tokens" not in d and "aet.traj.cum_input_tokens" not in d:
            continue  # a non-point metric family (e.g. tests_passed) — skip
        points.append(TrajectoryPoint(
            t_s=float(d.get("aet.traj.t_s", 0.0) or 0.0),
            cum_input_tokens=float(d.get("aet.traj.cum_input_tokens", 0.0) or 0.0),
            cum_output_tokens=float(d.get("aet.traj.cum_output_tokens", 0.0) or 0.0),
            cum_cache_tokens=float(d.get("aet.traj.cum_cache_tokens", 0.0) or 0.0),
            cum_cost_usd=float(d.get("aet.traj.cum_cost_usd", 0.0) or 0.0),
            round_index=int(d.get("aet.traj.round_index", 0) or 0),
            provisional_cost=bool(d.get("aet.traj.provisional_cost", 0.0)),
        ))

    rounds: list[RoundBoundary] = []
    milestones: list[TestMilestone] = []
    for ev in _read_jsonl(logs / "events.jsonl"):
        name = ev.get("event", "")
        p = ev.get("payload", {}) or {}
        if name == "aet.traj.round":
            rounds.append(RoundBoundary(**{k: p.get(k) for k in (
                "index", "t_start_s", "t_end_s", "cost_usd", "input_tokens",
                "output_tokens", "cache_tokens", "n_passed", "n_total", "session_id")
                if k in p}))
        elif name == "aet.traj.milestone":
            milestones.append(TestMilestone(
                t_s=float(p.get("t_s", 0.0) or 0.0),
                n_passed=int(p.get("n_passed", 0) or 0),
                n_total=int(p.get("n_total", 0) or 0),
                scope=p.get("scope", "all"),
                round_index=p.get("round_index"),
                source="logs"))

    # summary param → totals/metadata
    summary = {}
    params_path = logs / "params.json"
    if params_path.is_file():
        try:
            summary = (json.loads(params_path.read_text()) or {}).get("aet.traj.summary", {}) or {}
        except Exception:
            summary = {}

    traj = RunTrajectory(
        run_id=summary.get("run_id", run_path.name),
        source=summary.get("source", "logs"),
        model=summary.get("model", ""),
        duration_s=float(summary.get("duration_s", rounds[-1].t_end_s if rounds else 0.0)),
        num_rounds=int(summary.get("num_rounds", len(rounds))),
        points=points, rounds=rounds, milestones=milestones,
        final_cost_usd=float(summary.get("final_cost_usd",
                                         points[-1].cum_cost_usd if points else 0.0)),
        final_input_tokens=int(summary.get("final_input_tokens",
                                           points[-1].cum_input_tokens if points else 0)),
        final_output_tokens=int(summary.get("final_output_tokens",
                                            points[-1].cum_output_tokens if points else 0)),
        final_cache_tokens=int(summary.get("final_cache_tokens",
                                           points[-1].cum_cache_tokens if points else 0)),
    )
    return traj


# --------------------------------------------------------------------- materialize
def materialize_run(traj: RunTrajectory, into_dir: str | Path) -> Path:
    """Create a minimal canonical aet run dir from a trajectory (used by ``aet import --into``)."""
    from aet.core.run_spec import RunSpec
    from aet.core.run_manifest import RunManifest
    from aet.tracking.run_logger import EvalRunLogger

    run_path = Path(into_dir)
    (run_path / "logs").mkdir(parents=True, exist_ok=True)
    (run_path / "metrics").mkdir(parents=True, exist_ok=True)
    run_id = traj.run_id or run_path.name

    # the generic "default" suite so the run is queryable via `aet runs`/`aet compare`
    spec = RunSpec(project="aet", suite="default", method=traj.source or "import",
                   seed=0, run_id=run_id, project_root=run_path.parent,
                   target=traj.model or "unknown", model=traj.model or None,
                   is_smoke_test=False)
    RunManifest.create(spec, run_id, git_hash="imported").dump(run_path / "run_manifest.yaml")

    logger = EvalRunLogger.start(
        project=spec.project, suite=spec.suite, target=spec.target or "",
        method=spec.method, seed=spec.seed, run_id=run_id, run_path=run_path,
        tracking_mode="local")
    emit_trajectory(traj, logger, run_path)
    return run_path
