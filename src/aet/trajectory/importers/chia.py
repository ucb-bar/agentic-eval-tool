"""Import Chia's schema-versioned, privacy-safe profiler JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from aet.trajectory.model import (
    ActivityBand, InferenceRecord, RoundBoundary, RunTrajectory, TrajectoryPoint,
)


def _events(path: str | Path) -> list[dict]:
    source = Path(path)
    if source.is_dir():
        candidates = sorted(source.glob("**/*.jsonl")) + sorted(source.glob("**/*.log"))
    else:
        candidates = [source]
    out = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for line in candidate.read_text(errors="ignore").splitlines():
            try:
                event = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict) and event.get("schema") == "chia.agent_profile":
                out.append(event)
    return sorted(out, key=lambda event: float(event.get("ts", 0) or 0))


def import_chia(raw: str | Path, *, run_id: str = "", label: str | None = None,
                context_windows: dict[str, int] | None = None,
                default_cache_ttl_s: float | None = None, **_ignored) -> RunTrajectory:
    """Build an inference- and agent-aware trajectory from a Chia profile trace.

    ``context_windows`` and ``default_cache_ttl_s`` are explicit policies used
    only for derived occupancy/TTL signals. They never masquerade as provider
    measurements.
    """
    events = _events(raw)
    rid = run_id or label or Path(raw).stem
    traj = RunTrajectory(run_id=rid, source="import:chia")
    if not events:
        return traj
    # Event timestamps are completion times. Anchor at the earliest measured
    # start so the first request keeps its duration instead of being clipped at
    # trajectory t=0.
    origin = min(
        float(event.get("ts", 0) or 0) - max(0.0, float(event.get("duration_s", 0) or 0))
        for event in events
    )
    context_windows = context_windows or {}
    seen: set[tuple[str, int, str]] = set()

    for event in events:
        if event.get("type") != "llm_request":
            continue
        request_id = str(event.get("request_id") or "")
        attempt = int(event.get("attempt", 1) or 1)
        key = (request_id, attempt, str(event.get("span_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        end = max(0.0, float(event.get("ts", origin) or origin) - origin)
        duration = max(0.0, float(event.get("duration_s", 0) or 0))
        model = str(event.get("model") or "")
        window = event.get("context_window_tokens") or context_windows.get(model)
        fresh = int(event.get("input_tokens", 0) or 0)
        read = int(event.get("cache_read_tokens", 0) or 0)
        write = int(event.get("cache_write_tokens", 0) or 0)
        ttl = event.get("cache_ttl_s", default_cache_ttl_s)
        traj.inferences.append(InferenceRecord(
            request_id=request_id, t_start_s=max(0.0, end - duration), t_end_s=end,
            agent_id=str(event.get("agent_id") or ""),
            parent_agent_id=str(event.get("parent_agent_id") or ""),
            trace_id=str(event.get("trace_id") or ""), span_id=str(event.get("span_id") or ""),
            call_id=str(event.get("call_id") or ""), session_id=str(event.get("session_id") or ""),
            attempt=attempt, provider=str(event.get("provider") or ""), model=model,
            input_tokens=fresh, output_tokens=int(event.get("output_tokens", 0) or 0),
            cache_read_tokens=read, cache_write_tokens=write,
            reasoning_tokens=int(event.get("reasoning_tokens", 0) or 0),
            status=str(event.get("status") or "unknown"), retry=bool(event.get("retry", False)),
            cost_usd=(float(event["cost_usd"]) if event.get("cost_usd") is not None else None),
            cost_source=str(event.get("cost_source") or "unavailable"),
            billing_mode=str(event.get("billing_mode") or "per_token"),
            context_window_tokens=int(window) if window else None,
            estimated_context_tokens=fresh + read + write,
            cache_ttl_s=float(ttl) if ttl is not None else None,
        ))

    # Infer TTL loss only from the requested policy + an observed transition.
    prior_by_session: dict[str, InferenceRecord] = {}
    for rec in traj.inferences:
        key = rec.session_id or rec.agent_id or "unattributed"
        prior = prior_by_session.get(key)
        if (prior and rec.cache_ttl_s and prior.cache_read_tokens > 0
                and rec.cache_read_tokens == 0
                and rec.t_start_s - prior.t_end_s > rec.cache_ttl_s):
            rec.ttl_inference = "probable_expiry"
        elif rec.cache_ttl_s is not None:
            rec.ttl_inference = "no_expiry_signal"
        prior_by_session[key] = rec

    cin = cout = cr = cw = reason = cost = 0
    priced = True
    for index, rec in enumerate(traj.inferences):
        cin += rec.input_tokens
        cout += rec.output_tokens
        cr += rec.cache_read_tokens
        cw += rec.cache_write_tokens
        reason += rec.reasoning_tokens
        if rec.cost_usd is None:
            priced = False
        else:
            cost += rec.cost_usd
        traj.points.append(TrajectoryPoint(
            t_s=rec.t_end_s, cum_input_tokens=cin, cum_output_tokens=cout,
            cum_cache_tokens=cr + cw, cum_cache_read_tokens=cr,
            cum_cache_creation_tokens=cw, cum_reasoning_tokens=reason,
            cum_cost_usd=cost, round_index=index, provisional_cost=not priced,
        ))

    for event in events:
        if event.get("type") != "tool_activity":
            continue
        end = max(0.0, float(event.get("ts", origin) or origin) - origin)
        duration = max(0.0, float(event.get("duration_s", 0) or 0))
        traj.bands.append(ActivityBand(
            t0_s=max(0.0, end - duration), t1_s=end,
            category=str(event.get("category") or "tool"),
            tool_name=str(event.get("tool_name") or ""),
            is_error=event.get("status") == "failed",
        ))

    traj.duration_s = max((float(event.get("ts", origin) or origin) - origin for event in events),
                          default=0.0)
    traj.num_rounds = len(traj.inferences)
    traj.final_input_tokens = cin
    traj.final_output_tokens = cout
    traj.final_cache_read_tokens = cr
    traj.final_cache_creation_tokens = cw
    traj.final_cache_tokens = cr + cw
    traj.final_reasoning_tokens = reason
    traj.final_cost_usd = cost if priced else None
    if traj.inferences:
        traj.model = traj.inferences[-1].model
        traj.rounds = [RoundBoundary(
            index=0, t_start_s=0.0, t_end_s=traj.duration_s,
            cost_usd=(cost if priced else None),
            input_tokens=cin, output_tokens=cout, cache_tokens=cr + cw,
            cache_read_tokens=cr, cache_creation_tokens=cw, reasoning_tokens=reason,
        )]
    return traj
