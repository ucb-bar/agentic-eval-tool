"""Build a :class:`RunTrajectory` from parsed Claude stream results — one round at a time.

``append_round`` extends a trajectory with a single agent invocation, concatenating rounds on
one *active-wall* time axis (round k starts where round k-1 ended). Token/cost points come from
``turn_usage``; activity bands from ``turn_usage`` + ``tool_calls`` via the pluggable classifier.

Timing: this uses the **real per-event offsets** recovered by
``claude_stream.parse_timestamped_stream`` (``TurnUsage.start_offset_s``, ``ToolCall.turn_index``)
— strictly better than the old within-round weighting. Fed a plain ``parse_stream`` result (no
timestamps) the token curves are still exact; only the band timing degrades.

Cost: the round's authoritative ``total_cost_usd`` is spread across interior points by list-price
*shape* (so the curve rises smoothly yet sums to the real bill at the round end). When a stream
carries no billed number yet (mid-flight), the shape doubles as a provisional estimate and the
points are flagged ``provisional_cost``.
"""
from __future__ import annotations

from aet.tracking.claude_stream import ClaudeStreamResult, TurnUsage
from aet.trajectory.classify import ActivityClassifier
from aet.trajectory.model import ActivityBand, RoundBoundary, RunTrajectory, TrajectoryPoint
from aet.trajectory.pricing import PriceTable


def _round_duration_s(result: ClaudeStreamResult, turns: list[TurnUsage]) -> float:
    """Active wall span of one round — the max of every clock we can see."""
    stream_dur = float(getattr(result, "_stream_duration_s", 0.0) or 0.0)
    max_turn = max((t.start_offset_s for t in turns), default=0.0)
    max_tool = max((tc.start_offset_s + tc.duration_s for tc in result.tool_calls), default=0.0)
    return max(stream_dur, result.duration_ms / 1000.0, max_turn, max_tool)


def bands_from_result(result: ClaudeStreamResult, classifier: ActivityClassifier, *,
                      t_offset_s: float, round_index: int, round_duration_s: float,
                      ) -> list[ActivityBand]:
    """One band per assistant turn, contiguous to the next turn (last → round end).

    A turn's category follows the reference precedence: its (primary) tool call wins, else
    ``think`` if it carried an extended-thinking block / planning text, else ``bash``. Because a
    band runs until the *next* turn starts, a long tool-wait (e.g. a simulator) is captured by
    its real elapsed time — no weighting needed.
    """
    turns = sorted(result.turn_usage, key=lambda t: (t.start_offset_s, t.turn))
    tools_by_turn: dict[int, list] = {}
    for tc in result.tool_calls:
        tools_by_turn.setdefault(tc.turn_index, []).append(tc)

    bands: list[ActivityBand] = []
    for i, turn in enumerate(turns):
        start = t_offset_s + turn.start_offset_s
        nxt = turns[i + 1].start_offset_s if i + 1 < len(turns) else round_duration_s
        end = t_offset_s + max(nxt, turn.start_offset_s)
        tcs = tools_by_turn.get(i + 1, [])   # parse_timestamped_stream turn_index is 1-based
        if tcs:
            tc = tcs[0]
            cat, w = classifier.classify(tc.name, tc.input)
            bands.append(ActivityBand(start, end, cat, tool_name=tc.name, weight=w,
                                      round_index=round_index, is_error=tc.is_error))
        elif getattr(turn, "has_thinking", False):
            bands.append(ActivityBand(start, end, "think",
                                      weight=classifier.weight_for("think"),
                                      round_index=round_index))
        else:
            bands.append(ActivityBand(start, end, "bash",
                                      weight=classifier.weight_for("bash"),
                                      round_index=round_index))
    return bands


def append_round(traj: RunTrajectory, result: ClaudeStreamResult, *,
                 classifier: ActivityClassifier,
                 verdict: dict | None = None,
                 price_table: PriceTable | None = None,
                 round_index: int | None = None) -> RunTrajectory:
    """Extend ``traj`` in place with one agent round; returns it for chaining."""
    price_table = price_table or PriceTable()
    ri = round_index if round_index is not None else traj.num_rounds
    t0 = traj.duration_s
    base_in = traj.final_input_tokens
    base_out = traj.final_output_tokens
    base_cache_read = traj.final_cache_read_tokens
    base_cache_creation = traj.final_cache_creation_tokens
    base_cost = traj.final_cost_usd or 0.0   # None (unpriced) contributes nothing to the running sum

    turns = sorted(result.turn_usage, key=lambda t: (t.start_offset_s, t.turn))
    round_dur = _round_duration_s(result, turns)

    # cost shape per turn → spread the billed total (or a provisional estimate) smoothly
    shapes = [price_table.message_cost_shape(
                  t.input_tokens, t.output_tokens,
                  t.cache_read_input_tokens, t.cache_creation_input_tokens,
                  t.model or result.model)
              for t in turns]
    ssum = sum(shapes) or 1.0
    billed = float(result.cost_usd or 0.0)
    provisional = billed <= 0.0
    round_cost_total = billed if not provisional else sum(shapes)

    acc_in = acc_out = acc_cache_read = acc_cache_creation = acc_shape = 0.0
    for i, t in enumerate(turns):
        acc_in += t.input_tokens
        acc_out += t.output_tokens
        acc_cache_read += t.cache_read_input_tokens
        acc_cache_creation += t.cache_creation_input_tokens
        acc_shape += shapes[i]
        cum_cost = base_cost + round_cost_total * (acc_shape / ssum)
        cum_cache_read = base_cache_read + acc_cache_read
        cum_cache_creation = base_cache_creation + acc_cache_creation
        traj.points.append(TrajectoryPoint(
            t_s=t0 + t.start_offset_s,
            cum_input_tokens=base_in + acc_in,
            cum_output_tokens=base_out + acc_out,
            cum_cache_read_tokens=cum_cache_read,
            cum_cache_creation_tokens=cum_cache_creation,
            cum_cache_tokens=cum_cache_read + cum_cache_creation,
            cum_cost_usd=round(cum_cost, 6),
            round_index=ri,
            provisional_cost=provisional,
        ))
    acc_cache = acc_cache_read + acc_cache_creation

    traj.bands.extend(bands_from_result(
        result, classifier, t_offset_s=t0, round_index=ri, round_duration_s=round_dur))

    n_passed = n_total = None
    if verdict:
        n_passed = verdict.get("n_passed")
        n_total = verdict.get("n_total", verdict.get("n_capsules"))
    traj.rounds.append(RoundBoundary(
        index=ri, t_start_s=t0, t_end_s=t0 + round_dur,
        cost_usd=billed, input_tokens=int(acc_in), output_tokens=int(acc_out),
        cache_tokens=int(acc_cache), n_passed=n_passed, n_total=n_total,
        session_id=result.session_id,
    ))

    # advance running totals / axis
    traj.final_input_tokens = int(base_in + acc_in)
    traj.final_output_tokens = int(base_out + acc_out)
    traj.final_cache_read_tokens = int(base_cache_read + acc_cache_read)
    traj.final_cache_creation_tokens = int(base_cache_creation + acc_cache_creation)
    traj.final_cache_tokens = int(traj.final_cache_read_tokens + traj.final_cache_creation_tokens)
    traj.final_cost_usd = round(base_cost + round_cost_total, 6)
    traj.duration_s = t0 + round_dur
    traj.num_rounds = max(traj.num_rounds, ri + 1)
    traj.provisional = provisional
    if result.model:
        traj.model = result.model
    return traj
