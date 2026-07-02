"""Rate-limit detection: five-hour vs weekly vs a normal (worked) invocation, and wake timing."""
import json

from aet.ratelimit import (
    RateLimitState, parse_rate_limit, rate_limit_from_transcript, seconds_until_reset,
)


def _rl_event(limit_type, resets_at):
    return {"type": "rate_limit_event",
            "rate_limit_info": {"status": "rejected", "rateLimitType": limit_type,
                                "resetsAt": resets_at}}


def _asst_with_tool():
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]}}


def test_five_hour_rejection_no_work():
    events = [{"type": "system", "session_id": "s"}, _rl_event("five_hour", 1_900_000_000)]
    st = parse_rate_limit(events)
    assert st.rejected and st.is_five_hour and not st.is_weekly
    assert st.resets_at == 1_900_000_000


def test_weekly_rejection_detected():
    for lt in ("weekly", "seven_day"):
        st = parse_rate_limit([_rl_event(lt, 1_900_000_000)])
        assert st.rejected and st.is_weekly and not st.is_five_hour


def test_rejection_with_tool_work_is_not_burned():
    # a rejection appeared, but the invocation still did real work → not a burned attempt
    st = parse_rate_limit([_asst_with_tool(), _rl_event("five_hour", 1)])
    assert st.saw_rejection and not st.rejected


def test_result_session_limit_error_counts():
    st = parse_rate_limit([{"type": "result", "is_error": True,
                            "result": "You have hit your session limit."}])
    assert st.rejected and st.limit_type == "five_hour"


def test_normal_run_is_not_rejected():
    st = parse_rate_limit([_asst_with_tool(),
                           {"type": "result", "subtype": "success", "total_cost_usd": 0.1}])
    assert not st.rejected and not st.saw_rejection


def test_from_transcript_file(tmp_path):
    p = tmp_path / "round_00.transcript.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in [_rl_event("five_hour", 42)]) + "\n")
    st = rate_limit_from_transcript(p)
    assert st.rejected and st.resets_at == 42


def test_seconds_until_reset_and_stale():
    st = RateLimitState(rejected=True, limit_type="five_hour", resets_at=1000)
    assert seconds_until_reset(st, now=900, jitter=30) == 130      # (1000+30) - 900
    assert seconds_until_reset(st, now=1100) is None               # reset+jitter already passed → poll
    assert seconds_until_reset(RateLimitState(), now=0) is None    # no epoch → poll
