"""TrajectoryStream: incremental build, provisional-until-result, tail a file."""
import json

from aet.trajectory.classify import ActivityClassifier, capsule_bench_config
from aet.trajectory.stream import TrajectoryStream
from tests.test_trajectory_build import _round_events


def _lines():
    """The synthetic round from the build test, as raw json lines with ISO-less monotonic ts."""
    return [(ts, line) for ts, line in _round_events(session="s", t0=0.0, cost=0.02)]


def test_provisional_until_result_event():
    stream = TrajectoryStream(classifier=ActivityClassifier(capsule_bench_config()))
    events = _lines()
    result_line = events[-1]
    for ts, line in events[:-1]:      # everything except the terminal result event
        stream.feed_line(line, ts=ts)
    mid = stream.snapshot()
    assert mid.provisional is True
    assert all(p.provisional_cost for p in mid.points)
    prov_cost = mid.final_cost_usd
    assert prov_cost > 0.0            # list-price estimate before the bill

    stream.feed_line(result_line[1], ts=result_line[0])
    final = stream.snapshot()
    assert final.provisional is False
    assert not any(p.provisional_cost for p in final.points)
    assert abs(final.final_cost_usd - 0.02) < 1e-9   # flips to the billed number


def test_cumulative_tokens_nondecreasing_across_feeds():
    stream = TrajectoryStream(classifier=ActivityClassifier(capsule_bench_config()))
    totals = []
    for ts, line in _lines():
        stream.feed_line(line, ts=ts)
        totals.append(stream.snapshot().final_input_tokens
                      + stream.snapshot().final_output_tokens
                      + stream.snapshot().final_cache_tokens)
    assert all(totals[i] >= totals[i - 1] for i in range(1, len(totals)))


def test_on_update_callback_fires():
    seen = []
    stream = TrajectoryStream(classifier=ActivityClassifier(), on_update=lambda t: seen.append(t),
                              flush_every=1)
    for ts, line in _lines():
        stream.feed_line(line, ts=ts)
    assert len(seen) >= 3
    assert seen[-1].num_rounds == 1


def test_attach_file_no_follow(tmp_path):
    p = tmp_path / "round_00.transcript.jsonl"
    p.write_text("\n".join(line for _, line in _lines()) + "\n")
    stream = TrajectoryStream(classifier=ActivityClassifier(capsule_bench_config()))
    traj = stream.attach_file(p, follow=False)
    assert traj.num_rounds == 1
    assert traj.provisional is False           # the file already contains the result event
    assert any(b.category == "tool" for b in traj.bands)
