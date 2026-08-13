"""A reported cost is not necessarily money.

The Claude Code CLI emits `total_cost_usd` on a subscription seat too, where it is what those tokens
would have cost on the API and no card is charged. `aet` read that field as money with no notion of
which kind of run produced it, so summing a subscription run and a metered Bedrock run and calling
the total "spend" presented quota consumption as money — and a budget cap enforced against that sum
is enforced against a number partly nobody is paying.
"""

from aet.trajectory.billing import Spend, billing_mode_of, is_metered, split


def test_a_declared_mode_wins_over_the_inference():
    """Whoever launched the run knows how it is billed; an inference overriding them would be wrong
    in exactly the cases someone bothered to state."""
    assert billing_mode_of({"billing_mode": "subscription", "provider": "aet:bedrock"}) == "subscription"
    assert billing_mode_of({"billing_mode": "per_token", "provider": "whatever"}) == "per_token"


def test_an_unrecognised_declared_mode_is_refused():
    """Silently treating it as a seat would make the spend vanish from the ledger."""
    import pytest
    with pytest.raises(ValueError, match="outside"):
        billing_mode_of({"billing_mode": "free"})


def test_a_metered_provider_classifies_by_family():
    assert is_metered({"provider": "aet:bedrock:opus"})
    assert is_metered({"provider": "chia:vertex"})


def test_an_unclassified_provider_is_a_seat_not_a_charge():
    """The conservative direction. Counting an unknown provider as metered would inflate reported
    spend and consume a budget cap no card is backing."""
    assert not is_metered({"provider": "some-new-launcher"})
    assert not is_metered({})


def test_the_two_halves_are_never_added():
    s = split([
        {"provider": "aet:bedrock:opus", "cost_usd": 1.50},
        {"provider": "cli-subscription", "cost_usd": 9.99},
    ])
    assert s.metered_usd == 1.50
    assert s.subscription_usd == 9.99
    assert s.against_budget == 1.50, "a cap governs money, not notional quota value"
    assert "not money, not summed" in s.describe()


def test_an_unpriced_row_is_counted_not_zeroed():
    """An unpriced row and a free row are different facts — the same reason `PriceTable` refuses to
    guess a rate for an unknown model."""
    s = Spend()
    s.add({"provider": "aet:bedrock"}, None)
    assert s.unpriced_rows == 1
    assert s.metered_usd == 0.0 and s.metered_rows == 0


def test_the_stream_result_carries_its_mode():
    """The field has to reach the object a ledger reads, or the distinction stops at the parser."""
    from aet.tracking.claude_stream import ClaudeStreamResult
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(ClaudeStreamResult)}
    assert "billing_mode" in fields
    assert fields["billing_mode"].default == "subscription"


def test_a_transcript_with_no_result_event_still_produces_a_mode():
    """A killed or rate-limited run ends without a terminal `result` event. A name bound only on the
    happy path would raise NameError on exactly those runs."""
    from aet.tracking.claude_stream import parse_stream
    res = parse_stream('{"type":"system","subtype":"init","session_id":"s"}')
    assert res.billing_mode == "subscription"
    assert not res.has_result_event
