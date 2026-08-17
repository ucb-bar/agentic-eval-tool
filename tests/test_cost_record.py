"""CostRecord invariants: unpriced (None) is never a fabricated $0; provenance round-trips."""
import pytest

from aet.trajectory.cost import CostRecord


def test_unpriced_must_be_none():
    r = CostRecord.unpriced(model_requested="m", price_table_id="pt")
    assert r.value_usd is None
    assert r.kind == "unpriced"
    assert r.is_unpriced and not r.is_money


def test_unpriced_with_value_rejected():
    with pytest.raises(ValueError):
        CostRecord(value_usd=1.0, kind="unpriced")


def test_priced_without_value_rejected():
    with pytest.raises(ValueError):
        CostRecord(value_usd=None, kind="metered")


def test_free_is_distinct_from_unpriced():
    free = CostRecord(value_usd=0.0, kind="metered", source="provider_billed")
    assert free.value_usd == 0.0 and not free.is_unpriced and free.is_money


def test_bad_kind_or_source_rejected():
    with pytest.raises(ValueError):
        CostRecord(value_usd=1.0, kind="bogus")
    with pytest.raises(ValueError):
        CostRecord(value_usd=1.0, kind="metered", source="bogus")


def test_roundtrip():
    r = CostRecord(value_usd=0.5, kind="metered", source="price_calculated",
                   model_requested="gpt-5-codex", price_table_id="openai-2026-08-17",
                   breakdown_usd={"output_usd": 0.5})
    assert CostRecord.from_dict(r.to_dict()).to_dict() == r.to_dict()
    assert CostRecord.from_dict(None) is None


def test_subscription_notional_is_not_money():
    r = CostRecord(value_usd=1.23, kind="subscription_notional", source="price_calculated")
    assert not r.is_money
