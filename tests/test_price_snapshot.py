"""Versioned OpenAI price snapshot: load, hash, copy-into-experiment, and cost calculation."""
from aet.trajectory.price_snapshot import (
    PriceSnapshot, cost_record_for, DEFAULT_OPENAI_SNAPSHOT_ID,
)


def test_default_snapshot_loads_marked_unverified():
    snap = PriceSnapshot.default_openai()
    assert snap.price_table_id == DEFAULT_OPENAI_SNAPSHOT_ID
    # placeholder rates MUST advertise that they are unverified (gate on billed campaigns)
    assert snap.verified is False
    assert snap.pricing_url


def test_sha256_stable_and_content_addressed():
    a = PriceSnapshot.default_openai().sha256()
    b = PriceSnapshot.default_openai().sha256()
    assert a == b and len(a) == 64


def test_price_table_prices_codex_model():
    snap = PriceSnapshot.default_openai()
    table = snap.price_table()
    assert table.has_rate("gpt-5-codex")
    assert not table.has_rate("totally-unknown")   # no coarse fallback


def test_cost_record_applies_subset_semantics():
    snap = PriceSnapshot.default_openai()
    totals = {"input_tokens": 10000, "uncached_input_tokens": 5000,
              "cached_input_tokens": 4000, "cache_write_input_tokens": 1000,
              "output_tokens": 500}
    rec = cost_record_for(totals, snapshot=snap, model_requested="gpt-5-codex", calculated_at="")
    # 5000*1.25 + 4000*0.125 + 1000*0.0 + 500*10 all per Mtok
    expected = (5000 * 1.25 + 4000 * 0.125 + 1000 * 0.0 + 500 * 10.0) / 1_000_000.0
    assert abs(rec.value_usd - round(expected, 6)) < 1e-9
    assert rec.kind == "metered"
    assert rec.breakdown_usd["cache_write_usd"] == 0.0


def test_cost_record_unpriced_when_no_rate():
    snap = PriceSnapshot.default_openai()
    rec = cost_record_for({"input_tokens": 100, "uncached_input_tokens": 100},
                          snapshot=snap, model_requested="mystery", calculated_at="")
    assert rec.is_unpriced and rec.value_usd is None


def test_cost_record_unpriced_when_input_missing():
    snap = PriceSnapshot.default_openai()
    rec = cost_record_for({"input_tokens": None}, snapshot=snap,
                          model_requested="gpt-5-codex", calculated_at="")
    assert rec.is_unpriced


def test_copy_into_experiment_preserves_bytes(tmp_path):
    snap = PriceSnapshot.default_openai()
    dest = snap.copy_into_experiment(tmp_path)
    assert dest == tmp_path / "config" / "price-table.yaml"
    # the copied snapshot hashes identically to the source (reproducibility)
    assert PriceSnapshot.load(dest).sha256() == snap.sha256()


def test_subscription_billing_row_makes_notional():
    snap = PriceSnapshot.default_openai()
    rec = cost_record_for({"input_tokens": 100, "uncached_input_tokens": 100,
                           "output_tokens": 10}, snapshot=snap,
                          model_requested="gpt-5-codex",
                          billing_row={"provider": "openai", "billing_mode": "subscription"},
                          calculated_at="")
    assert rec.kind == "subscription_notional"
