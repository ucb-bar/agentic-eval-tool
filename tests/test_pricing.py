"""Provider-aware price table: Claude + Nova price correctly, unknown models are cost-unavailable
(not silently Opus-priced), and an override map / env var supplies a rate without a code change."""
import json

import pytest

from aet.trajectory.pricing import KNOWN_UNPRICED, PRICE_TABLE_ENV, PriceTable


def test_claude_rates_unchanged():
    pt = PriceTable()
    # opus list price: 15 in / 75 out / 1.5 cache-read / 18.75 cache-create per Mtok
    got = pt.estimate_usd(1_000_000, 1_000_000, 1_000_000, 1_000_000, model="claude-opus-4-8")
    assert got == pytest.approx(15.0 + 75.0 + 1.5 + 18.75)
    # message_cost_shape (the weight) also unchanged for claude
    shape = pt.message_cost_shape(1_000_000, 0, 0, 0, model="claude-sonnet-4")
    assert shape == pytest.approx(3.0)


def test_bedrock_version_specific_rates_reproduce_known_costs():
    """opus-4-6 / sonnet-4-6 / haiku-4-5 are billed at a distinct Bedrock tier from the classic
    Claude list price; the version-specific keys must win over the coarse ones and reproduce the
    exact costUSD Claude Code reports for these ids (verified from `modelUsage.costUSD`)."""
    pt = PriceTable()
    # opus-4-6: (5, 25, 0.5, 6.25) — NOT classic Opus 15/75
    assert pt.estimate_usd(386, 130, 0, 32291,
                           model="us.anthropic.claude-opus-4-6-v1") == pytest.approx(
        0.20699875, abs=1e-6)
    # sonnet-4-6: (3, 15, 0.3, 3.75)
    assert pt.estimate_usd(386, 81, 0, 31722,
                           model="us.anthropic.claude-sonnet-4-6") == pytest.approx(
        0.1213305, abs=1e-6)
    # haiku-4-5: (1, 5, 0.1, 1.25) — NOT haiku-3.5's 0.80/4
    assert pt.estimate_usd(10, 110, 0, 32298,
                           model="us.anthropic.claude-haiku-4-5-20251001-v1:0") == pytest.approx(
        0.0409325, abs=1e-6)


def test_opus_4_6_not_classic_opus_rate():
    """The versioned opus-4-6 id must resolve to the 5/25/0.5/6.25 tier, not classic 15/75."""
    pt = PriceTable()
    # one input Mtok at the classic rate would be $15; at the 4-6 tier it is $5
    got = pt.estimate_usd(1_000_000, 0, 0, 0, model="us.anthropic.claude-opus-4-6-v1")
    assert got == pytest.approx(5.0)
    assert got != pytest.approx(15.0)
    # full 1-Mtok-each sanity vs the classic-opus total (15+75+1.5+18.75 = 110.25)
    full = pt.estimate_usd(1_000_000, 1_000_000, 1_000_000, 1_000_000,
                           model="us.anthropic.claude-opus-4-6-v1")
    assert full == pytest.approx(5.0 + 25.0 + 0.5 + 6.25)
    assert full != pytest.approx(110.25)
    # a classic (unversioned / 4.1) opus id still uses the 15/75 generic
    assert pt.estimate_usd(1_000_000, 0, 0, 0,
                           model="us.anthropic.claude-opus-4-1") == pytest.approx(15.0)


def test_haiku_4_5_not_haiku_3_5_rate():
    """The versioned haiku-4-5 id resolves to 1/5/0.1/1.25, not haiku-3.5's 0.80/4/0.08/1.0."""
    pt = PriceTable()
    got = pt.estimate_usd(1_000_000, 0, 0, 0,
                          model="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert got == pytest.approx(1.0)
    assert got != pytest.approx(0.80)
    # older haiku still uses the generic 0.80 rate
    assert pt.estimate_usd(1_000_000, 0, 0, 0,
                           model="anthropic.claude-3-5-haiku") == pytest.approx(0.80)


def test_nova_rates_priced():
    pt = PriceTable()
    # input+output list prices verified from AWS Bedrock pricing
    assert pt.estimate_usd(1_000_000, 0, 0, 0, model="amazon.nova-micro-v1") == pytest.approx(0.035)
    assert pt.estimate_usd(0, 1_000_000, 0, 0, model="amazon.nova-lite-v1") == pytest.approx(0.24)
    assert pt.estimate_usd(1_000_000, 0, 0, 0, model="us.amazon.nova-pro-v1") == pytest.approx(0.80)


def test_unknown_model_is_cost_unavailable_not_opus():
    pt = PriceTable()
    for model in ("nemotron-4", "moonshotai/kimi-k2", "zai-glm-4.7", "deepseek-v3",
                  "qwen2.5-72b", "meta.llama3-70b", "mistral.large", "cohere.command-r"):
        assert pt.estimate_usd(1_000_000, 1_000_000, 0, 0, model=model) is None, model
        # weight degrades to zero (no fabricated shape), never Opus pricing
        assert pt.message_cost_shape(1_000_000, 0, 0, 0, model=model) == 0.0
    assert not pt.has_rate("nemotron-4")


def test_known_unpriced_documents_intent():
    # membership is documentation; every named family resolves to cost-unavailable
    pt = PriceTable()
    for key in KNOWN_UNPRICED:
        assert pt.estimate_usd(1_000, 1_000, 0, 0, model=f"vendor.{key}-x") is None


def test_explicit_fallback_still_available():
    # opt-in Opus fallback for callers that accept a coarse guess
    pt = PriceTable(fallback="opus")
    assert pt.estimate_usd(1_000_000, 0, 0, 0, model="some-unknown-model") == pytest.approx(15.0)


def test_override_map_supplies_price(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({
        "deepseek": [0.27, 1.10, 0.07, 0.27],
        "kimi": {"input": 0.60, "output": 2.50},
    }))
    pt = PriceTable.load(path)
    assert pt.estimate_usd(1_000_000, 0, 0, 0, model="deepseek-v3") == pytest.approx(0.27)
    assert pt.estimate_usd(0, 1_000_000, 0, 0, model="moonshot/kimi-k2") == pytest.approx(2.50)
    # built-in claude rates still present after a merge
    assert pt.estimate_usd(1_000_000, 0, 0, 0, model="claude-opus-4-8") == pytest.approx(15.0)


def test_from_env_reads_override(tmp_path, monkeypatch):
    path = tmp_path / "prices.yaml"
    path.write_text("qwen: [0.10, 0.30]\n")
    monkeypatch.setenv(PRICE_TABLE_ENV, str(path))
    pt = PriceTable.from_env()
    assert pt.estimate_usd(1_000_000, 0, 0, 0, model="qwen2.5-72b") == pytest.approx(0.10)
    # unset env => plain default table, unknown still unavailable
    monkeypatch.delenv(PRICE_TABLE_ENV)
    assert PriceTable.from_env().estimate_usd(1, 1, 0, 0, model="qwen2.5") is None
