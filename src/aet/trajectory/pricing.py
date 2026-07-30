"""Token price tables — used ONLY for provisional cost while streaming, and as the per-message
cost *shape* when distributing a round's billed total across its messages.

Authoritative cost always comes from the CLI's ``total_cost_usd`` at a round's result event. A
live monitor, though, has no billed number until that event lands, so it estimates from list
prices; and even in batch we spread the billed round total across interior points using
list-price weights (so the curve rises smoothly yet still sums to the real bill at round ends).

Rates are USD per **million** tokens. Cache-creation (write) is charged above input; cache-read
is charged far below. The built-in keys are matched as case-insensitive substrings of the model
id, so they also match the equivalent AWS Bedrock model ids (Claude keys match ``anthropic.*``;
the Nova keys match ``amazon.nova-*``).

Honesty policy — no invented prices. A model that matches neither a built-in rate, an
explicit override, nor an explicitly-configured ``fallback`` has **cost unavailable**:
:meth:`PriceTable.estimate_usd` returns ``None`` for it rather than silently borrowing Opus
pricing. Models we know are billable but whose real Bedrock/region list price we cannot verify
here are named in :data:`KNOWN_UNPRICED` (documentation of intent — they resolve to
cost-unavailable exactly like any other unknown key). Supply their real rates per deployment via
an override map (see :meth:`PriceTable.from_env` / :meth:`PriceTable.load`).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Environment variable naming a JSON/YAML override price map (see PriceTable.from_env).
PRICE_TABLE_ENV = "AET_PRICE_TABLE"

# USD / Mtok: (input, output, cache_read, cache_creation)
_DEFAULT_RATES: dict[str, tuple[float, float, float, float]] = {
    # Version-specific Bedrock rates, empirically verified by reverse-engineering Claude Code's
    # authoritative `modelUsage.costUSD` (each follows Anthropic's output=5×input,
    # cache_read=0.1×input, cache_write=1.25×input multipliers). These families are billed at a
    # different tier than the classic Claude list price, so they must match BEFORE the coarse
    # `opus`/`sonnet`/`haiku` keys — see `_rate`, which resolves longest-key-wins. Keys are
    # substrings of ids like `us.anthropic.claude-opus-4-6-v1` / `...haiku-4-5-20251001-v1:0`.
    "opus-4-6":   (5.0, 25.0, 0.5, 6.25),   # NOT classic Opus 15/75
    "sonnet-4-6": (3.0, 15.0, 0.3, 3.75),
    "haiku-4-5":  (1.0, 5.0, 0.1, 1.25),    # NOT haiku-3.5's 0.80/4
    # Anthropic Claude list price (matches the oscar-merlin reference PR_IN/OUT/CR/CW). These
    # substrings also match Bedrock's `anthropic.claude-*` / `us.anthropic.*` model ids. Kept as
    # generic fallbacks for versions without a specific entry (classic Opus 4/4.1 really is 15/75).
    "opus":   (15.0, 75.0, 1.5, 18.75),
    "sonnet": (3.0, 15.0, 0.3, 3.75),
    "haiku":  (0.80, 4.0, 0.08, 1.0),
    # Amazon Nova on-demand list price (input, output verified from AWS Bedrock pricing).
    # Nova's cache rates are not separately published here, so — stated honestly — cache_read is
    # approximated as input*0.25 and cache_creation as input (i.e. a cache write ≈ one input pass).
    # Override via AET_PRICE_TABLE with real cache rates where a deployment has them.
    "nova-micro": (0.035, 0.14, 0.035 * 0.25, 0.035),
    "nova-lite":  (0.06, 0.24, 0.06 * 0.25, 0.06),
    "nova-pro":   (0.80, 3.20, 0.80 * 0.25, 0.80),
}

# Models that ARE billable but whose real Bedrock/region list price we will not invent here.
# They resolve to cost-unavailable (estimate_usd → None) until a deployment supplies a rate via an
# override map. This set is documentation-of-intent; membership does not change resolution — any
# unknown key is cost-unavailable regardless — but it records which unknowns are deliberate.
KNOWN_UNPRICED: frozenset[str] = frozenset({
    "nemotron",
    "kimi", "moonshot",
    "glm", "zai",
    "qwen",
    "deepseek",
    "llama",
    "mistral",
    "command",
})


@dataclass
class PriceTable:
    """A model-key → USD/Mtok rate table.

    ``fallback`` is the key used for a model that matches no rate. It defaults to ``None`` (no
    fallback), so an unknown model is **cost-unavailable** rather than silently Opus-priced. Set
    it explicitly (e.g. ``PriceTable(fallback="opus")``) only when a coarse guess is acceptable.
    """

    rates: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: dict(_DEFAULT_RATES))
    fallback: str | None = None

    def _rate(self, model: str) -> tuple[float, float, float, float] | None:
        """The matching rate tuple, or ``None`` when the model is cost-unavailable.

        Resolution is **longest-key-wins**: among all keys that are a case-insensitive substring
        of the model id, the most specific (longest) one is chosen, so a versioned id like
        ``us.anthropic.claude-opus-4-6-v1`` matches ``opus-4-6`` rather than the coarse ``opus``.
        Ties break on the key string for determinism, independent of dict insertion order.
        """
        m = (model or "").lower()
        best_key: str | None = None
        for key in self.rates:
            if key in m and (best_key is None
                             or (len(key), key) > (len(best_key), best_key)):
                best_key = key
        if best_key is not None:
            return self.rates[best_key]
        if self.fallback is not None:
            return self.rates.get(self.fallback)
        return None

    def has_rate(self, model: str) -> bool:
        """True when a real (or fallback) rate is known for ``model``."""
        return self._rate(model) is not None

    def message_cost_shape(self, input_tok: float, output_tok: float,
                           cache_read_tok: float, cache_creation_tok: float,
                           model: str = "") -> float:
        """Relative list-price cost of one message — used as a weight, not an absolute $.

        A cost-unavailable model contributes a **zero** weight (no fabricated shape) rather than
        borrowing Opus pricing.
        """
        r = self._rate(model)
        if r is None:
            return 0.0
        pin, pout, pcr, pcw = r
        return (input_tok * pin + output_tok * pout
                + cache_read_tok * pcr + cache_creation_tok * pcw) / 1_000_000.0

    def estimate_usd(self, input_tok: float, output_tok: float,
                     cache_read_tok: float, cache_creation_tok: float,
                     model: str = "") -> float | None:
        """Absolute provisional cost estimate (for live streaming before the billed number).

        Returns ``None`` when the model's price is unavailable — the caller must treat that as
        "cost unknown", never as ``$0``.
        """
        if self._rate(model) is None:
            return None
        return self.message_cost_shape(input_tok, output_tok,
                                       cache_read_tok, cache_creation_tok, model)

    # ------------------------------------------------------------------ override hooks
    @classmethod
    def load(cls, path: str | Path, *, fallback: str | None = None,
             merge_defaults: bool = True) -> "PriceTable":
        """Build a table from a JSON or YAML override map at ``path``.

        The file maps a model-key (matched as a case-insensitive substring, like the built-ins) to
        a rate, given either as a 4-list ``[input, output, cache_read, cache_creation]`` or a dict
        with those keys (``cache_read``/``cache_creation`` default to ``0.0``). Overrides are
        layered on top of the built-in defaults unless ``merge_defaults=False``; a key present in
        both wins from the override. Rates are USD per million tokens.
        """
        p = Path(path)
        text = p.read_text()
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml  # pyyaml is a hard dependency of aet
            raw = yaml.safe_load(text) or {}
        else:
            raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError(f"price map at {p} must be a mapping of model-key -> rate")

        rates: dict[str, tuple[float, float, float, float]] = (
            dict(_DEFAULT_RATES) if merge_defaults else {})
        for key, spec in raw.items():
            rates[key] = _coerce_rate(spec, key)
        return cls(rates=rates, fallback=fallback)

    @classmethod
    def from_env(cls, *, fallback: str | None = None) -> "PriceTable":
        """A table honoring the ``AET_PRICE_TABLE`` env var (a path to a JSON/YAML override map).

        When the variable is unset or empty, returns the built-in default table. This lets a
        deployment supply Bedrock/region-specific or newly-launched model prices without a code
        change.
        """
        path = os.environ.get(PRICE_TABLE_ENV, "").strip()
        if not path:
            return cls(fallback=fallback)
        return cls.load(path, fallback=fallback)


def _coerce_rate(spec, key: str) -> tuple[float, float, float, float]:
    """Normalize a JSON/YAML rate spec (list or dict) into a 4-float tuple."""
    if isinstance(spec, dict):
        return (
            float(spec.get("input", 0.0)),
            float(spec.get("output", 0.0)),
            float(spec.get("cache_read", 0.0)),
            float(spec.get("cache_creation", 0.0)),
        )
    if isinstance(spec, (list, tuple)):
        vals = [float(x) for x in spec]
        if not 2 <= len(vals) <= 4:
            raise ValueError(
                f"rate for {key!r} must have 2-4 numbers [input, output, cache_read, "
                f"cache_creation]; got {len(vals)}")
        vals += [0.0] * (4 - len(vals))
        return (vals[0], vals[1], vals[2], vals[3])
    raise ValueError(f"rate for {key!r} must be a list or a mapping, not {type(spec).__name__}")
