"""Token price tables — used ONLY for provisional cost while streaming, and as the per-message
cost *shape* when distributing a round's billed total across its messages.

Authoritative cost always comes from the CLI's ``total_cost_usd`` at a round's result event. A
live monitor, though, has no billed number until that event lands, so it estimates from list
prices; and even in batch we spread the billed round total across interior points using
list-price weights (so the curve rises smoothly yet still sums to the real bill at round ends).

Rates are USD per **million** tokens. Cache-creation (write) is charged above input; cache-read
is charged far below. Defaults follow Claude list pricing; override per deployment as needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# USD / Mtok: (input, output, cache_read, cache_creation)
_DEFAULT_RATES: dict[str, tuple[float, float, float, float]] = {
    # Opus-class list price (matches the oscar-merlin reference PR_IN/OUT/CR/CW).
    "opus":   (15.0, 75.0, 1.5, 18.75),
    "sonnet": (3.0, 15.0, 0.3, 3.75),
    "haiku":  (0.80, 4.0, 0.08, 1.0),
}
_FALLBACK = "opus"


@dataclass
class PriceTable:
    rates: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: dict(_DEFAULT_RATES))
    fallback: str = _FALLBACK

    def _rate(self, model: str) -> tuple[float, float, float, float]:
        m = (model or "").lower()
        for key, r in self.rates.items():
            if key in m:
                return r
        return self.rates.get(self.fallback, _DEFAULT_RATES[_FALLBACK])

    def message_cost_shape(self, input_tok: float, output_tok: float,
                           cache_read_tok: float, cache_creation_tok: float,
                           model: str = "") -> float:
        """Relative list-price cost of one message — used as a weight, not an absolute $."""
        pin, pout, pcr, pcw = self._rate(model)
        return (input_tok * pin + output_tok * pout
                + cache_read_tok * pcr + cache_creation_tok * pcw) / 1_000_000.0

    def estimate_usd(self, input_tok: float, output_tok: float,
                     cache_read_tok: float, cache_creation_tok: float,
                     model: str = "") -> float:
        """Absolute provisional cost estimate (for live streaming before the billed number)."""
        return self.message_cost_shape(input_tok, output_tok,
                                       cache_read_tok, cache_creation_tok, model)
