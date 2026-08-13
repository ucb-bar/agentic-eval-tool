"""Whether a reported cost is money — because on a subscription seat it is not.

**The conflation this exists to prevent.** The Claude Code CLI emits `total_cost_usd` on every run,
including a run on a subscription seat. There it is what those tokens *would* have cost on the API;
no card is charged for them. `aet` has been reading that field as money
(`tracking/claude_stream.py`) with no notion of which kind of run produced it, so a ledger that sums
a subscription run and a metered Bedrock run and calls the total "spend" presents quota consumption
as money.

That is not hypothetical. The project this was ported from published a figure that conflated them
before the distinction was drawn, and it is the single most misleading thing a cost ledger can get
wrong: a budget cap enforced against the sum is enforced against a number that is partly imaginary,
and a cost-per-outcome comparison between two arms on different billing modes compares different
units.

**The classification is by provider, and it is declarable.** A row may state its own `billing_mode`
— the runner knows better than any inference — and where it does not, the provider id decides. The
list is of METERED providers rather than subscription ones: a provider nobody has classified bills
against a seat as far as this is concerned, which is the conservative direction. Counting an unknown
provider as metered would inflate reported spend and, worse, would consume a budget cap that no card
is backing.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Provider ids that bill per token against an account with a payment method. Substring match, so a
#: versioned id (`aet:bedrock:opus`) classifies with its family.
METERED_PROVIDERS: tuple[str, ...] = (
    "bedrock", "vertex", "openai_compat", "anthropic_api",
)

#: The two modes. `per_token` is money; `subscription` is quota, and its dollar figure is notional.
MODES = ("per_token", "subscription")


def billing_mode_of(row: dict) -> str:
    """`per_token` or `subscription`, from the row's own field or its provider id.

    A declared `billing_mode` always wins: whoever launched the run knows how it is billed, and an
    inference that overrode them would be wrong in exactly the cases they bothered to state.
    """
    declared = row.get("billing_mode")
    if declared:
        mode = str(declared)
        if mode not in MODES:
            raise ValueError(f"billing_mode={mode!r} is outside {MODES}; an unrecognised mode would "
                             f"be silently treated as a seat and vanish from reported spend")
        return mode
    provider = str(row.get("provider") or "")
    return "per_token" if any(p in provider for p in METERED_PROVIDERS) else "subscription"


def is_metered(row: dict) -> bool:
    return billing_mode_of(row) == "per_token"


@dataclass
class Spend:
    """A cost total that has not lost track of what kind of cost it is.

    The two halves are never added. `metered_usd` is what a card was charged; `subscription_usd` is
    what the same tokens would have cost on the API and is reported beside it, labelled, so a reader
    can see the scale of the quota consumption without it entering a budget.
    """

    metered_usd: float = 0.0
    subscription_usd: float = 0.0
    metered_rows: int = 0
    subscription_rows: int = 0
    #: Rows carrying no cost figure at all. Counted rather than treated as zero — an unpriced row and
    #: a free row are different facts, and `PriceTable` already refuses to guess for the same reason.
    unpriced_rows: int = 0

    def add(self, row: dict, usd: float | None) -> None:
        if usd is None:
            self.unpriced_rows += 1
            return
        if is_metered(row):
            self.metered_usd += float(usd)
            self.metered_rows += 1
        else:
            self.subscription_usd += float(usd)
            self.subscription_rows += 1

    @property
    def against_budget(self) -> float:
        """What a spend cap governs. **Metered only** — a cap enforced against the sum would be
        enforced against a number no card is backing."""
        return self.metered_usd

    def to_dict(self) -> dict:
        return {"metered_usd": round(self.metered_usd, 6),
                "subscription_usd_notional": round(self.subscription_usd, 6),
                "metered_rows": self.metered_rows,
                "subscription_rows": self.subscription_rows,
                "unpriced_rows": self.unpriced_rows,
                "against_budget_usd": round(self.against_budget, 6)}

    def describe(self) -> str:
        parts = [f"${self.metered_usd:.4f} metered ({self.metered_rows} row(s))"]
        if self.subscription_rows:
            parts.append(f"${self.subscription_usd:.4f} NOTIONAL on a subscription seat "
                         f"({self.subscription_rows} row(s)) — not money, not summed")
        if self.unpriced_rows:
            parts.append(f"{self.unpriced_rows} row(s) unpriced")
        return "; ".join(parts)


def split(rows) -> Spend:
    """Total a sequence of rows, keeping the modes apart. Each row supplies its own `cost_usd`."""
    s = Spend()
    for row in rows:
        usd = row.get("cost_usd", row.get("total_cost_usd"))
        s.add(row, None if usd is None else float(usd))
    return s
