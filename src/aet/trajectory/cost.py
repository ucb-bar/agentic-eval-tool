"""A cost figure that never loses track of whether it is money, where it came from, and which
price map produced it.

``final_cost_usd`` on a :class:`~aet.trajectory.model.RunTrajectory` answers "how much" but not
"is that a bill, a list-price estimate on a subscription seat, or a value we could not price at
all". Those are different facts and a ledger that flattens them mis-reports spend (see
``trajectory/billing.py`` for the conflation this guards against). A :class:`CostRecord` carries
the number *with* its provenance so a reader — and a budget cap — can treat it correctly.

The cardinal distinction: ``value_usd is None`` means **unpriced** (unknown), which is NOT the
same as ``value_usd == 0.0`` (genuinely free). An unpriced run must never silently become ``$0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# kind: is the dollar figure money, notional (subscription seat), or absent?
COST_KINDS = ("metered", "subscription_notional", "unpriced")
# source: how the figure was obtained.
COST_SOURCES = ("provider_billed", "price_calculated", "admin_reconciled")


@dataclass
class CostRecord:
    """Nullable cost + full provenance (schema §3 of the Codex×AET contract)."""

    value_usd: float | None = None
    kind: str = "unpriced"            # one of COST_KINDS
    source: str = "price_calculated"  # one of COST_SOURCES
    model_requested: str = ""
    model_resolved: str = ""
    price_table_id: str = ""
    price_table_sha256: str = ""
    pricing_url: str = ""
    calculated_at: str = ""
    long_context_multiplier_applied: bool = False
    service_tier: str = "standard"
    # optional per-bucket breakdown (uncached_input / cache_read / cache_write / output), each USD
    breakdown_usd: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in COST_KINDS:
            raise ValueError(f"cost kind {self.kind!r} not in {COST_KINDS}")
        if self.source not in COST_SOURCES:
            raise ValueError(f"cost source {self.source!r} not in {COST_SOURCES}")
        # An unpriced record must not carry a dollar value, and a priced one must carry a number:
        # this is the invariant that keeps "unknown" from decaying into "$0".
        if self.kind == "unpriced" and self.value_usd is not None:
            raise ValueError("an 'unpriced' cost must have value_usd=None, not a number")
        if self.kind != "unpriced" and self.value_usd is None:
            raise ValueError(f"a {self.kind!r} cost must carry a value_usd, not None")

    @property
    def is_unpriced(self) -> bool:
        return self.value_usd is None

    @property
    def is_money(self) -> bool:
        """True only for a real metered charge — what a budget cap governs."""
        return self.kind == "metered" and self.value_usd is not None

    @classmethod
    def unpriced(cls, *, model_requested: str = "", model_resolved: str = "",
                 price_table_id: str = "", reason: str = "") -> "CostRecord":
        return cls(value_usd=None, kind="unpriced", source="price_calculated",
                   model_requested=model_requested, model_resolved=model_resolved or model_requested,
                   price_table_id=price_table_id, service_tier=reason or "standard")

    def to_dict(self) -> dict:
        return {
            "value_usd": self.value_usd,
            "kind": self.kind,
            "source": self.source,
            "model_requested": self.model_requested,
            "model_resolved": self.model_resolved,
            "price_table_id": self.price_table_id,
            "price_table_sha256": self.price_table_sha256,
            "pricing_url": self.pricing_url,
            "calculated_at": self.calculated_at,
            "long_context_multiplier_applied": self.long_context_multiplier_applied,
            "service_tier": self.service_tier,
            "breakdown_usd": dict(self.breakdown_usd),
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "CostRecord | None":
        if not d:
            return None
        return cls(
            value_usd=(None if d.get("value_usd") is None else float(d["value_usd"])),
            kind=d.get("kind", "unpriced"),
            source=d.get("source", "price_calculated"),
            model_requested=d.get("model_requested", ""),
            model_resolved=d.get("model_resolved", ""),
            price_table_id=d.get("price_table_id", ""),
            price_table_sha256=d.get("price_table_sha256", ""),
            pricing_url=d.get("pricing_url", ""),
            calculated_at=d.get("calculated_at", ""),
            long_context_multiplier_applied=bool(d.get("long_context_multiplier_applied", False)),
            service_tier=d.get("service_tier", "standard"),
            breakdown_usd=dict(d.get("breakdown_usd", {}) or {}),
        )
