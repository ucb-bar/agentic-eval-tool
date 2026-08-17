"""Versioned price snapshots — pin the exact rate map a run was costed with, and copy it into the
experiment root so a campaign is reproducible after list prices change.

A price *table* (``pricing.PriceTable``) is the live lookup; a price *snapshot* is that table
frozen with an id, a source URL, a capture date, a ``verified`` flag, and a content hash. Every
:class:`~aet.trajectory.cost.CostRecord` names the snapshot id + sha256 that produced it, so a
number can always be traced back to the exact rates — and a reviewer can refuse to bill against an
``verified: false`` (placeholder) snapshot.

Rates are USD per **million** tokens, 4 buckets ``(input, output, cache_read, cache_write)``. Codex
reports a distinct ``cache_write`` bucket; it maps onto ``PriceTable``'s ``cache_creation`` slot,
so the same table prices both Claude-style and Codex-style usage.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from aet.trajectory.billing import billing_mode_of
from aet.trajectory.cost import CostRecord
from aet.trajectory.pricing import PriceTable, _coerce_rate

# The bundled default snapshot ships beside this module (see pyproject package-data).
_DATA_DIR = Path(__file__).parent / "data" / "openai_prices"
DEFAULT_OPENAI_SNAPSHOT_ID = "openai-2026-08-17"


@dataclass
class PriceSnapshot:
    """A frozen, hashable rate map + its provenance."""

    price_table_id: str
    rates: dict[str, tuple[float, float, float, float]]
    pricing_url: str = ""
    captured_at: str = ""
    provider: str = "openai"
    verified: bool = False
    service_tier: str = "standard"
    raw_bytes: bytes = b""          # the exact file bytes, for a stable content hash

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> "PriceSnapshot":
        p = Path(path)
        raw = p.read_bytes()
        text = raw.decode("utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml
            doc = yaml.safe_load(text) or {}
        else:
            doc = json.loads(text)
        rates: dict[str, tuple[float, float, float, float]] = {}
        for key, spec in (doc.get("rates") or {}).items():
            # accept the Codex bucket name `cache_write` as an alias for `cache_creation`
            if isinstance(spec, dict) and "cache_write" in spec and "cache_creation" not in spec:
                spec = {**spec, "cache_creation": spec.get("cache_write", 0.0)}
            rates[key] = _coerce_rate(spec, key)
        return cls(
            price_table_id=doc.get("price_table_id", p.stem),
            rates=rates,
            pricing_url=doc.get("pricing_url", ""),
            captured_at=str(doc.get("captured_at", "")),
            provider=doc.get("provider", "openai"),
            verified=bool(doc.get("verified", False)),
            service_tier=doc.get("service_tier", "standard"),
            raw_bytes=raw,
        )

    @classmethod
    def default_openai(cls) -> "PriceSnapshot":
        return cls.load(_DATA_DIR / f"{DEFAULT_OPENAI_SNAPSHOT_ID}.yaml")

    # ------------------------------------------------------------------ derived
    def sha256(self) -> str:
        """Content hash of the exact snapshot bytes (stable across reads)."""
        if self.raw_bytes:
            return hashlib.sha256(self.raw_bytes).hexdigest()
        payload = json.dumps({"id": self.price_table_id, "rates": self.rates}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def price_table(self, *, merge_defaults: bool = False) -> PriceTable:
        """A :class:`PriceTable` over this snapshot's rates (no coarse fallback)."""
        from aet.trajectory.pricing import _DEFAULT_RATES
        rates = dict(_DEFAULT_RATES) if merge_defaults else {}
        rates.update(self.rates)
        return PriceTable(rates=rates, fallback=None)

    def copy_into_experiment(self, experiment_root: str | Path,
                             filename: str = "price-table.yaml") -> Path:
        """Copy the resolved snapshot into ``<experiment_root>/config/<filename>`` (reproducibility).

        Writes the exact captured bytes when available (so the hash the run recorded still matches),
        else serializes the rate map. Returns the written path.
        """
        root = Path(experiment_root)
        dest = root / "config" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.raw_bytes:
            dest.write_bytes(self.raw_bytes)
        else:
            import yaml
            dest.write_text(yaml.safe_dump({
                "price_table_id": self.price_table_id, "provider": self.provider,
                "pricing_url": self.pricing_url, "captured_at": self.captured_at,
                "verified": self.verified, "unit": "per_million_tokens",
                "rates": {k: list(v) for k, v in self.rates.items()},
            }, sort_keys=False))
        return dest


def cost_record_for(totals: dict, *, snapshot: PriceSnapshot,
                    model_requested: str, model_resolved: str = "",
                    billing_row: dict | None = None,
                    calculated_at: str | None = None) -> CostRecord:
    """Turn a :meth:`CodexRun.totals` dict into a fully-provenanced :class:`CostRecord`.

    Applies the token subset semantics: prices ``uncached_input`` at the input rate, ``cache_read``
    at the cache-read rate, ``cache_write`` at the cache-write rate, ``output`` at the output rate.
    Returns an **unpriced** record (``value_usd=None``) when the model has no rate or when input
    tokens were never reported — never ``$0``.

    ``billing_row`` (``{"provider": ..., "billing_mode": ...}``) decides money-vs-notional; default
    is metered ``openai`` (the API-key path), which is the conservative direction for a budget.
    """
    model_resolved = model_resolved or model_requested
    table = snapshot.price_table()
    # A caller that wants a byte-deterministic result (e.g. the idempotent batch importer) passes an
    # explicit ``calculated_at`` (often ""); the live recorder leaves it None to stamp wall-now.
    if calculated_at is None:
        calculated_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rate = table._rate(model_resolved) or table._rate(model_requested)
    input_total = totals.get("input_tokens")
    if rate is None or input_total is None:
        return CostRecord.unpriced(
            model_requested=model_requested, model_resolved=model_resolved,
            price_table_id=snapshot.price_table_id,
            reason=("no_rate" if rate is None else "no_input_tokens"))

    pin, pout, pcr, pcw = rate
    uncached = totals.get("uncached_input_tokens") or 0
    cache_read = totals.get("cached_input_tokens") or 0
    cache_write = totals.get("cache_write_input_tokens") or 0
    output = totals.get("output_tokens") or 0
    breakdown = {
        "uncached_input_usd": uncached * pin / 1_000_000.0,
        "cache_read_usd": cache_read * pcr / 1_000_000.0,
        "cache_write_usd": cache_write * pcw / 1_000_000.0,
        "output_usd": output * pout / 1_000_000.0,
    }
    value = round(sum(breakdown.values()), 6)

    row = dict(billing_row or {"provider": "openai"})
    mode = billing_mode_of(row)
    kind = "metered" if mode == "per_token" else "subscription_notional"
    return CostRecord(
        value_usd=value, kind=kind, source="price_calculated",
        model_requested=model_requested, model_resolved=model_resolved,
        price_table_id=snapshot.price_table_id, price_table_sha256=snapshot.sha256(),
        pricing_url=snapshot.pricing_url, calculated_at=calculated_at,
        service_tier=snapshot.service_tier,
        breakdown_usd={k: round(v, 6) for k, v in breakdown.items()},
    )
