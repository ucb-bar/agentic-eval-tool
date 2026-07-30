"""Cross-experiment spend rollup — aggregate token/cost across many run directories.

Given a set of run directories (or roots that contain them) this module answers "how much have we
spent, on what models, and how did cumulative spend grow over calendar time" — the numbers a
research program needs to stay under a fixed budget across many experiments.

It reuses the repo's existing readers rather than re-parsing raw logs:
  * per-run cost/tokens come from ``logs/metrics.jsonl`` (the exact metric-name contract that
    ``aet runs`` reads: ``aet.agent.cost_usd`` + ``gen_ai.usage.*``) and fall back to the canonical
    ``metrics/trajectory.json`` (:class:`~aet.trajectory.model.RunTrajectory`) when a run only
    carries the trajectory artifact;
  * per-run identity/timestamp comes from ``run_record.json`` / ``run_manifest.yaml`` /
    ``logs/params.json``.

Honesty policy: a run with **no recorded cost** is counted as *cost-unavailable* — it is surfaced
separately (``unpriced_runs``) and never folded into the total as ``$0``.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# metric-name contract shared with `aet runs` (see cli/commands/reporting.py::_cmd_runs)
_M_COST = "aet.agent.cost_usd"
_M_INPUT = "gen_ai.usage.input_tokens"
_M_OUTPUT = "gen_ai.usage.output_tokens"
_M_CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
_M_CACHE_CREATE = "gen_ai.usage.cache_creation.input_tokens"

# within-run per-model split, written by EvalRunLogger.log_model_usage as
# `per_model.<safe-model>.<field>` (safe = model with `-`/`.` replaced by `_`).
_M_PER_MODEL_PREFIX = "per_model."
_PER_MODEL_FIELDS = {
    "cost_usd": "cost",
    "input_tokens_raw": "input",
    "output_tokens": "output",
    "cache_read_tokens": "cache_read",
    "cache_creation_tokens": "cache_creation",
}


@dataclass
class TokenTotals:
    """Token counts split by billing type (cache_total = read + creation, or a combined figure
    when only a trajectory's merged cache count is available)."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_creation: int = 0
    cache_total: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.cache_total

    def add(self, other: "TokenTotals") -> None:
        self.input += other.input
        self.output += other.output
        self.cache_read += other.cache_read
        self.cache_creation += other.cache_creation
        self.cache_total += other.cache_total

    def to_dict(self) -> dict:
        return {
            "input": self.input, "output": self.output,
            "cache_read": self.cache_read, "cache_creation": self.cache_creation,
            "cache_total": self.cache_total, "total": self.total,
        }


@dataclass
class RunModelSpend:
    """One model's within-run spend/tokens (from a run's ``per_model.<model>.*`` metrics)."""

    model: str
    cost_usd: float | None = None           # None => this model's cost was unavailable
    tokens: TokenTotals = field(default_factory=TokenTotals)

    @property
    def cost_available(self) -> bool:
        return self.cost_usd is not None

    def to_dict(self) -> dict:
        return {
            "model": self.model, "cost_usd": self.cost_usd,
            "cost_available": self.cost_available, "tokens": self.tokens.to_dict(),
        }


@dataclass
class RunSpend:
    """The spend/tokens/identity of a single run."""

    run_dir: str
    run_id: str = ""
    suite: str = ""
    model: str = ""
    timestamp: str | None = None            # ISO created_at, for calendar ordering
    cost_usd: float | None = None           # None => cost unavailable (never treated as $0)
    tokens: TokenTotals = field(default_factory=TokenTotals)
    # within-run per-model split (keyed by recorded safe-model name); empty when the run recorded
    # no `per_model.*` metrics — then the whole run is attributed to its single identity `model`.
    per_model: dict[str, RunModelSpend] = field(default_factory=dict)

    @property
    def cost_available(self) -> bool:
        return self.cost_usd is not None

    def to_dict(self) -> dict:
        return {
            "run_dir": self.run_dir, "run_id": self.run_id, "suite": self.suite,
            "model": self.model, "timestamp": self.timestamp,
            "cost_usd": self.cost_usd, "cost_available": self.cost_available,
            "tokens": self.tokens.to_dict(),
            "per_model": {k: v.to_dict() for k, v in self.per_model.items()},
        }


@dataclass
class ModelSpend:
    """Aggregated spend/tokens for one model across the rollup (a true within-run split: a run that
    used several models contributes to each of their :class:`ModelSpend` records)."""

    model: str
    cost_usd: float = 0.0
    n_runs: int = 0
    n_priced_runs: int = 0
    tokens: TokenTotals = field(default_factory=TokenTotals)
    activity_share: float = 0.0             # fraction of the rollup's total billed tokens

    def to_dict(self) -> dict:
        return {
            "model": self.model, "cost_usd": self.cost_usd,
            "n_runs": self.n_runs, "n_priced_runs": self.n_priced_runs,
            "tokens": self.tokens.to_dict(), "activity_share": self.activity_share,
        }


@dataclass
class SpendRollup:
    """The aggregate of many :class:`RunSpend` records."""

    runs: list[RunSpend] = field(default_factory=list)
    total_cost_usd: float = 0.0
    tokens: TokenTotals = field(default_factory=TokenTotals)
    per_model: dict[str, ModelSpend] = field(default_factory=dict)
    cumulative: list[dict] = field(default_factory=list)   # time-ordered {run_id, timestamp, cost, cumulative}
    unpriced_runs: int = 0                                  # runs with cost unavailable
    budget_usd: float | None = None
    headroom_usd: float | None = None                      # budget - total (None if no budget)
    over_budget: bool = False

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def to_dict(self) -> dict:
        return {
            "n_runs": self.n_runs,
            "total_cost_usd": self.total_cost_usd,
            "tokens": self.tokens.to_dict(),
            "per_model": {k: v.to_dict() for k, v in self.per_model.items()},
            "cumulative": self.cumulative,
            "unpriced_runs": self.unpriced_runs,
            "budget_usd": self.budget_usd,
            "headroom_usd": self.headroom_usd,
            "over_budget": self.over_budget,
            "runs": [r.to_dict() for r in self.runs],
        }


# ---------------------------------------------------------------------------- discovery
_RUN_MARKERS = (
    ("logs", "metrics.jsonl"),
    ("metrics", "trajectory.json"),
    ("run_manifest.yaml",),
    ("run_record.json",),
)


def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and any(p.joinpath(*parts).exists() for parts in _RUN_MARKERS)


def discover_run_dirs(root: str | Path) -> list[Path]:
    """Run directories under ``root`` (or ``root`` itself if it is one).

    A run directory is one carrying any of ``logs/metrics.jsonl``, ``metrics/trajectory.json``,
    ``run_manifest.yaml`` or ``run_record.json``. The walk does not descend into a run dir once
    found, so nested artifact trees are not mistaken for sub-runs.
    """
    root = Path(root)
    if _is_run_dir(root):
        return [root]
    found: list[Path] = []
    for child in sorted(root.rglob("*")):
        if not child.is_dir():
            continue
        # skip anything already inside a discovered run dir
        if any(child == r or r in child.parents for r in found):
            continue
        if _is_run_dir(child):
            found.append(child)
    return found


# ---------------------------------------------------------------------------- single-run read
def _read_jsonl(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def _read_metrics(
    run_dir: Path,
) -> tuple[float | None, TokenTotals, bool, dict[str, RunModelSpend]]:
    """(cost_usd, tokens, saw_agent_metrics, per_model) from ``logs/metrics.jsonl`` — the `aet runs`
    contract, extended with the within-run ``per_model.<model>.*`` split.

    Values are logged as final scalars; the last occurrence of each name wins.
    """
    path = run_dir / "logs" / "metrics.jsonl"
    cost: float | None = None
    tok = TokenTotals()
    saw = False
    per_model: dict[str, RunModelSpend] = {}
    if not path.is_file():
        return cost, tok, saw, per_model
    for m in _read_jsonl(path):
        name = m.get("name")
        val = m.get("value")
        if val is None or not name:
            continue
        if name == _M_COST:
            cost = float(val)
            saw = True
        elif name == _M_INPUT:
            tok.input = int(val)
            saw = True
        elif name == _M_OUTPUT:
            tok.output = int(val)
            saw = True
        elif name == _M_CACHE_READ:
            tok.cache_read = int(val)
            saw = True
        elif name == _M_CACHE_CREATE:
            tok.cache_creation = int(val)
            saw = True
        elif name.startswith(_M_PER_MODEL_PREFIX):
            # `per_model.<safe-model>.<field>`; safe-model never contains a dot (`-`/`.` → `_`),
            # so the last dotted segment is the field and the rest is the model key.
            model_key, _, mfield = name[len(_M_PER_MODEL_PREFIX):].rpartition(".")
            if not model_key or mfield not in _PER_MODEL_FIELDS:
                continue
            rms = per_model.setdefault(model_key, RunModelSpend(model=model_key))
            attr = _PER_MODEL_FIELDS[mfield]
            if attr == "cost":
                rms.cost_usd = float(val)
            else:
                setattr(rms.tokens, attr, int(val))
    tok.cache_total = tok.cache_read + tok.cache_creation
    for rms in per_model.values():
        rms.tokens.cache_total = rms.tokens.cache_read + rms.tokens.cache_creation
    return cost, tok, saw, per_model


def _read_trajectory(run_dir: Path):
    """The canonical trajectory artifact, or None. Reuses RunTrajectory.from_json."""
    fast = run_dir / "metrics" / "trajectory.json"
    if not fast.is_file():
        return None
    from aet.trajectory.model import RunTrajectory
    try:
        return RunTrajectory.from_json(fast)
    except Exception:
        return None


def _read_identity(run_dir: Path) -> tuple[str, str, str, str | None]:
    """(run_id, suite, model, timestamp) from run_record.json / run_manifest.yaml / params.json."""
    run_id = run_dir.name
    suite = run_dir.parent.name
    model = ""
    timestamp: str | None = None

    rec = run_dir / "run_record.json"
    if rec.is_file():
        try:
            d = json.loads(rec.read_text())
            run_id = d.get("run_id") or run_id
            suite = d.get("suite") or suite
            model = d.get("model") or model
            timestamp = d.get("created_at") or timestamp
        except Exception:
            pass

    man = run_dir / "run_manifest.yaml"
    if man.is_file():
        try:
            from aet.core.run_manifest import RunManifest
            m = RunManifest.load(man)
            run_id = m.run_id or run_id
            suite = m.suite or suite
            model = model or (m.model or "")
            timestamp = timestamp or (m.created_at or None)
        except Exception:
            pass

    if not model:
        params = run_dir / "logs" / "params.json"
        if params.is_file():
            try:
                p = json.loads(params.read_text())
                model = p.get("gen_ai.response.model", "") or model
            except Exception:
                pass
    return run_id, suite, model, timestamp


def read_run_spend(run_dir: str | Path) -> RunSpend:
    """Read one run's spend/tokens/identity, reusing the repo's readers.

    Cost/tokens come from ``logs/metrics.jsonl`` first (the `aet runs` metric contract); a run that
    only carries ``metrics/trajectory.json`` is read from that instead. A run with neither has
    ``cost_usd=None`` (cost unavailable).
    """
    run_dir = Path(run_dir)
    run_id, suite, model, timestamp = _read_identity(run_dir)

    cost, tok, saw, per_model = _read_metrics(run_dir)
    if not saw:
        traj = _read_trajectory(run_dir)
        if traj is not None:
            cost = float(traj.final_cost_usd)
            tok = TokenTotals(
                input=int(traj.final_input_tokens),
                output=int(traj.final_output_tokens),
                cache_total=int(traj.final_cache_tokens),
            )
            model = model or traj.model

    # A trajectory may also refine the timestamp-less metrics path with a first-line ts.
    if timestamp is None:
        mp = run_dir / "logs" / "metrics.jsonl"
        if mp.is_file():
            for m in _read_jsonl(mp):
                if m.get("ts"):
                    timestamp = m["ts"]
                    break

    return RunSpend(run_dir=str(run_dir), run_id=run_id, suite=suite, model=model,
                    timestamp=timestamp, cost_usd=cost, tokens=tok, per_model=per_model)


# ---------------------------------------------------------------------------- rollup
def rollup_runs(run_dirs: Iterable[str | Path], *,
                budget_usd: float | None = None) -> SpendRollup:
    """Aggregate spend/tokens across ``run_dirs`` (roots are expanded to the runs they contain).

    Produces total cost, tokens by type, per-model spend, and a calendar-time-ordered
    cumulative-spend series. Runs with no recorded cost are counted in ``unpriced_runs`` and never
    added as ``$0``. When ``budget_usd`` is given, ``headroom_usd = budget_usd - total_cost_usd``
    and ``over_budget`` flags a total above the ceiling.
    """
    # expand roots → concrete run dirs (deduped, order-stable)
    seen: set[str] = set()
    resolved: list[Path] = []
    for root in run_dirs:
        for rd in discover_run_dirs(root):
            key = str(rd.resolve())
            if key not in seen:
                seen.add(key)
                resolved.append(rd)

    roll = SpendRollup(budget_usd=budget_usd)
    for rd in resolved:
        rs = read_run_spend(rd)
        roll.runs.append(rs)
        roll.tokens.add(rs.tokens)

        # Run-level total is authoritative and counted once (keeps total/headroom behavior).
        if rs.cost_available:
            roll.total_cost_usd += rs.cost_usd
        else:
            roll.unpriced_runs += 1

        # Per-model split: prefer the run's recorded within-run `per_model.*` breakdown so a run
        # that mixed an orchestrator + delegated sub-agent models contributes to EACH model's
        # ModelSpend. A run without that breakdown falls back to its single identity model.
        if rs.per_model:
            contributions = [
                (rms.model, rms.cost_usd, rms.tokens) for rms in rs.per_model.values()
            ]
        else:
            contributions = [(rs.model or "(unknown)", rs.cost_usd, rs.tokens)]
        for model_key, mcost, mtok in contributions:
            ms = roll.per_model.setdefault(model_key, ModelSpend(model=model_key))
            ms.n_runs += 1
            ms.tokens.add(mtok)
            if mcost is not None:
                ms.cost_usd += mcost
                ms.n_priced_runs += 1

    # Each model's activity share = its fraction of the rollup's total billed tokens.
    total_billed = roll.tokens.total
    for ms in roll.per_model.values():
        ms.activity_share = (ms.tokens.total / total_billed) if total_billed else 0.0

    # calendar-time cumulative series (runs without a timestamp sort last, by run_id)
    def _key(r: RunSpend):
        return (r.timestamp is None, r.timestamp or "", r.run_id)

    cum = 0.0
    for r in sorted(roll.runs, key=_key):
        if r.cost_available:
            cum += r.cost_usd
        roll.cumulative.append({
            "run_id": r.run_id, "timestamp": r.timestamp,
            "cost_usd": r.cost_usd, "cumulative_usd": cum,
        })

    if budget_usd is not None:
        roll.headroom_usd = budget_usd - roll.total_cost_usd
        roll.over_budget = roll.total_cost_usd > budget_usd
    return roll
