"""Canonical agentic-trajectory data-model — the spine of ``aet.trajectory``.

A ``RunTrajectory`` is the repo-agnostic record of *what an agent did over time* while
producing a deliverable: cumulative token consumption (input / output / cache), cumulative
cost, an activity timeline (thinking / reading / writing / bash / long tool-waits), test-pass
milestones from an external oracle, and one-shot progress checkpoints (first parse, first
elaboration, all-public-passing). It is the single structure every consumer reads — the batch
importer, the native recorder, the live monitor, and the plots.

Design invariants:
  * **Pure stdlib, no numpy.** Every field is a plain list/scalar so this module stays a
    dependency-free core; array smoothing for plots lives in ``aet.viz``.
  * **Append-only.** ``points``/``bands``/``milestones``/``checkpoints``/``rounds`` are grown
    incrementally, so the exact same object is built by a completed-run importer and by a live
    stream, one point at a time. There is one code path, not two.
  * **Self-describing.** ``classifier_config`` is carried on the trajectory so a reader can see
    (and reproduce) how activities were categorised — the harness never hardcodes tool rules.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 1.3 adds first-class inference attempts and agent hierarchy. Older files load
# unchanged because the new collection defaults empty. 1.2 added cache splits.
SCHEMA_VERSION = "1.3"


@dataclass
class TrajectoryPoint:
    """One sample on the cumulative token/cost curves (one agent message/turn)."""

    t_s: float                       # seconds from trajectory start (t=0 at round 0)
    cum_input_tokens: float = 0.0
    cum_output_tokens: float = 0.0
    cum_cache_tokens: float = 0.0    # cache_read + cache_creation (matches load_arm's T_ca)
    cum_cache_read_tokens: float = 0.0      # cache hits (billed ~10× cheaper)
    cum_cache_creation_tokens: float = 0.0  # cache writes (billed at a ~25% premium)
    cum_reasoning_tokens: float = 0.0       # reasoning output (a SUBSET of cum_output_tokens)
    cum_cost_usd: float = 0.0        # authoritative at round ends; provisional mid-stream
    round_index: int = 0
    provisional_cost: bool = False   # True while streaming before the round's result event

    @property
    def cum_total_tokens(self) -> float:
        # reasoning is a subset of output — NOT added again, or output would be double-counted
        return self.cum_input_tokens + self.cum_output_tokens + self.cum_cache_tokens


@dataclass
class ActivityBand:
    """A contiguous interval spent in one activity category (drives the activity-share view)."""

    t0_s: float
    t1_s: float
    category: str                    # "think" | "read" | "write" | "bash" | "tool" (extensible)
    tool_name: str = ""
    weight: float = 1.0              # classifier weight (long waits weigh more in the share)
    round_index: int = 0
    is_error: bool = False

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t1_s - self.t0_s)


@dataclass
class TestMilestone:
    """An external-oracle test-pass reading at a wall time (e.g. selfcheck 13 → 17 → 20)."""

    __test__ = False   # not a pytest test class despite the "Test" prefix

    t_s: float
    n_passed: int
    n_total: int
    scope: str = "all"
    round_index: int | None = None
    source: str = ""                 # "selfcheck_log" | "qa_verdict" | ...


#: Ordered. A run passes through these in sequence, and the ordering is what makes "time to first X"
#: comparable across runs that never reach the same depth.
CHECKPOINT_KINDS = (
    "first_file",           # the agent wrote its first candidate artifact
    "first_parse",          # something it wrote parsed
    "first_module_elab",    # one module elaborated
    "full_elab",            # the whole design elaborated
    "first_public_pass",    # the first public test passed
    "public_50",            # half the public requirements passing
    "public_90",
    "public_all",           # every mandatory public test passing
)


@dataclass
class Checkpoint:
    """A one-shot progress landmark at a wall time.

    Distinct from ``TestMilestone``, which is a *reading* on a single pass/total axis and can move in
    both directions. A checkpoint is a threshold crossed once: the first time the agent's output
    parsed, the first time a module elaborated. Time-to-first-parse and time-to-first-elaboration are
    not expressible as ``n_passed/n_total`` without abusing ``scope``, and the existing
    ``EvalRunLogger.record_elaboration`` records an *iteration index* rather than a ``t_s``, so
    neither lands on the trajectory today.

    ``reached`` is always True for a stored checkpoint — absence is how "never reached" is
    represented, because a checkpoint at ``t_s = None`` would sort somewhere and be plotted."""

    t_s: float
    kind: str                        # one of CHECKPOINT_KINDS (not enforced: callers may add their own)
    scope: str = "all"               # e.g. a module_id, when the checkpoint is per-module
    round_index: int | None = None
    source: str = ""                 # "build_log" | "harness" | ...
    detail: str = ""


@dataclass
class RoundBoundary:
    """One agent invocation ("round") — its wall span + billed totals + QA verdict."""

    index: int
    t_start_s: float
    t_end_s: float
    cost_usd: float | None = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0            # cache_read + cache_creation (kept as their sum, back-compat)
    cache_read_tokens: int = 0       # 1.2: the read/write split (0 on pre-1.2 rounds)
    cache_creation_tokens: int = 0   # 1.2: cache writes
    reasoning_tokens: int = 0        # 1.2: reasoning output (a SUBSET of output_tokens)
    n_passed: int | None = None
    n_total: int | None = None
    session_id: str = ""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_end_s - self.t_start_s)


@dataclass
class InferenceRecord:
    """One provider request/attempt with measured and explicitly inferred cache facts.

    Token counters are provider-reported measurements. ``reasoning_tokens`` is a
    subset of output, and cache read/write are separate input classes. Context
    occupancy and TTL expiry are estimates, never physical KV-cache measurements.
    """

    request_id: str
    t_start_s: float
    t_end_s: float
    agent_id: str = ""
    parent_agent_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    call_id: str = ""
    session_id: str = ""
    attempt: int = 1
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    status: str = "unknown"
    retry: bool = False
    cost_usd: float | None = None
    cost_source: str = "unavailable"
    billing_mode: str = "per_token"
    context_window_tokens: int | None = None
    estimated_context_tokens: int | None = None
    cache_ttl_s: float | None = None
    ttl_inference: str = "unavailable"

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_end_s - self.t_start_s)

    @property
    def billed_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def cache_hit_ratio(self) -> float | None:
        total = self.billed_input_tokens
        return self.cache_read_tokens / total if total else None

    @property
    def context_occupancy_ratio(self) -> float | None:
        if not self.context_window_tokens or self.estimated_context_tokens is None:
            return None
        # Do not clamp: >1 is useful evidence that the configured window or
        # provider semantics do not match, not proof of physical over-capacity.
        return self.estimated_context_tokens / self.context_window_tokens


@dataclass
class RunTrajectory:
    """The full canonical trajectory for one run — built once, read by every consumer."""

    run_id: str = ""
    source: str = ""                 # "aet-native" | "import:capsule-bench" | "stream" ...
    model: str = ""
    schema_version: str = SCHEMA_VERSION
    duration_s: float = 0.0          # active wall = sum of round durations
    num_rounds: int = 0
    provisional: bool = False        # True while streaming before a terminal result event
    points: list[TrajectoryPoint] = field(default_factory=list)
    bands: list[ActivityBand] = field(default_factory=list)
    milestones: list[TestMilestone] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    rounds: list[RoundBoundary] = field(default_factory=list)
    inferences: list[InferenceRecord] = field(default_factory=list)
    # NULLABLE by design: ``None`` means *unpriced* (cost unknown), which is NOT the same as a
    # genuinely-free ``0.0``. A source that could not price the run (unknown model/rate, missing
    # token bucket) leaves this ``None`` and callers must render/aggregate it as "unknown", never $0.
    final_cost_usd: float | None = 0.0
    final_input_tokens: int = 0
    final_output_tokens: int = 0
    final_cache_tokens: int = 0             # cache_read + cache_creation (kept as their sum)
    final_cache_read_tokens: int = 0
    final_cache_creation_tokens: int = 0
    final_reasoning_tokens: int = 0         # 1.2: reasoning output (a SUBSET of final_output_tokens)
    #: Optional cost provenance (billing_mode/source/price snapshot). Present on sources that
    #: produce it (the Codex importer); ``None`` for older/other sources. See ``trajectory/cost.py``.
    cost: "dict | None" = None
    classifier_config: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "source": self.source,
            "model": self.model,
            "duration_s": self.duration_s,
            "num_rounds": self.num_rounds,
            "provisional": self.provisional,
            "final_cost_usd": self.final_cost_usd,
            "final_input_tokens": self.final_input_tokens,
            "final_output_tokens": self.final_output_tokens,
            "final_cache_tokens": self.final_cache_tokens,
            "final_cache_read_tokens": self.final_cache_read_tokens,
            "final_cache_creation_tokens": self.final_cache_creation_tokens,
            "final_reasoning_tokens": self.final_reasoning_tokens,
            "cost": self.cost,
            "classifier_config": self.classifier_config,
            "points": [asdict(p) for p in self.points],
            "bands": [asdict(b) for b in self.bands],
            "milestones": [asdict(m) for m in self.milestones],
            "checkpoints": [asdict(c) for c in self.checkpoints],
            "rounds": [asdict(r) for r in self.rounds],
            "inferences": [asdict(r) for r in self.inferences],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunTrajectory":
        return cls(
            run_id=d.get("run_id", ""),
            source=d.get("source", ""),
            model=d.get("model", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            duration_s=float(d.get("duration_s", 0.0)),
            num_rounds=int(d.get("num_rounds", 0)),
            provisional=bool(d.get("provisional", False)),
            # preserve None (unpriced) vs a number (priced). Absent key → 0.0 keeps pre-1.2 files,
            # which always wrote a number, loading exactly as before.
            final_cost_usd=(None if "final_cost_usd" in d and d["final_cost_usd"] is None
                            else float(d.get("final_cost_usd", 0.0))),
            final_input_tokens=int(d.get("final_input_tokens", 0)),
            final_output_tokens=int(d.get("final_output_tokens", 0)),
            final_cache_tokens=int(d.get("final_cache_tokens", 0)),
            final_cache_read_tokens=int(d.get("final_cache_read_tokens", 0)),
            final_cache_creation_tokens=int(d.get("final_cache_creation_tokens", 0)),
            final_reasoning_tokens=int(d.get("final_reasoning_tokens", 0)),
            cost=d.get("cost", None),
            classifier_config=d.get("classifier_config", {}) or {},
            points=[TrajectoryPoint(**p) for p in d.get("points", [])],
            bands=[ActivityBand(**b) for b in d.get("bands", [])],
            milestones=[TestMilestone(**m) for m in d.get("milestones", [])],
            # .get with a default: a v1.0 trajectory has no checkpoints key and must still load.
            checkpoints=[Checkpoint(**c) for c in d.get("checkpoints", [])],
            rounds=[RoundBoundary(**r) for r in d.get("rounds", [])],
            # 1.2 and older files have no inference collection and load unchanged.
            inferences=[InferenceRecord(**r) for r in d.get("inferences", [])],
        )

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return p

    @classmethod
    def from_json(cls, path: str | Path) -> "RunTrajectory":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_run_dir(cls, run_path: str | Path) -> "RunTrajectory":
        """Reconstruct from a canonical aet run dir (``metrics/trajectory.json`` or ``logs/``).

        Prefers the fast-path artifact; falls back to replaying ``logs/`` events + step-metrics.
        The ``logs/`` reconstruction lives in ``aet.trajectory.recording`` (imported lazily to
        keep this module import-light)."""
        run_path = Path(run_path)
        fast = run_path / "metrics" / "trajectory.json"
        if fast.is_file():
            return cls.from_json(fast)
        from aet.trajectory.recording import trajectory_from_logs
        return trajectory_from_logs(run_path)

    # ------------------------------------------------------------------ derived views
    def token_series(self) -> dict[str, list[float]]:
        """Plain-list series for plotting: t (minutes) + cumulative token/spend curves."""
        return {
            "t": [p.t_s / 60.0 for p in self.points],
            "input": [p.cum_input_tokens for p in self.points],
            "output": [p.cum_output_tokens for p in self.points],
            "cache": [p.cum_cache_tokens for p in self.points],
            "cache_read": [p.cum_cache_read_tokens for p in self.points],
            "cache_creation": [p.cum_cache_creation_tokens for p in self.points],
            "reasoning": [p.cum_reasoning_tokens for p in self.points],
            "total": [p.cum_total_tokens for p in self.points],
            "spend": [p.cum_cost_usd for p in self.points],
        }

    def milestone_series(self) -> list[tuple[float, int]]:
        """(minute, n_passed) pairs, sorted by time — the gold test-pass steps."""
        return sorted(((m.t_s / 60.0, m.n_passed) for m in self.milestones), key=lambda x: x[0])

    def time_to(self, kind: str, scope: str = "all") -> float | None:
        """Seconds to the FIRST time this checkpoint was reached, or ``None`` if it never was.

        ``None`` is load-bearing and must survive to the caller: a run that never elaborated has no
        time-to-elaboration, and substituting the run duration would silently convert "never got
        there" into "got there at the very end" — which is the difference between a censored
        observation and a slow one."""
        hits = [c.t_s for c in self.checkpoints if c.kind == kind and c.scope == scope]
        return min(hits) if hits else None

    def checkpoint_ladder(self, scope: str = "all") -> list[tuple[str, float | None]]:
        """``(kind, t_s | None)`` in CHECKPOINT_KINDS order — the parse → elaborate → pass figure.

        Unreached checkpoints are kept in the list with ``None`` rather than dropped, so a run that
        stalled at parse and a run that was never measured render differently."""
        return [(k, self.time_to(k, scope)) for k in CHECKPOINT_KINDS]

    def per_agent_rollup(self) -> dict[str, dict]:
        """Measured token/cost totals and derived activity share per agent."""
        out: dict[str, dict] = {}
        for rec in self.inferences:
            key = rec.agent_id or "unattributed"
            row = out.setdefault(key, {
                "agent_id": key, "parent_agent_id": rec.parent_agent_id,
                "requests": 0, "retries": 0, "input_tokens": 0,
                "output_tokens": 0, "cache_read_tokens": 0,
                "cache_write_tokens": 0, "reasoning_tokens": 0,
                "cost_usd": 0.0, "unpriced_requests": 0, "active_s": 0.0,
            })
            row["requests"] += 1
            row["retries"] += int(rec.retry)
            row["input_tokens"] += rec.input_tokens
            row["output_tokens"] += rec.output_tokens
            row["cache_read_tokens"] += rec.cache_read_tokens
            row["cache_write_tokens"] += rec.cache_write_tokens
            row["reasoning_tokens"] += rec.reasoning_tokens
            row["active_s"] += rec.duration_s
            if rec.cost_usd is None:
                row["unpriced_requests"] += 1
            else:
                row["cost_usd"] += rec.cost_usd
        total = sum(float(row["active_s"]) for row in out.values())
        for row in out.values():
            row["activity_share"] = (row["active_s"] / total) if total else 0.0
        return out

    # ------------------------------------------------------------------ tests-over-time
    def tests_total(self) -> int | None:
        """The '/N' denominator for tests-passing (e.g. 20). Milestones win, else round verdicts.

        ``None`` when this run recorded no test count at all — which is not the same as a run that
        scored 0 out of something. This used to return a literal ``20`` in that case, and every
        caller rendered it: a run from a source that records no tests (an LLM-call trajectory, say)
        drew a chip reading ``final 0/20``, asserting a denominator nothing had measured. Callers
        must omit the fraction rather than substitute a number of their own.
        """
        for m in self.milestones:
            if m.n_total:
                return int(m.n_total)
        for r in reversed(self.rounds):
            if r.n_total:
                return int(r.n_total)
        return None

    def tests_steps(self) -> tuple[list[float], list[int]]:
        """(minutes, n_passing) non-decreasing step series over the run.

        Uses the self-check milestones when present (the intermediate 13→17→20 rises); falls back to
        per-round QA verdicts at round-end times. Returns an empty-progression ([0], [0] → duration)
        when neither exists — so tests-over-time views degrade gracefully to 'no progression'."""
        pts = self.milestone_series()   # (minute, count), sorted
        if not pts:
            pts = [(r.t_end_s / 60.0, int(r.n_passed))
                   for r in self.rounds if r.n_passed is not None]
        xs: list[float] = [0.0]
        ys: list[int] = [0]
        last = 0
        for x, c in pts:
            c = max(int(c), last)       # enforce non-decreasing
            xs.append(float(x))
            ys.append(c)
            last = c
        xs.append(self.duration_s / 60.0)
        ys.append(ys[-1])
        return xs, ys

    def final_tests(self) -> int:
        """Best tests-passing the run reached (max milestone, else last round verdict, else 0)."""
        if self.milestones:
            return int(max(m.n_passed for m in self.milestones))
        passed = [r.n_passed for r in self.rounds if r.n_passed is not None]
        return int(max(passed)) if passed else 0
