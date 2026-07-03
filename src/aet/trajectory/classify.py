"""Pluggable activity classifier — maps a tool call to an activity category + weight.

The whole point of keeping this configurable is that ``aet`` is repo-agnostic: the rule that
"a Bash call running verilator is a long *tool-wait*, not ordinary shell" is Gemmini-specific
knowledge and must live in a **config object**, never in harness source. A caller (a suite, a
CLI flag, a JSON file) supplies the rules; the core only knows the generic categories.

Categories (extensible):
  think · read · write · bash · tool   (``tool`` = a long external wait, e.g. a simulator run)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# Default per-category weights. ``tool`` is weighted heavily so a single long simulator wait
# reads as a wide band in the activity share (matches the oscar-merlin reference weighting).
DEFAULT_WEIGHTS: dict[str, float] = {
    "think": 1.0,
    "read": 1.0,
    "write": 1.4,
    "bash": 1.2,
    "tool": 28.0,
}


@dataclass
class LongWaitRule:
    """Reclassify a tool call as a long wait when its input contains any marker string.

    e.g. ``LongWaitRule("Bash", ["--sim verilator", '"verilator"'])`` turns a shell call that
    shells out to a cycle-accurate simulator into a ``tool`` band.
    """

    tool: str
    contains_any: list[str]
    category: str = "tool"
    weight: float = 28.0

    def matches(self, tool_name: str, tool_input: dict) -> bool:
        if tool_name.lower() != self.tool.lower():
            return False
        try:
            blob = json.dumps(tool_input, ensure_ascii=False)
        except Exception:
            blob = str(tool_input)
        return any(marker in blob for marker in self.contains_any)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "contains_any": list(self.contains_any),
            "category": self.category,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LongWaitRule":
        return cls(
            tool=d["tool"],
            contains_any=list(d.get("contains_any", [])),
            category=d.get("category", "tool"),
            weight=float(d.get("weight", 28.0)),
        )


@dataclass
class ActivityConfig:
    """Everything the classifier needs — tool→category maps, long-wait rules, weights."""

    read_tools: frozenset[str] = frozenset({"read"})
    write_tools: frozenset[str] = frozenset({"edit", "write", "notebookedit", "multiedit"})
    long_wait_rules: list[LongWaitRule] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def to_dict(self) -> dict:
        return {
            "read_tools": sorted(self.read_tools),
            "write_tools": sorted(self.write_tools),
            "long_wait_rules": [r.to_dict() for r in self.long_wait_rules],
            "weights": dict(self.weights),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActivityConfig":
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(d.get("weights", {}) or {})
        return cls(
            read_tools=frozenset(t.lower() for t in d.get("read_tools", ["read"])),
            write_tools=frozenset(
                t.lower() for t in d.get("write_tools",
                                        ["edit", "write", "notebookedit", "multiedit"])),
            long_wait_rules=[LongWaitRule.from_dict(r) for r in d.get("long_wait_rules", [])],
            weights=weights,
        )

    @classmethod
    def from_json_file(cls, path: str) -> "ActivityConfig":
        return cls.from_dict(json.loads(open(path).read()))


class ActivityClassifier:
    """Classify a tool call into ``(category, weight)`` using an :class:`ActivityConfig`."""

    def __init__(self, config: ActivityConfig | None = None) -> None:
        self.config = config or ActivityConfig()

    def weight_for(self, category: str) -> float:
        return float(self.config.weights.get(category, 1.0))

    def classify(self, tool_name: str, tool_input: dict | None = None) -> tuple[str, float]:
        """Return the activity category + weight for a single tool call.

        Long-wait rules win first (a verilator Bash is ``tool``, not ``bash``); then the
        read/write tool sets; everything else is ordinary ``bash``/generic tool use.
        """
        tool_input = tool_input or {}
        for rule in self.config.long_wait_rules:
            if rule.matches(tool_name, tool_input):
                return rule.category, rule.weight
        low = tool_name.lower()
        if low in self.config.read_tools:
            return "read", self.weight_for("read")
        if low in self.config.write_tools:
            return "write", self.weight_for("write")
        # Bash and any other/unknown tool default to the shell/bash lane.
        return "bash", self.weight_for("bash")


# --------------------------------------------------------------------- factory configs
def capsule_bench_config(circt: bool = False) -> ActivityConfig:
    """The Gemmini capsule-bench activity rules — supplied as DATA, so aet core stays generic.

    A Bash call that invokes the verilator simulator (or, in CIRCT runs, the RTL-facts /
    ISA-gen tooling) is a long external wait → the ``tool`` lane.
    """
    rules = [LongWaitRule("Bash", ["--sim verilator", '"verilator"'])]
    if circt:
        rules.append(LongWaitRule("Bash", ["rtl_check", "gen_isa", "rtl_facts", "facts.json"]))
    return ActivityConfig(long_wait_rules=rules)


def spec_to_rtl_config() -> ActivityConfig:
    """abc-testing spec-to-rtl / rtl-to-spec activity rules — a Bash call that runs the Verilator
    testbench oracle (``./run.sh``, ``run_test.py``, ``verilator``) is a long external wait → the
    ``tool`` lane, so it reads as the distinct 'tool wait' band (matching the reference figures)."""
    return ActivityConfig(long_wait_rules=[
        # ./run.sh / run_test.py / verilator are the obvious oracle invocations; the build/sim
        # scaffolding (verilator's obj_dir/sim_build, the *_test build dirs, make/cmake, and a
        # /usr/bin/time-wrapped build) is *also* a long external wait — without these markers a
        # clean rebuild that never literally prints "verilator" lands in the ordinary `bash` lane
        # and shows up as a giant green block instead of the red tool-wait it really is.
        LongWaitRule("Bash", [
            "run.sh", "run_test.py", "verilator", "obj_dir", "sim_build",
            "make ", "cmake", "/usr/bin/time",
        ], category="tool"),
    ])
