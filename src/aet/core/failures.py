"""Hardware benchmark failure taxonomy."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum


class FailureCategory(str, Enum):
    """Taxonomy for Verilator/oracle failure modes."""
    SYNTAX_ERROR        = "syntax_error"
    ELABORATION_ERROR   = "elaboration_error"
    SIMULATION_MISMATCH = "simulation_mismatch"
    TESTBENCH_TIMEOUT   = "testbench_timeout"
    TOOL_CRASH          = "tool_crash"
    TAINT               = "taint"
    RUNNER_CRASH        = "runner_crash"
    UNKNOWN             = "unknown"


@dataclass
class FailureRecord:
    category: FailureCategory
    detail: str
    failure_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    iteration: int | None = None
    file: str | None = None
    line: int | None = None
    raw_output: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FailureRecord":
        data = dict(data)
        data["category"] = FailureCategory(data.get("category", "unknown"))
        data.pop("failure_id", None)
        return cls(**data)
