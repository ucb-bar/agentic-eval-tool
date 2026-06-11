"""Structured failure taxonomy for hardware benchmark runs."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class FailureCategory(str, Enum):
    # --- Synthesis/elaboration ---
    SYNTAX_ERROR = "syntax_error"
    ELABORATION_ERROR = "elaboration_error"
    INTERFACE_MISMATCH = "interface_mismatch"
    WIDTH_MISMATCH = "width_mismatch"
    SYNTHESIS_FAILURE = "synthesis_failure"
    # --- Functional ---
    RESET_FAILURE = "reset_failure"
    FUNCTIONAL_MISMATCH = "functional_mismatch"
    NUMERIC_MISMATCH = "numeric_mismatch"
    HIDDEN_TEST_FAILURE = "hidden_test_failure"
    # --- Protocol / structural ---
    PROTOCOL_VIOLATION = "protocol_violation"
    TIMING_WINDOW_VIOLATION = "timing_window_violation"
    STRUCTURAL_INVARIANT_VIOLATION = "structural_invariant_violation"
    FORBIDDEN_PATTERN = "forbidden_pattern"
    # --- Coverage ---
    COVERAGE_GAP = "coverage_gap"
    # --- PPA ---
    TIMING_FAILURE = "timing_failure"
    AREA_BUDGET_FAILURE = "area_budget_failure"
    POWER_BUDGET_FAILURE = "power_budget_failure"
    # --- Process ---
    TIMEOUT = "timeout"
    AGENT_INVALID_EDIT = "agent_invalid_edit"
    TAINT = "taint"
    RUNNER_CRASH = "runner_crash"
    # --- Legacy aliases (kept for backward compatibility) ---
    SIMULATION_MISMATCH = "simulation_mismatch"
    TESTBENCH_TIMEOUT = "testbench_timeout"
    TOOL_CRASH = "tool_crash"
    UNKNOWN = "unknown"


@dataclass
class FailureRecord:
    category: FailureCategory
    detail: str
    failure_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    iteration: int | None = None
    file: str | None = None
    line: int | None = None
    raw_output: str = ""
    # Spec fields
    contract_id: str | None = None
    module: str | None = None
    signal: str | None = None
    test: str | None = None
    expected: str = ""
    observed: str = ""
    first_seen_iteration: int | None = None
    resolved_iteration: int | None = None
    likely_cause: str = ""
    artifact_refs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "failure_id": self.failure_id,
            "category": self.category.value,
            "detail": self.detail,
            "iteration": self.iteration,
            "file": self.file,
            "line": self.line,
            "raw_output": self.raw_output[:2000],
            "contract_id": self.contract_id,
            "module": self.module,
            "signal": self.signal,
            "test": self.test,
            "expected": self.expected,
            "observed": self.observed,
            "first_seen_iteration": self.first_seen_iteration,
            "resolved_iteration": self.resolved_iteration,
            "likely_cause": self.likely_cause,
            "artifact_refs": self.artifact_refs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FailureRecord":
        cat_val = data.get("category", "unknown")
        try:
            category = FailureCategory(cat_val)
        except ValueError:
            category = FailureCategory.UNKNOWN
        return cls(
            category=category,
            detail=data.get("detail", ""),
            failure_id=data.get("failure_id", uuid.uuid4().hex),
            iteration=data.get("iteration"),
            file=data.get("file"),
            line=data.get("line"),
            raw_output=data.get("raw_output", ""),
            contract_id=data.get("contract_id"),
            module=data.get("module"),
            signal=data.get("signal"),
            test=data.get("test"),
            expected=data.get("expected", ""),
            observed=data.get("observed", ""),
            first_seen_iteration=data.get("first_seen_iteration"),
            resolved_iteration=data.get("resolved_iteration"),
            likely_cause=data.get("likely_cause", ""),
            artifact_refs=data.get("artifact_refs", []),
        )
