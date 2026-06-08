from __future__ import annotations
from typing import Any, Callable
from aet.core.run_spec import RunSpec
from aet.execution.base import ExecutionBackend

class LocalExecutor(ExecutionBackend):
    """Runs experiment specs sequentially in-process."""

    def submit(self, spec: RunSpec, fn: Callable) -> Any:
        return fn(spec)

    def run_many(self, specs: list[RunSpec], fn: Callable) -> list[Any]:
        return [fn(spec) for spec in specs]
