from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable
from aet.core.run_spec import RunSpec

class ExecutionBackend(ABC):
    @abstractmethod
    def submit(self, spec: RunSpec, fn: Callable) -> Any: ...

    @abstractmethod
    def run_many(self, specs: list[RunSpec], fn: Callable) -> list[Any]: ...
