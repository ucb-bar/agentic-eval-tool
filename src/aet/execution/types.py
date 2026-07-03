from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ExecutionSpec:
    backend: str = "local"
    max_concurrency: int = 1
