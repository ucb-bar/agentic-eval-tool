from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ExecutionSpec:
    backend: str = "local"
    ray_address: str | None = None
    max_concurrency: int = 1
