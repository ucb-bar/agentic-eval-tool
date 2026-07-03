from __future__ import annotations
from typing import Any, Callable
from aet.core.errors import AetError
from aet.core.run_spec import RunSpec
from aet.execution.base import ExecutionBackend

class RayExecutor(ExecutionBackend):
    """Distributed execution via Ray. Skeleton implementation — local mode is fully supported."""

    def __init__(self, ray_address: str | None = None) -> None:
        try:
            import ray  # noqa: F401
        except ImportError:
            raise AetError(
                "Ray not installed. Install with: pip install 'aet[ray]'\n"
                "For now, use --execution local"
            )
        self._ray_address = ray_address

    def submit(self, spec: RunSpec, fn: Callable) -> Any:
        raise NotImplementedError(
            "Ray execution backend is not implemented. Use --execution local."
        )

    def run_many(self, specs: list[RunSpec], fn: Callable) -> list[Any]:
        raise NotImplementedError(
            "Ray execution backend is not implemented. Use --execution local."
        )
