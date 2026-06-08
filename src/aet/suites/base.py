from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

class EvalSuite(ABC):
    """Base class for all evaluation suites."""

    @abstractmethod
    def init_run(self, spec, paths, logger) -> None:
        """Create suite-specific directory structure inside the run dir."""
        ...

    @abstractmethod
    def validate(self, spec, paths, logger) -> dict:
        """Run all validators. Return validation report dict. Must not crash on incomplete runs."""
        ...

    @abstractmethod
    def collect_metrics(self, spec, paths, logger) -> dict:
        """Build and write metrics files. Return summary dict."""
        ...

    @abstractmethod
    def compare(self, run_paths: list[Path], report_dir: Path, logger) -> None:
        """Aggregate runs into comparison reports."""
        ...
