from __future__ import annotations
from aet.core.errors import SuiteNotFoundError

def get_suite(name: str):
    """Return an instantiated EvalSuite for the given name."""
    from aet.suites.default.suite import DefaultSuite
    from aet.suites.targetgen.suite import TargetGenSuite
    from aet.suites.hardware_benchmark.suite import HardwareBenchmarkSuite

    SUITE_REGISTRY = {
        "default": DefaultSuite,
        "targetgen": TargetGenSuite,
        "hardware_benchmark": HardwareBenchmarkSuite,
        "hardware-benchmark": HardwareBenchmarkSuite,
    }
    cls = SUITE_REGISTRY.get(name)
    if cls is None:
        raise SuiteNotFoundError(
            f"Unknown suite: {name!r}. Available: {sorted(SUITE_REGISTRY)}"
        )
    return cls()
