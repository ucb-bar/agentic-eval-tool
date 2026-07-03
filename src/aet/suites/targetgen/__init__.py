"""TargetGen suite — validators re-exported for convenient access.

NOTE: this suite is Merlin/oscar-merlin-specific (it hard-codes that project's directory layout
and integration checks). It ships as a **bundled example** of a full aet suite, not as a
general-purpose validator. See ``docs/targetgen_suite.md``."""
from aet.suites.targetgen import (  # noqa: F401  (intentional re-exports)
    validate_schema,
    validate_evidence,
    validate_xdsl,
    validate_passes,
    validate_dialect_design,
    validate_runtime_mock,
    validate_merlin_integration,
)

__all__ = [
    "validate_schema", "validate_evidence", "validate_xdsl", "validate_passes",
    "validate_dialect_design", "validate_runtime_mock", "validate_merlin_integration",
]
