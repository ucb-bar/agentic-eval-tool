"""aet exception hierarchy — all errors derive from :class:`AetError`."""


class AetError(Exception):
    """Base class for every aet-raised error."""


class SuiteNotFoundError(AetError):
    """Raised when a requested suite is not in the registry."""


class RunAlreadyExistsError(AetError):
    """Raised when a run directory already exists and ``--force`` was not given."""


class ValidationError(AetError):
    """Raised when a run's outputs fail validation."""


class ExecutionError(AetError):
    """Raised when an execution backend fails to run a task."""


class TemplateError(AetError):
    """Raised when a project template cannot be found or instantiated."""
