"""aet.viz — optional plotting layer (requires the ``[viz]`` extra: matplotlib + numpy).

Import-light on purpose: importing this package never pulls in matplotlib. The plot functions
live in ``aet.viz.trajectory_plot`` and raise a friendly, actionable ImportError if the extra is
absent. Use :func:`require_viz` to check availability up front.
"""
from __future__ import annotations


def require_viz() -> None:
    """Raise a friendly ImportError if the visualization extra is not installed."""
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "aet visualization requires matplotlib and numpy. "
            "Install with:  pip install 'aet[viz]'"
        ) from e


def __getattr__(name: str):
    # lazy access: `from aet.viz import plot_trajectory` works only when [viz] is present
    if name in ("plot_trajectory", "plot_comparison", "activity_share"):
        from aet.viz import trajectory_plot
        return getattr(trajectory_plot, name)
    raise AttributeError(f"module 'aet.viz' has no attribute {name!r}")


__all__ = ["require_viz", "plot_trajectory", "plot_comparison", "activity_share"]
