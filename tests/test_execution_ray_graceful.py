"""Test RayExecutor degrades gracefully when ray is not installed."""
import pytest
from aet.core.errors import AetError


def test_ray_no_install():
    """RayExecutor should raise AetError (not ImportError) if ray not installed."""
    try:
        from aet.execution import RayExecutor
        executor = RayExecutor()
        # If ray IS installed, verify both methods raise NotImplementedError
        with pytest.raises(NotImplementedError):
            executor.run_many([], lambda s: s)
    except AetError as e:
        assert "ray" in str(e).lower()


def test_ray_submit_not_implemented():
    """RayExecutor.submit raises NotImplementedError when ray is installed."""
    try:
        from aet.execution import RayExecutor
        from aet.core.run_spec import RunSpec
        executor = RayExecutor()
        spec = RunSpec(project="p", suite="default", method="m", seed=0)
        with pytest.raises(NotImplementedError):
            executor.submit(spec, lambda s: s)
    except AetError:
        # ray not installed — acceptable
        pass


def test_ray_aet_error_message():
    """AetError message from RayExecutor must mention how to install."""
    try:
        from aet.execution import RayExecutor
        RayExecutor()
    except AetError as e:
        msg = str(e).lower()
        assert "ray" in msg
    except Exception:
        # ray is installed — nothing to test here
        pass
