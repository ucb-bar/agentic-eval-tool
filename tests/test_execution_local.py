"""Tests for LocalExecutor."""
import pytest
from pathlib import Path

from aet.execution import LocalExecutor
from aet.core.run_spec import RunSpec


def _make_spec(seed=0):
    return RunSpec(project="p", suite="default", method="m", seed=seed)


class TestLocalExecutor:
    def test_run_many_sequential(self):
        executor = LocalExecutor()
        specs = [_make_spec(seed=i) for i in range(3)]
        results = executor.run_many(specs, lambda s: s.seed * 2)
        assert results == [0, 2, 4]

    def test_run_many_empty(self):
        executor = LocalExecutor()
        results = executor.run_many([], lambda s: s.seed)
        assert results == []

    def test_run_many_single(self):
        executor = LocalExecutor()
        results = executor.run_many([_make_spec(seed=7)], lambda s: s.seed)
        assert results == [7]

    def test_submit_single(self):
        executor = LocalExecutor()
        spec = _make_spec(seed=5)
        result = executor.submit(spec, lambda s: s.seed + 10)
        assert result == 15

    def test_run_many_preserves_order(self):
        executor = LocalExecutor()
        specs = [_make_spec(seed=i) for i in [3, 1, 4, 1, 5]]
        results = executor.run_many(specs, lambda s: s.seed)
        assert results == [3, 1, 4, 1, 5]

    def test_run_many_with_exception_propagates(self):
        executor = LocalExecutor()
        specs = [_make_spec(seed=0)]
        with pytest.raises(ValueError):
            executor.run_many(specs, lambda s: (_ for _ in ()).throw(ValueError("boom")))

    def test_run_many_with_identity_fn(self):
        executor = LocalExecutor()
        specs = [_make_spec(seed=i) for i in range(4)]
        results = executor.run_many(specs, lambda s: s)
        assert [r.seed for r in results] == [0, 1, 2, 3]
