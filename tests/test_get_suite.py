"""Tests for aet.suites.get_suite."""
import pytest

from aet.suites import get_suite
from aet.suites.default.suite import DefaultSuite
from aet.suites.targetgen.suite import TargetGenSuite
from aet.core.errors import SuiteNotFoundError


class TestGetSuite:
    def test_get_default_suite(self):
        suite = get_suite("default")
        assert isinstance(suite, DefaultSuite)

    def test_get_targetgen_suite(self):
        suite = get_suite("targetgen")
        assert isinstance(suite, TargetGenSuite)

    def test_unknown_suite_raises_suite_not_found(self):
        with pytest.raises(SuiteNotFoundError):
            get_suite("no_such_suite")

    def test_suite_not_found_error_is_aet_error(self):
        from aet.core.errors import AetError
        with pytest.raises(AetError):
            get_suite("nonexistent")

    def test_error_message_contains_suite_name(self):
        with pytest.raises(SuiteNotFoundError, match="bad_suite"):
            get_suite("bad_suite")

    def test_get_suite_returns_new_instance_each_time(self):
        s1 = get_suite("default")
        s2 = get_suite("default")
        assert s1 is not s2

    def test_default_suite_has_required_methods(self):
        suite = get_suite("default")
        assert callable(getattr(suite, "init_run", None))
        assert callable(getattr(suite, "validate", None))
        assert callable(getattr(suite, "collect_metrics", None))
        assert callable(getattr(suite, "compare", None))

    def test_targetgen_suite_has_required_methods(self):
        suite = get_suite("targetgen")
        assert callable(getattr(suite, "init_run", None))
        assert callable(getattr(suite, "validate", None))
        assert callable(getattr(suite, "collect_metrics", None))
        assert callable(getattr(suite, "compare", None))
