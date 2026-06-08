"""Tests for aet.core.run_spec.RunSpec."""
import pytest
from pathlib import Path

from aet.core.run_spec import RunSpec


class TestRunSpec:
    def test_required_fields(self):
        spec = RunSpec(project="myp", suite="default", method="v0", seed=42)
        assert spec.project == "myp"
        assert spec.suite == "default"
        assert spec.method == "v0"
        assert spec.seed == 42

    def test_defaults(self):
        spec = RunSpec(project="p", suite="s", method="m", seed=1)
        assert spec.run_id is None
        assert spec.tracking_mode == "local"
        assert spec.target is None
        assert spec.model is None
        assert spec.dtype is None
        assert spec.substrate is None
        assert spec.execution == "local"
        assert spec.is_smoke_test is True
        assert spec.budget == "cheap_smoke"
        assert spec.promotion_flag is False
        assert spec.force is False
        assert spec.mlflow_tracking_uri is None
        assert spec.experiment_name is None
        assert spec.otel_endpoint is None
        assert spec.extra == {}

    def test_project_root_default_is_path(self):
        spec = RunSpec(project="p", suite="s", method="m", seed=0)
        assert isinstance(spec.project_root, Path)

    def test_project_root_can_be_set(self, tmp_path):
        spec = RunSpec(project="p", suite="s", method="m", seed=0,
                       project_root=tmp_path)
        assert spec.project_root == tmp_path

    def test_optional_fields(self):
        spec = RunSpec(
            project="p", suite="targetgen", method="v0", seed=3,
            target="gemmini", model="resnet50", dtype="int8",
            substrate="fpga", tracking_mode="mlflow",
            is_smoke_test=False, budget="full",
            promotion_flag=True,
        )
        assert spec.target == "gemmini"
        assert spec.model == "resnet50"
        assert spec.dtype == "int8"
        assert spec.substrate == "fpga"
        assert spec.tracking_mode == "mlflow"
        assert spec.is_smoke_test is False
        assert spec.budget == "full"
        assert spec.promotion_flag is True

    def test_extra_field_default_empty_dict(self):
        spec1 = RunSpec(project="p", suite="s", method="m", seed=1)
        spec2 = RunSpec(project="p", suite="s", method="m", seed=2)
        # Extra dicts should not be shared between instances
        spec1.extra["foo"] = "bar"
        assert "foo" not in spec2.extra
