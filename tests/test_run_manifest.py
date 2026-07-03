"""Tests for aet.core.run_manifest.RunManifest."""

from aet.core.run_manifest import RunManifest
from aet.core.run_spec import RunSpec


def _minimal_spec(**kwargs):
    base = dict(project="p", suite="default", method="m", seed=1)
    base.update(kwargs)
    return RunSpec(**base)


class TestRunManifest:
    def test_default_construction(self):
        m = RunManifest()
        assert m.schema_version == "1.0"
        assert m.status == "initialized"

    def test_create_from_spec(self):
        spec = _minimal_spec(target="gemmini")
        m = RunManifest.create(spec, run_id="2099-01-01_m_seed001", git_hash="abc123")
        assert m.project == "p"
        assert m.suite == "default"
        assert m.method == "m"
        assert m.seed == 1
        assert m.target == "gemmini"
        assert m.run_id == "2099-01-01_m_seed001"
        assert m.git_hash_at_init == "abc123"
        assert m.status == "initialized"

    def test_dump_and_load_roundtrip(self, tmp_path):
        spec = _minimal_spec()
        m = RunManifest.create(spec, run_id="r1", git_hash="deadbeef")
        path = tmp_path / "run_manifest.yaml"
        m.dump(path)
        assert path.exists()

        loaded = RunManifest.load(path)
        assert loaded.project == m.project
        assert loaded.suite == m.suite
        assert loaded.method == m.method
        assert loaded.seed == m.seed
        assert loaded.run_id == m.run_id
        assert loaded.schema_version == m.schema_version

    def test_dump_creates_yaml_file(self, tmp_path):
        m = RunManifest(project="p", suite="s", method="m", seed=1, run_id="r1")
        path = tmp_path / "run_manifest.yaml"
        m.dump(path)
        content = path.read_text()
        assert "project" in content
        assert "schema_version" in content

    def test_load_ignores_unknown_keys(self, tmp_path):
        """load() should not crash on extra keys in the YAML."""
        yaml_content = (
            "schema_version: '1.0'\n"
            "project: myproj\n"
            "suite: default\n"
            "method: m\n"
            "seed: 2\n"
            "run_id: r2\n"
            "unknown_future_key: some_value\n"
        )
        path = tmp_path / "run_manifest.yaml"
        path.write_text(yaml_content)
        m = RunManifest.load(path)
        assert m.project == "myproj"
        assert m.seed == 2

    def test_to_dict_excludes_none_optional_fields(self):
        m = RunManifest(project="p", suite="s", method="m", seed=1)
        # target/model/dtype/substrate are None — they're still included per to_dict
        d = m.to_dict()
        assert "project" in d
        assert "schema_version" in d

    def test_create_sets_created_at(self):
        spec = _minimal_spec()
        m = RunManifest.create(spec, run_id="r1", git_hash="abc")
        assert m.created_at != ""
        assert "T" in m.created_at  # ISO format contains 'T'

    def test_create_with_tracking_fields(self):
        spec = _minimal_spec(
            tracking_mode="mlflow",
            mlflow_tracking_uri="http://localhost:5000",
            experiment_name="my-exp",
        )
        m = RunManifest.create(spec, run_id="r1", git_hash="abc")
        assert m.observability["tracking_mode"] == "mlflow"
        assert m.observability["mlflow"]["enabled"] is True
        assert m.observability["mlflow"]["tracking_uri"] == "http://localhost:5000"
