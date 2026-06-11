"""Tests for aet.core.artifact_store."""
import json
import pytest
from pathlib import Path
from aet.core.artifact_store import ArtifactStore, ArtifactOrigin, ArtifactRecord


def _write(tmp_path, name, content="hello"):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_record_creates_manifest(tmp_path):
    store = ArtifactStore(tmp_path, run_id="r1")
    p = _write(tmp_path, "score.json", '{"testbench_pass": 1}')
    store.record(p, ArtifactOrigin.ORACLE_OUTPUT)
    assert (tmp_path / "artifact_manifest.json").exists()


def test_record_computes_sha256(tmp_path):
    store = ArtifactStore(tmp_path, run_id="r1")
    p = _write(tmp_path, "dut.sv", "module foo(); endmodule")
    rec = store.record(p, ArtifactOrigin.AGENT_WRITTEN)
    assert rec.sha256 is not None
    assert len(rec.sha256) == 64


def test_record_tracks_size(tmp_path):
    store = ArtifactStore(tmp_path, run_id="r1")
    content = "x" * 1000
    p = _write(tmp_path, "large.txt", content)
    rec = store.record(p, ArtifactOrigin.HARNESS_COPIED)
    assert rec.size_bytes == 1000


def test_manifest_json_serializable(tmp_path):
    store = ArtifactStore(tmp_path, run_id="r1")
    p = _write(tmp_path, "f.txt")
    store.record(p, ArtifactOrigin.GENERATED)
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert len(manifest["artifacts"]) == 1


def test_find_by_origin(tmp_path):
    store = ArtifactStore(tmp_path, run_id="r1")
    p1 = _write(tmp_path, "a.txt")
    p2 = _write(tmp_path, "b.txt")
    store.record(p1, ArtifactOrigin.AGENT_WRITTEN)
    store.record(p2, ArtifactOrigin.ORACLE_OUTPUT)
    agents = store.find_by_origin(ArtifactOrigin.AGENT_WRITTEN)
    oracles = store.find_by_origin(ArtifactOrigin.ORACLE_OUTPUT)
    assert len(agents) == 1
    assert len(oracles) == 1


def test_find_by_sha256(tmp_path):
    store = ArtifactStore(tmp_path, run_id="r1")
    p = _write(tmp_path, "dup.txt", "same content")
    rec = store.record(p, ArtifactOrigin.USER_PROVIDED)
    found = store.find_by_sha256(rec.sha256)
    assert len(found) == 1
    assert found[0].path == str(p)


def test_protected_flag(tmp_path):
    store = ArtifactStore(tmp_path, run_id="r1")
    p = _write(tmp_path, "tb.sv")
    rec = store.record(p, ArtifactOrigin.USER_PROVIDED, protected=True)
    assert rec.protected is True
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert manifest["artifacts"][0]["protected"] is True


def test_origin_is_str_in_manifest(tmp_path):
    store = ArtifactStore(tmp_path, run_id="r1")
    p = _write(tmp_path, "x.txt")
    store.record(p, ArtifactOrigin.GENERATED)
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert manifest["artifacts"][0]["origin"] == "generated"


def test_len(tmp_path):
    store = ArtifactStore(tmp_path, run_id="r1")
    assert len(store) == 0
    store.record(_write(tmp_path, "a.txt"), ArtifactOrigin.GENERATED)
    store.record(_write(tmp_path, "b.txt"), ArtifactOrigin.GENERATED)
    assert len(store) == 2
