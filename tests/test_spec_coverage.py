"""Tests for spec-required additions: expanded failures, artifact fields, run_spec fields, new logger methods."""
import json
import pytest
from pathlib import Path
from aet.core.failures import FailureCategory, FailureRecord
from aet.core.artifact_store import ArtifactStore, ArtifactOrigin, ArtifactRecord
from aet.tracking.run_logger import EvalRunLogger


# ---------------------------------------------------------------------------
# FailureCategory — 19 spec-required categories
# ---------------------------------------------------------------------------

_SPEC_CATEGORIES = [
    "syntax_error", "elaboration_error", "interface_mismatch", "width_mismatch",
    "synthesis_failure", "reset_failure", "functional_mismatch", "numeric_mismatch",
    "hidden_test_failure", "protocol_violation", "timing_window_violation",
    "structural_invariant_violation", "forbidden_pattern", "coverage_gap",
    "timing_failure", "area_budget_failure", "power_budget_failure",
    "timeout", "agent_invalid_edit",
]

def test_failure_category_spec_complete():
    values = {fc.value for fc in FailureCategory}
    for cat in _SPEC_CATEGORIES:
        assert cat in values, f"Missing spec-required category: {cat}"

def test_failure_record_spec_fields():
    rec = FailureRecord(
        category=FailureCategory.PROTOCOL_VIOLATION,
        detail="handshake missing",
        contract_id="ctrl.phase_sequence",
        module="CTRL_FSM",
        signal="drain_en",
        test="pe_valid_compute_3x4",
        expected="drain_en=1",
        observed="drain_en=0",
        first_seen_iteration=2,
        resolved_iteration=5,
        likely_cause="off-by-one in counter",
        artifact_refs=["trace.json"],
    )
    d = rec.to_dict()
    assert d["contract_id"] == "ctrl.phase_sequence"
    assert d["module"] == "CTRL_FSM"
    assert d["expected"] == "drain_en=1"
    assert d["artifact_refs"] == ["trace.json"]

def test_failure_record_round_trip_new_fields():
    rec = FailureRecord(
        category=FailureCategory.STRUCTURAL_INVARIANT_VIOLATION,
        detail="forbidden mux",
        module="PE_MAC",
        likely_cause="agent used explicit mux",
    )
    rec2 = FailureRecord.from_dict(rec.to_dict())
    assert rec2.category == FailureCategory.STRUCTURAL_INVARIANT_VIOLATION
    assert rec2.module == "PE_MAC"
    assert rec2.likely_cause == "agent used explicit mux"


# ---------------------------------------------------------------------------
# ArtifactStore — kind, created_at_iteration, input_refs, line_count
# ---------------------------------------------------------------------------

def test_artifact_record_kind_and_iteration(tmp_path):
    f = tmp_path / "dut.sv"
    f.write_text("module dut(); endmodule\n")
    store = ArtifactStore(tmp_path, run_id="r1")
    rec = store.record(f, ArtifactOrigin.AGENT_WRITTEN,
                       kind="rtl", created_at_iteration=3,
                       input_refs=["prompt_003.txt"])
    assert rec.kind == "rtl"
    assert rec.created_at_iteration == 3
    assert rec.input_refs == ["prompt_003.txt"]
    assert rec.line_count == 1

def test_artifact_store_find_by_kind(tmp_path):
    f1 = tmp_path / "a.sv"; f1.write_text("x\n")
    f2 = tmp_path / "b.log"; f2.write_text("y\n")
    store = ArtifactStore(tmp_path, run_id="r1")
    store.record(f1, ArtifactOrigin.AGENT_WRITTEN, kind="rtl")
    store.record(f2, ArtifactOrigin.ORACLE_OUTPUT, kind="log")
    assert len(store.find_by_kind("rtl")) == 1
    assert len(store.find_by_kind("log")) == 1
    assert len(store.find_by_kind("tb")) == 0

def test_artifact_store_find_protected(tmp_path):
    f = tmp_path / "tb.sv"; f.write_text("tb\n")
    store = ArtifactStore(tmp_path, run_id="r1")
    store.record(f, ArtifactOrigin.PROTECTED_EVALUATOR, protected=True)
    assert len(store.find_protected()) == 1

def test_artifact_manifest_has_new_fields(tmp_path):
    f = tmp_path / "rtl.sv"; f.write_text("module m();\nendmodule\n")
    store = ArtifactStore(tmp_path, run_id="r1")
    store.record(f, ArtifactOrigin.AUTHORED, kind="rtl", created_at_iteration=1)
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    rec = manifest["artifacts"][0]
    assert rec["kind"] == "rtl"
    assert rec["created_at_iteration"] == 1
    assert rec["line_count"] == 2


# ---------------------------------------------------------------------------
# RunSpec — new optional fields
# ---------------------------------------------------------------------------

def test_run_spec_new_fields():
    from aet.core.run_spec import RunSpec
    spec = RunSpec(
        project="abc-testing", suite="hw", method="opus/xhigh",
        seed=0, target="debug-sv/trisc-sc/01",
        benchmark_level="B1",
        baseline_variant="D3",
        model_strategy="M0",
        time_budget_minutes=60,
        token_budget=500000,
        human_intervention_policy="none",
        machine_spec_version="sha256:abc",
        ir_version="sha256:def",
        scaffold_version="sha256:ghi",
        hidden_eval_version="sha256:jkl",
    )
    assert spec.benchmark_level == "B1"
    assert spec.time_budget_minutes == 60
    assert spec.human_intervention_policy == "none"
    assert spec.hidden_eval_version == "sha256:jkl"


# ---------------------------------------------------------------------------
# EvalRunLogger — new event methods
# ---------------------------------------------------------------------------

def _make_logger(tmp_path):
    return EvalRunLogger.start(
        run_id="spec_r1", run_path=tmp_path, tracking_mode="local",
        target="debug-sv/trisc-sc/01", method="opus/xhigh",
        seed=0, project="abc-testing", suite="hardware_benchmark",
    )

def _events(tmp_path):
    p = tmp_path / "logs" / "events.jsonl"
    return [json.loads(l) for l in p.read_text().strip().splitlines()]

def test_log_eval_start_end(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_eval_start("public", iteration=3)
    logger.log_eval_end(8, 10, score=0.8, eval_type="public")
    logger.finish("pass")
    evs = _events(tmp_path)
    assert any(e["event"] == "eval.start" for e in evs)
    assert any(e["event"] == "eval.end" for e in evs)
    end_ev = next(e for e in evs if e["event"] == "eval.end")
    assert end_ev["payload"]["pass_rate"] == pytest.approx(0.8)

def test_log_eval_stage_result(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_eval_stage_result("directed_tests", passed=False, score=18, max_score=25)
    logger.finish("fail")
    evs = _events(tmp_path)
    ev = next(e for e in evs if e["event"] == "eval.stage_result")
    assert ev["payload"]["stage"] == "directed_tests"
    assert ev["payload"]["score"] == 18

def test_log_eval_assertion_result(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_eval_assertion_result(
        "done_after_drain", passed=False, cycle=10,
        expected="done_o=1", observed="done_o=0",
        contract_id="ctrl.drain_complete",
    )
    logger.finish("fail")
    evs = _events(tmp_path)
    ev = next(e for e in evs if e["event"] == "eval.assertion_result")
    assert ev["payload"]["contract_id"] == "ctrl.drain_complete"
    assert ev["payload"]["cycle"] == 10

def test_log_eval_coverage_result(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_eval_coverage_result("every_fsm_state", covered=4, total=5, missing=["DONE"])
    logger.finish("pass")
    evs = _events(tmp_path)
    ev = next(e for e in evs if e["event"] == "eval.coverage_result")
    assert ev["payload"]["coverage_rate"] == pytest.approx(0.8)
    assert ev["payload"]["missing"] == ["DONE"]

def test_log_synth_start_and_ppa(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_synth_start(tool="yosys", target_library="NanGate45", clock_target_ns=5.0)
    logger.log_ppa(area_um2=1843.0, cell_count=420, critical_path_ns=4.7,
                   slack_ns=0.3, power_uw=710.0, frequency_mhz=212.8,
                   tool="yosys", ppa_validity="functional_pass_required")
    logger.finish("pass")
    evs = _events(tmp_path)
    assert any(e["event"] == "synth.start" for e in evs)
    assert any(e["event"] == "ppa.report" for e in evs)
    ppa_ev = next(e for e in evs if e["event"] == "ppa.report")
    assert ppa_ev["payload"]["area_um2"] == 1843.0

def test_log_ppa_writes_metrics(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_ppa(area_um2=2000.0, slack_ns=-0.1)
    logger.finish("pass")
    p = tmp_path / "logs" / "metrics.jsonl"
    metrics = {json.loads(l)["name"]: json.loads(l)["value"]
               for l in p.read_text().strip().splitlines()}
    assert metrics["ppa.area_um2"] == 2000.0
    assert metrics["ppa.slack_ns"] == -0.1

def test_log_run_timeout(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_run_timeout(reason="wall_timeout", elapsed_s=3600.0)
    logger.finish("abort")
    evs = _events(tmp_path)
    ev = next(e for e in evs if e["event"] == "run.timeout")
    assert ev["payload"]["elapsed_s"] == 3600.0

def test_log_file_read_and_delete(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_file_read("spec/SPEC.md", sha256="abc123", size_bytes=4096)
    logger.log_file_delete("tmp/old.sv", sha256_before="def456")
    logger.finish("pass")
    evs = _events(tmp_path)
    assert any(e["event"] == "file.read" for e in evs)
    assert any(e["event"] == "file.delete" for e in evs)
    read_ev = next(e for e in evs if e["event"] == "file.read")
    assert read_ev["payload"]["sha256"] == "abc123"

def test_log_git_events(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_git_commit("deadbeef", message="fix: reset bug", files_changed=2, additions=10, deletions=3)
    logger.log_git_diff(files_changed=1, additions=5, deletions=2, ref_before="HEAD~1", ref_after="HEAD")
    logger.finish("pass")
    evs = _events(tmp_path)
    assert any(e["event"] == "git.commit" for e in evs)
    assert any(e["event"] == "git.diff" for e in evs)

def test_log_human_sub_types(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_human_clarification("What does valid mean?", "Valid means both inputs are non-zero")
    logger.log_human_patch("rtl/dut.sv", description="fix reset", iteration=4)
    logger.log_human_override("skipped hidden eval", why="evaluator bug", artifact_affected="eval_report.json")
    logger.finish("pass")
    evs = _events(tmp_path)
    assert any(e["event"] == "human.clarification" for e in evs)
    assert any(e["event"] == "human.patch" for e in evs)
    assert any(e["event"] == "human.override" for e in evs)

def test_write_eval_report(tmp_path):
    logger = _make_logger(tmp_path)
    tests = [
        {"test": "pe_valid_compute_3x4", "passed": False, "contract": "pe.mac_formula",
         "expected": "c_data_o=12", "observed": "c_data_o=0"},
        {"test": "pe_clear_clears_acc", "passed": True, "contract": "pe.clear_clears_acc"},
    ]
    contracts = [{"contract_id": "pe.mac_formula", "public_pass": False, "hidden_pass": False}]
    path = logger.write_eval_report(tests, contracts=contracts)
    logger.finish("fail")
    assert path.exists()
    report = json.loads(path.read_text())
    assert report["schema_version"] == "1.0"
    assert len(report["tests"]) == 2
    assert len(report["contracts"]) == 1
    assert report["contracts"][0]["contract_id"] == "pe.mac_formula"

def test_write_metrics_structured(tmp_path):
    logger = _make_logger(tmp_path)
    path = logger.write_metrics_structured(
        cost={"wall_clock_sec": 2714, "llm_input_tokens": 184230, "agent_iterations": 18},
        quality={"syntax_pass": True, "public_score": 92, "hidden_score": 71},
        process={"first_elaboration_iter": 3, "regression_count": 4,
                 "dominant_failure_category": "protocol_violation"},
    )
    logger.finish("pass")
    assert path.name == "metrics.json"
    m = json.loads(path.read_text())
    assert m["cost"]["agent_iterations"] == 18
    assert m["quality"]["hidden_score"] == 71
    assert m["process"]["dominant_failure_category"] == "protocol_violation"

def test_hardware_suite_init_run_creates_artifact_dirs(tmp_path):
    from aet.core.run_spec import RunSpec
    from aet.core.run_paths import RunPaths
    from aet.suites import get_suite
    spec = RunSpec(project="abc", suite="hardware_benchmark",
                   method="opus/xhigh", seed=0, target="t", run_id="r1")
    paths = RunPaths.from_spec(spec, "r1")
    suite = get_suite("hardware_benchmark")
    suite.init_run(spec, paths, None)
    rp = paths.run_path
    for subdir in ["artifacts/prompts", "artifacts/rtl", "artifacts/tb",
                   "artifacts/logs", "artifacts/synth", "artifacts/ppa",
                   "artifacts/waveforms", "snapshots/initial_repo",
                   "snapshots/final_repo", "hashes"]:
        assert (rp / subdir).is_dir(), f"Missing: {subdir}"
