"""Run-report writers (aet.tracking.reports) — the report shapes, tested in isolation."""
import json

from aet.tracking import reports


def test_run_record_shape(tmp_path):
    p = reports.write_run_record(tmp_path, run_id="r1", project="proj", suite="s", target="t",
                                 method="m", seed=3, mode="local", extra={"repo_sha": "abc"})
    assert p == tmp_path / "run_record.json"
    d = json.loads(p.read_text())
    assert d["run_id"] == "r1" and d["seed"] == 3 and d["tracking_mode"] == "local"
    assert d["repo_sha"] == "abc" and "created_at" in d


def test_summary_metrics_goes_under_metrics(tmp_path):
    p = reports.write_summary_metrics(tmp_path, run_id="r1", project="p", suite="s", method="m",
                                      seed=0, target="t", extra={"hw.functional_pass": 1})
    assert p == tmp_path / "metrics" / "summary_metrics.json"
    assert json.loads(p.read_text())["hw.functional_pass"] == 1


def test_eval_report_defaults_empty_lists(tmp_path):
    p = reports.write_eval_report(tmp_path, run_id="r1", tests=[{"test": "t", "passed": True}])
    d = json.loads(p.read_text())
    assert d["tests"][0]["passed"] is True
    assert d["contracts"] == [] and d["assertions"] == [] and d["coverage"] == []


def test_metrics_structured_sections(tmp_path):
    p = reports.write_metrics_structured(tmp_path, run_id="r1", cost={"usd": 1.0},
                                         quality={"pass": True}, process={"iters": 2})
    d = json.loads(p.read_text())
    assert d["cost"]["usd"] == 1.0 and d["quality"]["pass"] is True and d["process"]["iters"] == 2
