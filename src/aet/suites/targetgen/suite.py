from contextlib import nullcontext
from pathlib import Path
from aet.suites.base import EvalSuite


class TargetGenSuite(EvalSuite):
    def init_run(self, spec, paths, logger):
        # Create generated/<target>-mlir/ dir
        # Create contracts/ dir
        # Write README.md in generated/<target>-mlir/
        target = spec.target or "unknown"
        generated_dir = paths.generated / f"{target}-mlir"
        generated_dir.mkdir(parents=True, exist_ok=True)
        (generated_dir / "README.md").write_text(
            f"# Generated artifacts for {target}\n\n"
            f"Run: {spec.run_id}\nMethod: {spec.method}\nSeed: {spec.seed}\n"
        )
        paths.contracts.mkdir(parents=True, exist_ok=True)
        paths.logs.mkdir(parents=True, exist_ok=True)
        paths.metrics.mkdir(parents=True, exist_ok=True)
        if logger:
            logger.log_event("init_run.completed", {
                "run_id": spec.run_id,
                "target": target,
                "generated_dir": str(generated_dir),
            })

    def validate(self, spec, paths, logger):
        """Run all targetgen validators. Returns validation report dict. Never crashes."""
        import json
        from datetime import datetime, timezone

        from aet.suites.targetgen import (
            validate_schema, validate_evidence, validate_xdsl,
            validate_passes, validate_dialect_design,
            validate_runtime_mock, validate_merlin_integration,
        )

        _VALIDATOR_MAP = {
            "schema": validate_schema,
            "evidence": validate_evidence,
            "xdsl": validate_xdsl,
            "passes": validate_passes,
            "dialect_design": validate_dialect_design,
            "runtime_mock": validate_runtime_mock,
            "merlin_integration": validate_merlin_integration,
        }

        report = {
            "schema_version": "1.0",
            "run_id": spec.run_id,
            "suite": "targetgen",
            "validated_at": datetime.now(tz=timezone.utc).isoformat(),
            "validators": {},
            "overall": "pass",
            "total_errors": 0,
            "total_warnings": 0,
        }

        # Load manifest for validators
        manifest_path = paths.run_path / "run_manifest.yaml"
        try:
            from aet.core.yaml_utils import load_yaml
            manifest = load_yaml(manifest_path)
        except Exception:
            manifest = {}

        project_root = spec.project_root if spec.project_root != Path() else paths.run_path.parent.parent.parent

        import inspect
        for i, (name, module) in enumerate(_VALIDATOR_MAP.items()):
            span_cm = (
                logger.start_tool_span(f"validate_{name}", validator_name=name)
                if logger else nullcontext()
            )
            with span_cm:
                try:
                    sig = inspect.signature(module.run)
                    if "project_root" in sig.parameters:
                        result = module.run(paths.run_path, manifest, project_root=project_root)
                    else:
                        result = module.run(paths.run_path, manifest)
                    report["validators"][name] = result
                    if result.get("errors"):
                        report["total_errors"] += len(result["errors"])
                    if result.get("warnings"):
                        report["total_warnings"] += len(result.get("warnings", []))
                    if logger:
                        passed = len(result.get("errors", [])) == 0
                        logger.log_metric_step(f"validator.{name}.passed", 1.0 if passed else 0.0, step=i)
                        logger.log_metric_step("cumulative_errors", float(report["total_errors"]), step=i)
                        logger.log_event(f"validation.{name}.completed", {
                            "errors": result.get("errors", []),
                            "warnings": result.get("warnings", []),
                        })
                except Exception as exc:
                    report["validators"][name] = {"errors": [str(exc)], "warnings": []}
                    report["total_errors"] += 1
                    if logger:
                        logger.log_metric_step(f"validator.{name}.passed", 0.0, step=i)
                        logger.log_metric_step("cumulative_errors", float(report["total_errors"]), step=i)
                        logger.log_event(f"validation.{name}.error", {"error": str(exc)})

        if report["total_errors"] > 0:
            report["overall"] = "fail"
        elif report["total_warnings"] > 5:
            report["overall"] = "partial"

        if logger:
            score = 1.0 if report["overall"] == "pass" else 0.0
            logger.log_evaluation_result("validation_overall", score, report["overall"])
            logger.log_metric("total_errors", report["total_errors"])
            logger.log_metric("total_warnings", report["total_warnings"])

        try:
            paths.run_path.mkdir(parents=True, exist_ok=True)
            report_path = paths.run_path / "validation_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            if logger:
                logger.log_artifact(report_path)
        except Exception:
            pass

        if logger:
            logger.log_event("validation.completed", {
                "overall": report["overall"],
                "errors": report["total_errors"],
            })

        return report

    def collect_metrics(self, spec, paths, logger):
        import json
        from aet.suites.targetgen.collect_metrics import build_summary
        from aet.core.yaml_utils import load_yaml

        with (logger.start_tool_span("collect_metrics") if logger else nullcontext()):
            manifest = load_yaml(paths.run_path / "run_manifest.yaml")
            validator_results = {}
            report_path = paths.run_path / "validation_report.json"
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text())
                    validator_results = report.get("validators", {})
                except Exception:
                    pass

            # arch_rules: empty for now (rules are tracked in validation_report)
            summary = build_summary(manifest, validator_results, arch_rules=[])
            paths.metrics.mkdir(parents=True, exist_ok=True)
            (paths.metrics / "summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
            return summary

    def compare(self, run_paths, report_dir, logger):
        from aet.suites.targetgen.compare import run as compare_run
        # run_paths is list of run dirs; find project root and suite
        if run_paths:
            # Navigate up: run_path = project_root/runs/suite/run_id
            first = run_paths[0]
            suite_dir = first.parent
            runs_dir = suite_dir.parent
            project_root = runs_dir.parent
            compare_run(project_root, "targetgen", report_dir)
