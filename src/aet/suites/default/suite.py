from __future__ import annotations
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from aet.suites.base import EvalSuite
from aet.core.metrics import _NA, coerce_na


class DefaultSuite(EvalSuite):
    """General-purpose evaluation suite.

    Resilient by design: incomplete or empty runs produce partial/fail
    status rather than exceptions.
    """

    # ------------------------------------------------------------------
    # init_run
    # ------------------------------------------------------------------

    def init_run(self, spec, paths, logger) -> None:
        """Create generated/ directory and a README explaining the run."""
        generated = paths.generated
        generated.mkdir(parents=True, exist_ok=True)

        readme = generated / "README.md"
        timestamp = datetime.now(timezone.utc).isoformat()
        lines = [
            f"# Run: {spec.run_id}",
            "",
            f"- **Project**: {spec.project}",
            f"- **Suite**: {spec.suite}",
            f"- **Method**: {spec.method}",
            f"- **Seed**: {spec.seed}",
            f"- **Started**: {timestamp}",
            f"- **Budget**: {spec.budget}",
            f"- **Smoke test**: {spec.is_smoke_test}",
        ]
        if spec.target:
            lines.append(f"- **Target**: {spec.target}")
        if spec.model:
            lines.append(f"- **Model**: {spec.model}")
        if spec.dtype:
            lines.append(f"- **Dtype**: {spec.dtype}")
        if spec.substrate:
            lines.append(f"- **Substrate**: {spec.substrate}")
        lines += [
            "",
            "Artifacts produced by the run will appear alongside this file.",
        ]
        readme.write_text("\n".join(lines) + "\n")
        if logger and hasattr(logger, "log_event"):
            logger.log_event("init_run.completed", {"generated": str(generated)})

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------

    def validate(self, spec, paths, logger) -> dict:
        """Validate run outputs. Never raises; returns a report dict."""
        errors: list[str] = []
        status = "pass"

        try:
            generated = paths.generated
            if not generated.exists():
                errors.append("generated/ directory missing")
                status = "fail"
            else:
                contents = [p for p in generated.iterdir() if p.name != "README.md"]
                if not contents:
                    status = "partial"
                    errors.append("generated/ is empty (no artifacts beyond README)")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"unexpected error during validation: {exc}")
            status = "fail"

        report = {
            "run_id": spec.run_id,
            "suite": spec.suite,
            "method": spec.method,
            "status": status,
            "errors": errors,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write validation report
        try:
            paths.run_path.mkdir(parents=True, exist_ok=True)
            report_path = paths.run_path / "validation_report.json"
            report_path.write_text(json.dumps(report, indent=2))
            if logger and hasattr(logger, "log_event"):
                logger.log_event("validation.completed", {"status": status, "errors": len(errors)})
        except Exception:
            pass

        # Write identity summary metrics
        try:
            paths.metrics.mkdir(parents=True, exist_ok=True)
            summary = {
                "run_id": spec.run_id,
                "suite": spec.suite,
                "method": spec.method,
                "seed": spec.seed,
                "target": coerce_na(spec.target),
                "model": coerce_na(spec.model),
                "dtype": coerce_na(spec.dtype),
                "substrate": coerce_na(spec.substrate),
                "budget": spec.budget,
                "validation_status": status,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            summary_path = paths.metrics / "summary_metrics.json"
            summary_path.write_text(json.dumps(summary, indent=2))
        except Exception:
            pass

        return report

    # ------------------------------------------------------------------
    # collect_metrics
    # ------------------------------------------------------------------

    def collect_metrics(self, spec, paths, logger) -> dict:
        """Read validator output and return a summary dict."""
        summary_path = paths.metrics / "summary_metrics.json"
        if summary_path.exists():
            try:
                return json.loads(summary_path.read_text())
            except Exception:
                pass

        # Fallback: minimal identity record
        fallback = {
            "run_id": spec.run_id,
            "suite": spec.suite,
            "method": spec.method,
            "seed": spec.seed,
            "target": coerce_na(spec.target),
            "model": coerce_na(spec.model),
            "dtype": coerce_na(spec.dtype),
            "substrate": coerce_na(spec.substrate),
            "budget": spec.budget,
            "validation_status": "unknown",
            "recorded_at": coerce_na(_NA),
        }
        return fallback

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------

    def compare(self, run_paths: list[Path], report_dir: Path, logger) -> None:
        """Aggregate per-run summary_metrics.json into metrics.csv and summary.md."""
        report_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        for run_dir in run_paths:
            summary_path = run_dir / "metrics" / "summary_metrics.json"
            if not summary_path.exists():
                continue
            try:
                rows.append(json.loads(summary_path.read_text()))
            except Exception:
                pass

        if not rows and logger and hasattr(logger, "log_event"):
            logger.log_event("compare.empty", {})

        # --- metrics.csv ---
        csv_path = report_dir / "metrics.csv"
        fieldnames = sorted({k for row in rows for k in row})
        try:
            with csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "NA") for k in fieldnames})
        except Exception:
            pass

        # --- summary.md ---
        md_path = report_dir / "summary.md"
        try:
            lines = [
                "# Comparison Summary",
                "",
                f"Generated: {datetime.now(timezone.utc).isoformat()}",
                f"Runs compared: {len(rows)}",
                "",
            ]
            if rows:
                # Header
                lines.append("| " + " | ".join(fieldnames) + " |")
                lines.append("| " + " | ".join("---" for _ in fieldnames) + " |")
                for row in rows:
                    lines.append(
                        "| " + " | ".join(str(row.get(k, "NA")) for k in fieldnames) + " |"
                    )
            else:
                lines.append("_No run data available._")
            md_path.write_text("\n".join(lines) + "\n")
        except Exception:
            pass
