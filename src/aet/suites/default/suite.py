from __future__ import annotations
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from aet.suites.base import EvalSuite
from aet.core.metrics import _NA, coerce_na, welch_ttest, confidence_interval, effect_size, jaccard_similarity


def _write_regression_report(rows: list, baseline: dict, report_dir: Path) -> None:
    baseline_cost  = baseline.get("aet.agent.cost_usd")
    baseline_score = baseline.get("task_achievement_score")

    lines = [
        "# Regression Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Baseline run: {baseline.get('run_id', 'unknown')}",
        "",
        "| run_id | cost | Δcost% | score | Δscore | status |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        run_id = row.get("run_id", "?")
        cost   = row.get("aet.agent.cost_usd")
        score  = row.get("task_achievement_score")

        delta_cost_pct = None
        delta_score    = None

        if baseline_cost is not None and cost is not None:
            try:
                delta_cost_pct = (float(cost) - float(baseline_cost)) / abs(float(baseline_cost)) * 100
            except (ZeroDivisionError, TypeError):
                pass
        if baseline_score is not None and score is not None:
            try:
                delta_score = float(score) - float(baseline_score)
            except TypeError:
                pass

        cost_regression  = (delta_cost_pct is not None and delta_cost_pct > 20)
        score_regression = (delta_score is not None and delta_score < -0.05)
        is_regression    = cost_regression or score_regression

        if is_regression:
            status = "✗ REGRESSION"
        elif delta_cost_pct is not None or delta_score is not None:
            status = "✓ OK"
        else:
            status = "⚠ no data"

        cost_s   = f"${float(cost):.4f}" if cost  is not None else "NA"
        score_s  = f"{float(score):.4f}" if score is not None else "NA"
        dcost_s  = f"{delta_cost_pct:+.1f}%" if delta_cost_pct is not None else "NA"
        dscore_s = f"{delta_score:+.4f}"      if delta_score    is not None else "NA"

        lines.append(f"| {run_id} | {cost_s} | {dcost_s} | {score_s} | {dscore_s} | {status} |")

    lines += [
        "",
        "Regression criteria:",
        "  - ✗ REGRESSION: cost > baseline × 1.20  OR  score < baseline − 0.05",
        "  - ✓ OK: within threshold",
        "  - ⚠: comparison data unavailable",
    ]
    (report_dir / "regression_report.md").write_text("\n".join(lines) + "\n")


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

        # --- statistical_comparison.md ---
        try:
            _KEY_METRICS = ["aet.agent.cost_usd", "aet.agent.num_turns", "task_achievement_score"]

            def _sig_marker(p):
                if p is None:
                    return "n/a"
                if p < 0.001:
                    return "***"
                if p < 0.01:
                    return "**"
                if p < 0.05:
                    return "*"
                return "ns"

            from aet.core.metrics import mean_std as _mean_std
            methods_map: dict[str, list] = {}
            for row in rows:
                m = row.get("method", "unknown")
                methods_map.setdefault(m, []).append(row)

            method_names = sorted(methods_map)
            if len(method_names) >= 2:
                stat_lines = [
                    "# Statistical Comparison",
                    "",
                    f"Generated: {datetime.now(timezone.utc).isoformat()}",
                    f"Runs compared: {len(rows)}",
                    "",
                ]
                pairs = [(method_names[i], method_names[j])
                         for i in range(len(method_names))
                         for j in range(i + 1, len(method_names))]
                for ma, mb in pairs:
                    stat_lines.append(f"## {ma} vs {mb}")
                    stat_lines.append("")
                    for metric in _KEY_METRICS:
                        vals_a = [r.get(metric) for r in methods_map[ma]
                                  if isinstance(r.get(metric), (int, float))]
                        vals_b = [r.get(metric) for r in methods_map[mb]
                                  if isinstance(r.get(metric), (int, float))]
                        if not vals_a and not vals_b:
                            continue
                        stat_lines.append(f"### {metric}")
                        stat_lines.append("")
                        mean_a, std_a = _mean_std(vals_a)
                        mean_b, std_b = _mean_std(vals_b)
                        ci_a = confidence_interval(vals_a)
                        ci_b = confidence_interval(vals_b)
                        t_stat, p_val = welch_ttest(vals_a, vals_b)
                        d = effect_size(vals_a, vals_b)
                        sig = _sig_marker(p_val)
                        na_str = str(len(vals_a)) if vals_a else "0"
                        nb_str = str(len(vals_b)) if vals_b else "0"
                        stat_lines += [
                            f"| | {ma} (n={na_str}) | {mb} (n={nb_str}) |",
                            "|---|---|---|",
                        ]
                        mean_a_s = f"{mean_a} ± {std_a}" if mean_a is not None else "NA"
                        mean_b_s = f"{mean_b} ± {std_b}" if mean_b is not None else "NA"
                        stat_lines.append(f"| mean ± std | {mean_a_s} | {mean_b_s} |")
                        ci_a_s = f"[{ci_a[0]}, {ci_a[1]}]" if ci_a[0] is not None else "NA"
                        ci_b_s = f"[{ci_b[0]}, {ci_b[1]}]" if ci_b[0] is not None else "NA"
                        stat_lines.append(f"| 95% CI | {ci_a_s} | {ci_b_s} |")
                        t_s = f"t={t_stat:.4f}, p={p_val:.4f} {sig}" if t_stat is not None else "NA"
                        stat_lines.append(f"| Welch t-test | {t_s} | |")
                        d_s = f"{d:.4f}" if d is not None else "NA"
                        stat_lines.append(f"| Cohen's d | {d_s} | |")
                        stat_lines.append("")
                stat_lines += [
                    "---",
                    "Significance: * p<0.05  ** p<0.01  *** p<0.001  ns not significant",
                ]
                (report_dir / "statistical_comparison.md").write_text(
                    "\n".join(stat_lines) + "\n"
                )
        except Exception:
            pass

        # --- trajectory_similarity.md ---
        try:
            seqs = [(str(row.get("run_id") or "?"), row.get("tool_sequence", []))
                    for row in rows]
            seqs = [(rid, seq) for rid, seq in seqs
                    if isinstance(seq, list) and seq]
            if len(seqs) >= 2:
                ids = [rid for rid, _ in seqs]
                traj_lines = [
                    "# Trajectory Similarity",
                    "",
                    "Pairwise Jaccard similarity of tool sequences.",
                    "",
                    "| run_id | " + " | ".join(ids) + " |",
                    "|---|" + "|".join("---" for _ in ids) + "|",
                ]
                for rid_a, seq_a in seqs:
                    cells = [f"{jaccard_similarity(seq_a, seq_b):.3f}"
                             for _, seq_b in seqs]
                    traj_lines.append(f"| {rid_a} | " + " | ".join(cells) + " |")
                (report_dir / "trajectory_similarity.md").write_text(
                    "\n".join(traj_lines) + "\n"
                )
        except Exception:
            pass

        # --- regression_report.md ---
        try:
            if run_paths:
                inferred_root = run_paths[0].parent.parent.parent
                suite_name = run_paths[0].parent.name
                baseline_file = inferred_root / "baselines" / suite_name / "baseline.json"
                if baseline_file.exists():
                    baseline = json.loads(baseline_file.read_text())
                    _write_regression_report(rows, baseline, report_dir)
        except Exception:
            pass

