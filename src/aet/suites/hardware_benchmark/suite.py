"""HardwareBenchmarkSuite — aet suite for abc-testing hardware AI benchmarks."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from aet.suites.base import EvalSuite
from aet.core.metrics import welch_ttest, confidence_interval, effect_size, coerce_na, mean_std


_KEY_METRICS = [
    "hw.testbench_pass",
    "hw.localization_recall",
    "hw.localization_precision",
    "hw.regression_count",
    "hw.first_elaboration_iter",
    "hw.first_public_pass_iter",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "run.wall_time_s",
]


def _sig_marker(p) -> str:
    if p is None:
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


class HardwareBenchmarkSuite(EvalSuite):

    def init_run(self, spec, paths, logger) -> None:
        hw_dir = paths.run_path / "hw_benchmark"
        hw_dir.mkdir(parents=True, exist_ok=True)
        (paths.run_path / "metrics").mkdir(parents=True, exist_ok=True)
        (paths.run_path / "logs").mkdir(parents=True, exist_ok=True)
        (hw_dir / "README.md").write_text(
            f"# Hardware Benchmark Run: {spec.run_id}\n\n"
            f"- Benchmark: {getattr(spec, 'benchmark', 'abc-testing') or 'abc-testing'}\n"
            f"- Variant: {getattr(spec, 'variant', None) or spec.target or ''}\n"
            f"- Tool tier: {getattr(spec, 'tool_tier', None) or spec.method or ''}\n"
            f"- Started: {datetime.now(timezone.utc).isoformat()}\n"
        )
        if logger:
            logger.info("init_run.completed suite=hardware_benchmark")

    def validate(self, spec, paths, logger) -> dict:
        errors = []
        status = "pass"

        score_path = paths.run_path / "hw_benchmark" / "score.json"
        if not score_path.exists():
            score_path = paths.run_path / "score.json"

        if not score_path.exists():
            errors.append("score.json not found")
            status = "fail"
        else:
            try:
                score = json.loads(score_path.read_text())
                required_keys = ["testbench_pass", "localization_recall",
                                 "localization_precision", "tainted"]
                missing = [k for k in required_keys if k not in score]
                if missing:
                    errors.append(f"score.json missing keys: {missing}")
                    status = "partial"
            except Exception as e:
                errors.append(f"score.json parse error: {e}")
                status = "fail"

        metrics_path = paths.run_path / "metrics" / "summary_metrics.json"
        if not metrics_path.exists():
            errors.append("summary_metrics.json not found")
            if status == "pass":
                status = "partial"

        report = {
            "run_id": spec.run_id,
            "suite": "hardware_benchmark",
            "status": status,
            "errors": errors,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
        (paths.run_path / "validation_report.json").write_text(
            json.dumps(report, indent=2)
        )
        if logger:
            logger.info("validation.completed status=%s errors=%d", status, len(errors))
        return report

    def collect_metrics(self, spec, paths, logger) -> dict:
        summary_path = paths.run_path / "metrics" / "summary_metrics.json"
        if summary_path.exists():
            try:
                return json.loads(summary_path.read_text())
            except Exception:
                pass

        score_path = paths.run_path / "hw_benchmark" / "score.json"
        if not score_path.exists():
            score_path = paths.run_path / "score.json"
        if score_path.exists():
            try:
                score = json.loads(score_path.read_text())
                return {
                    "run_id": spec.run_id,
                    "suite": "hardware_benchmark",
                    "method": spec.method,
                    "seed": spec.seed,
                    "target": coerce_na(spec.target),
                    "hw.testbench_pass": score.get("testbench_pass"),
                    "hw.localization_recall": score.get("localization_recall"),
                    "hw.localization_precision": score.get("localization_precision"),
                    "hw.tainted": score.get("tainted"),
                    "run.wall_time_s": score.get("wall_time_seconds"),
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
            except Exception:
                pass

        return {
            "run_id": spec.run_id,
            "suite": "hardware_benchmark",
            "validation_status": "unknown",
        }

    def compare(self, run_paths: list[Path], report_dir: Path, logger) -> None:
        import math
        report_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for rp in run_paths:
            smp = rp / "metrics" / "summary_metrics.json"
            if smp.exists():
                try:
                    rows.append(json.loads(smp.read_text()))
                except Exception:
                    pass

        # metrics.csv
        fieldnames = sorted({k for r in rows for k in r})
        csv_path = report_dir / "metrics.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "NA") for k in fieldnames})

        if not rows:
            return

        # Group by method
        method_map: dict[str, list[dict]] = {}
        for row in rows:
            m = row.get("method", "unknown")
            method_map.setdefault(m, []).append(row)

        def _nums(method_rows, key):
            return [r[key] for r in method_rows if isinstance(r.get(key), (int, float))]

        def _mean(vals):
            return sum(vals) / len(vals) if vals else None

        def _stddev(vals):
            if len(vals) < 2:
                return None
            m = sum(vals) / len(vals)
            return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

        def _fmt(v, decimals=2):
            return f"{v:.{decimals}f}" if v is not None else "NA"

        lines = [
            "# Hardware Benchmark Comparison",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Runs compared: {len(rows)}",
            "",
            "## Pass rate by method",
            "",
        ]

        # Detect task type from rows
        is_gen = any(r.get("task") == "spec-to-rtl" for r in rows)

        if is_gen:
            lines += [
                "| method | n | func_pass_rate | ±σ | mean_perf_ratio | mean_wall_s | cheat_rate |",
                "|---|---|---|---|---|---|---|",
            ]
            for method, mrows in sorted(method_map.items()):
                passes = _nums(mrows, "hw.functional_pass")
                ratios = _nums(mrows, "hw.perf_ratio")
                walls = _nums(mrows, "run.wall_time_s")
                cheats = _nums(mrows, "hw.cheat_suspected")
                pr = _mean(passes); pr_sd = _stddev(passes)
                lines.append(
                    f"| {method} | {len(mrows)} | {_fmt(pr)} | {_fmt(pr_sd)} | "
                    f"{_fmt(_mean(ratios), 3)} | {_fmt(_mean(walls), 1)} | "
                    f"{_fmt(_mean(cheats), 2)} |"
                )
        else:
            lines += [
                "| method | n | pass_rate | ±σ | mean_recall | mean_precision | mean_wall_s |",
                "|---|---|---|---|---|---|---|",
            ]
            for method, mrows in sorted(method_map.items()):
                passes = _nums(mrows, "hw.testbench_pass")
                recalls = _nums(mrows, "hw.localization_recall")
                precs = _nums(mrows, "hw.localization_precision")
                walls = _nums(mrows, "run.wall_time_s")
                pr = _mean(passes); pr_sd = _stddev(passes)
                lines.append(
                    f"| {method} | {len(mrows)} | {_fmt(pr)} | {_fmt(pr_sd)} | "
                    f"{_fmt(_mean(recalls), 3)} | {_fmt(_mean(precs), 3)} | "
                    f"{_fmt(_mean(walls), 1)} |"
                )

        # Capability boundary: flag methods with high variance (σ > 0.3 on pass)
        boundary_methods = []
        for method, mrows in method_map.items():
            key = "hw.functional_pass" if is_gen else "hw.testbench_pass"
            vals = _nums(mrows, key)
            sd = _stddev(vals)
            if sd is not None and sd > 0.3:
                boundary_methods.append(method)
        if boundary_methods:
            lines += [
                "",
                f"> **Capability boundary detected** — high pass variance (σ > 0.3) "
                f"in: {', '.join(sorted(boundary_methods))}. "
                f"These methods are near the task difficulty threshold; "
                f"targeted skills/tools have highest expected leverage here.",
            ]

        # Struggle metrics by method
        struggle_keys = [
            ("agent.stall_rate", "stall_rate", ".3f"),
            ("agent.read_to_write_ratio", "read/write", ".2f"),
            ("agent.oracle_without_edit_count", "oracle_no_edit", ".1f"),
            ("agent.max_consecutive_no_edit_turns", "max_idle_turns", ".1f"),
            ("agent.total_oracle_runs", "total_oracle_runs", ".1f"),
        ]
        any_struggle = any(
            any(r.get(k) is not None for r in rows) for k, _, _ in struggle_keys
        )
        if any_struggle:
            lines += [
                "",
                "## Agent struggle metrics by method",
                "",
                "| method | n | " + " | ".join(label for _, label, _ in struggle_keys) + " |",
                "|---|---|" + "|".join("---" for _ in struggle_keys) + "|",
            ]
            for method, mrows in sorted(method_map.items()):
                cells = []
                for key, _, fmt in struggle_keys:
                    vals = _nums(mrows, key)
                    m = _mean(vals)
                    cells.append(f"{m:{fmt}}" if m is not None else "NA")
                lines.append(f"| {method} | {len(mrows)} | " + " | ".join(cells) + " |")

        # Progression metrics by method
        prog_keys = [
            ("agent.first_elaboration_iter", "mean_iters_to_elab", ".1f"),
            ("agent.first_pass_iter", "mean_iters_to_pass", ".1f"),
            ("agent.tok_in_at_first_elaboration", "tok_at_elab", ".0f"),
            ("agent.tok_in_at_first_pass", "tok_at_pass", ".0f"),
        ]
        any_prog = any(
            any(r.get(k) is not None for r in rows) for k, _, _ in prog_keys
        )
        if any_prog:
            lines += [
                "",
                "## Progression metrics by method",
                "",
                "| method | n | " + " | ".join(label for _, label, _ in prog_keys) + " |",
                "|---|---|" + "|".join("---" for _ in prog_keys) + "|",
            ]
            for method, mrows in sorted(method_map.items()):
                cells = []
                for key, _, fmt in prog_keys:
                    vals = _nums(mrows, key)
                    m = _mean(vals)
                    cells.append(f"{m:{fmt}}" if m is not None else "NA")
                lines.append(f"| {method} | {len(mrows)} | " + " | ".join(cells) + " |")

        # Statistical comparison across methods
        if len(method_map) >= 2:
            try:
                method_names = sorted(method_map)
                stat_metrics = (
                    [("hw.functional_pass", "functional_pass"),
                     ("hw.perf_ratio", "perf_ratio")]
                    if is_gen else
                    [("hw.localization_recall", "recall"),
                     ("hw.localization_precision", "precision")]
                )
                lines += ["", "## Statistical comparison (Welch t-test)", ""]
                for metric_key, metric_label in stat_metrics:
                    lines += [f"### {metric_label}", ""]
                    lines += [
                        "| A | B | t | p | sig | Cohen's d | CI_A | CI_B |",
                        "|---|---|---|---|---|---|---|---|",
                    ]
                    for i, ma in enumerate(method_names):
                        for mb in method_names[i+1:]:
                            a = _nums(method_map[ma], metric_key)
                            b = _nums(method_map[mb], metric_key)
                            t, p = welch_ttest(a, b)
                            d = effect_size(a, b)
                            ci_a = confidence_interval(a)
                            ci_b = confidence_interval(b)
                            t_s = f"{t:.3f}" if t is not None else "NA"
                            p_s = f"p={p:.4f}" if p is not None else "NA"
                            d_s = f"{d:.3f}" if d is not None else "NA"
                            ci_a_s = (f"[{ci_a[0]:.3f},{ci_a[1]:.3f}]"
                                      if ci_a[0] is not None else "NA")
                            ci_b_s = (f"[{ci_b[0]:.3f},{ci_b[1]:.3f}]"
                                      if ci_b[0] is not None else "NA")
                            lines.append(
                                f"| {ma} | {mb} | {t_s} | {p_s} | "
                                f"{_sig_marker(p)} | {d_s} | {ci_a_s} | {ci_b_s} |"
                            )
            except Exception:
                pass

        (report_dir / "hw_summary.md").write_text("\n".join(lines) + "\n")
