"""Aggregate all runs for a suite into reports."""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from aet.suites.targetgen.collect_metrics import _ALL_COLUMNS, _STRING_COLUMNS
from aet.core.metrics import welch_ttest, confidence_interval, effect_size


def _load_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "metrics" / "summary_metrics.json"
    manifest_path = run_dir / "run_manifest.yaml"
    if not summary_path.exists() or not manifest_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def _mean_std(values: list) -> tuple:
    nums = [v for v in values if v is not None and isinstance(v, (int, float))]
    if not nums:
        return None, None
    mean = sum(nums) / len(nums)
    if len(nums) < 2:
        return round(mean, 4), None
    variance = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)
    return round(mean, 4), round(math.sqrt(variance), 4)


def _fmt(mean, std) -> str:
    if mean is None:
        return "NA"
    if std is None:
        return str(mean)
    return f"{mean} ± {std}"


def run(root: Path, suite: str, output_dir: Path) -> int:
    runs_dir = root / "runs" / suite
    if not runs_dir.exists():
        print(f"ERROR: no runs directory for suite {suite}: {runs_dir}", file=sys.stderr)
        return 1

    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    rows = []
    for rd in run_dirs:
        data = _load_run(rd)
        if data is not None:
            rows.append(data)

    if not rows:
        print(f"No validated runs found for suite {suite}", file=sys.stderr)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.csv").write_text(",".join(_ALL_COLUMNS) + "\n")
        (output_dir / "ablation_table.md").write_text(
            f"# Ablation Table: {suite}\n\n*No validated runs.*\n"
        )
        (output_dir / "summary.md").write_text(
            f"# {suite} — no validated runs yet\n"
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # metrics.csv — one row per run, full column set, NA for missing
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ALL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {col: row.get(col, "NA") for col in _ALL_COLUMNS}
            # Normalise None → NA for CSV
            for k, v in out.items():
                if v is None:
                    out[k] = "NA"
            writer.writerow(out)

    # ablation_table.md — one row per method, mean ± std, smoke runs excluded
    non_smoke = [r for r in rows if not r.get("is_smoke_test", True)]
    methods = sorted({r["method"] for r in non_smoke})

    numeric_cols = [
        c for c in _ALL_COLUMNS
        if c not in _STRING_COLUMNS and c not in ("is_smoke_test", "seed")
    ]

    table_lines = [
        f"# Ablation Table: {suite}\n",
        "| method | seeds | " + " | ".join(numeric_cols) + " |",
        "|---|---|" + "|".join("---" for _ in numeric_cols) + "|",
    ]
    for method in methods:
        method_rows = [r for r in non_smoke if r["method"] == method]
        seeds = sorted({r["seed"] for r in method_rows})
        cells = []
        for col in numeric_cols:
            vals = [r.get(col) for r in method_rows]
            mean, std = _mean_std(vals)
            cells.append(_fmt(mean, std))
        table_lines.append(
            f"| {method} | {','.join(str(s) for s in seeds)} | "
            + " | ".join(cells) + " |"
        )

    if not methods:
        table_lines.append("| *(no real baseline runs yet)* | — |" + "|".join("NA" for _ in numeric_cols) + "|")

    (output_dir / "ablation_table.md").write_text("\n".join(table_lines) + "\n")

    # summary.md
    smoke_count = sum(1 for r in rows if r.get("is_smoke_test", True))
    real_count = len(rows) - smoke_count
    (output_dir / "summary.md").write_text(
        f"# {suite} — comparison summary\n\n"
        f"- Total runs: {len(rows)} ({smoke_count} smoke, {real_count} real)\n"
        f"- Methods with real runs: {len(methods)}\n\n"
        f"See `ablation_table.md` for per-method aggregated metrics.\n"
        f"See `metrics.csv` for per-run raw data.\n"
    )

    # statistical_comparison.md
    try:
        numeric_metrics = [c for c in numeric_cols if any(
            isinstance(r.get(c), (int, float)) for r in rows
        )]

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

        method_names = sorted({r["method"] for r in non_smoke})
        if len(method_names) >= 2:
            stat_lines = [
                f"# Statistical Comparison: {suite}",
                "",
                f"Generated: {datetime.now(timezone.utc).isoformat()}",
                f"Real runs compared: {real_count}",
                "",
            ]
            pairs = [(method_names[i], method_names[j])
                     for i in range(len(method_names))
                     for j in range(i + 1, len(method_names))]
            for ma, mb in pairs:
                rows_a = [r for r in non_smoke if r["method"] == ma]
                rows_b = [r for r in non_smoke if r["method"] == mb]
                stat_lines.append(f"## {ma} vs {mb}")
                stat_lines.append("")
                for metric in numeric_metrics[:10]:  # limit to first 10 numeric metrics
                    vals_a = [r.get(metric) for r in rows_a
                              if isinstance(r.get(metric), (int, float))]
                    vals_b = [r.get(metric) for r in rows_b
                              if isinstance(r.get(metric), (int, float))]
                    if not vals_a and not vals_b:
                        continue
                    t_stat, p_val = welch_ttest(vals_a, vals_b)
                    d = effect_size(vals_a, vals_b)
                    sig = _sig_marker(p_val)
                    mean_a, std_a = _mean_std(vals_a)
                    mean_b, std_b = _mean_std(vals_b)
                    stat_lines.append(f"### {metric}")
                    stat_lines.append("")
                    na_s = str(len(vals_a))
                    nb_s = str(len(vals_b))
                    stat_lines += [
                        f"| | {ma} (n={na_s}) | {mb} (n={nb_s}) |",
                        "|---|---|---|",
                    ]
                    ma_s = f"{mean_a} ± {std_a}" if mean_a is not None else "NA"
                    mb_s = f"{mean_b} ± {std_b}" if mean_b is not None else "NA"
                    stat_lines.append(f"| mean ± std | {ma_s} | {mb_s} |")
                    t_s = f"t={t_stat:.4f}, p={p_val:.4f} {sig}" if t_stat is not None else "NA"
                    stat_lines.append(f"| Welch t-test | {t_s} | |")
                    d_s = f"{d:.4f}" if d is not None else "NA"
                    stat_lines.append(f"| Cohen's d | {d_s} | |")
                    stat_lines.append("")
            stat_lines += [
                "---",
                "Significance: * p<0.05  ** p<0.01  *** p<0.001  ns not significant",
            ]
            (output_dir / "statistical_comparison.md").write_text(
                "\n".join(stat_lines) + "\n"
            )
    except Exception:
        pass

    # regression_report.md
    try:
        baseline_file = root / "baselines" / suite / "baseline.json"
        if baseline_file.exists():
            baseline = json.loads(baseline_file.read_text())
            baseline_cost  = baseline.get("aet.agent.cost_usd")
            baseline_score = baseline.get("task_achievement_score")
            reg_lines = [
                f"# Regression Report: {suite}",
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
                delta_cost_pct, delta_score = None, None
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
                is_reg = (delta_cost_pct is not None and delta_cost_pct > 20) or \
                         (delta_score is not None and delta_score < -0.05)
                status = "✗ REGRESSION" if is_reg else (
                    "✓ OK" if (delta_cost_pct is not None or delta_score is not None) else "⚠ no data"
                )
                cost_s   = f"${float(cost):.4f}" if cost  is not None else "NA"
                score_s  = f"{float(score):.4f}" if score is not None else "NA"
                dcost_s  = f"{delta_cost_pct:+.1f}%" if delta_cost_pct is not None else "NA"
                dscore_s = f"{delta_score:+.4f}"      if delta_score    is not None else "NA"
                reg_lines.append(f"| {run_id} | {cost_s} | {dcost_s} | {score_s} | {dscore_s} | {status} |")
            reg_lines += [
                "",
                "Regression criteria:",
                "  - ✗ REGRESSION: cost > baseline × 1.20  OR  score < baseline − 0.05",
                "  - ✓ OK: within threshold",
            ]
            (output_dir / "regression_report.md").write_text("\n".join(reg_lines) + "\n")
    except Exception:
        pass

    print(f"reports written to {output_dir}")
    print(f"  {len(rows)} runs ({smoke_count} smoke, {real_count} real baseline)")
    print(f"  {csv_path}")
    print(f"  {output_dir / 'ablation_table.md'}")
    return 0
