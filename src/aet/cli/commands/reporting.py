"""Reporting commands: compare, baseline, runs, show."""
from __future__ import annotations

import sys
from pathlib import Path

from aet.core.run_manifest import RunManifest
from aet.suites import get_suite

from aet.cli._common import (
    _resolve_project_root,
    _save_fig,
)

def _cmd_compare(args) -> None:
    project_root = _resolve_project_root(args)
    suite_name = args.suite

    if args.output_dir:
        report_dir = Path(args.output_dir).resolve()
    else:
        report_dir = project_root / "reports" / suite_name

    runs_root = project_root / "runs" / suite_name
    if runs_root.exists():
        run_paths = [p for p in runs_root.glob("*") if p.is_dir()]
    else:
        run_paths = []

    # Optionally filter smoke-test runs
    if getattr(args, "no_smoke", False):
        filtered = []
        for rp in run_paths:
            manifest_path = rp / "run_manifest.yaml"
            if manifest_path.exists():
                try:
                    m = RunManifest.load(manifest_path)
                    if not m.is_smoke_test:
                        filtered.append(rp)
                except Exception:
                    filtered.append(rp)
            else:
                filtered.append(rp)
        run_paths = filtered

    report_dir.mkdir(parents=True, exist_ok=True)

    # Minimal logger for compare (no run-specific context needed)
    import logging
    logger = logging.getLogger("aet.compare")

    suite = get_suite(suite_name)
    suite.compare(run_paths, report_dir, logger)

    print(f"[aet] Compare complete: wrote report to {report_dir}")

    # Optional trajectory comparison plot (guarded: needs the [viz] extra + trajectory data)
    if getattr(args, "plots", False):
        _write_comparison_plot(run_paths, report_dir)


def _write_comparison_plot(run_paths, report_dir: Path) -> None:
    from aet.trajectory.model import RunTrajectory
    trajs = []
    for rp in sorted(run_paths):
        traj_json = rp / "metrics" / "trajectory.json"
        if traj_json.is_file():
            try:
                trajs.append(RunTrajectory.from_json(traj_json))
            except Exception:
                continue
    if not trajs:
        print("[aet] --plots: no run has metrics/trajectory.json; skipping trajectory plot")
        return
    try:
        from aet.viz.trajectory_plot import plot_comparison
        from aet.viz.comparison import plot_rate_panels, plot_cost_vs_time, plot_tests_facets
    except ImportError as e:
        print(f"[aet] --plots: {e}", file=sys.stderr)
        return
    labels = [t.run_id for t in trajs]
    # the full presentation set: stacked cumulative + rate panels + cost-vs-time + tests facets.
    # tests-facets degrades gracefully when a run has no over-time test signal (flat lane).
    figures = {
        "trajectory_comparison": plot_comparison(trajs),
        "rate_panels": plot_rate_panels(trajs, labels),
        "cost_vs_time": plot_cost_vs_time(trajs, labels),
        "tests_facets": plot_tests_facets(trajs, labels),
    }
    for name, fig in figures.items():
        out = report_dir / f"{name}.png"
        for p in _save_fig(fig, out, 200):
            print(f"[aet] wrote {p}  ({len(trajs)} runs)")


def _baseline_dir(project_root: Path, suite: str) -> Path:
    return project_root / "baselines" / suite


def _cmd_baseline_set(args) -> None:
    import json as _json
    project_root = _resolve_project_root(args)
    suite = args.suite
    baseline_path = _baseline_dir(project_root, suite) / "baseline.json"

    runs_root = project_root / "runs" / suite
    if not runs_root.exists():
        print(f"[aet] Error: no runs directory for suite {suite}: {runs_root}", file=sys.stderr)
        sys.exit(1)

    if getattr(args, "run_id", None):
        run_dir = runs_root / args.run_id
        summary_path = run_dir / "metrics" / "summary_metrics.json"
        if not summary_path.exists():
            print(f"[aet] Error: summary_metrics.json not found at {summary_path}", file=sys.stderr)
            sys.exit(1)
        summary = _json.loads(summary_path.read_text())
    else:
        best_summary = None
        best_score = float("-inf")
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            sp = run_dir / "metrics" / "summary_metrics.json"
            if not sp.exists():
                continue
            try:
                s = _json.loads(sp.read_text())
                score = s.get("task_achievement_score")
                if score is not None and float(score) > best_score:
                    best_score = float(score)
                    best_summary = s
            except Exception:
                pass
        if best_summary is None:
            print(f"[aet] Error: no valid runs found for suite {suite}", file=sys.stderr)
            sys.exit(1)
        summary = best_summary

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(_json.dumps(summary, indent=2))
    print(f"[aet] Baseline set: {baseline_path}")


def _cmd_baseline_show(args) -> None:
    project_root = _resolve_project_root(args)
    suite = args.suite
    baseline_path = _baseline_dir(project_root, suite) / "baseline.json"
    if not baseline_path.exists():
        print(f"[aet] No baseline found for suite {suite} at {baseline_path}", file=sys.stderr)
        sys.exit(1)
    print(baseline_path.read_text())


def _cmd_baseline(args) -> None:
    if not hasattr(args, "baseline_func"):
        args._baseline_parser.print_help()
        sys.exit(0)
    args.baseline_func(args)


def _cmd_runs(args) -> None:
    import json

    project_root = _resolve_project_root(args)
    runs_root = project_root / "runs"
    suite_filter = getattr(args, "suite", None)

    run_dirs: list[Path] = []
    if runs_root.exists():
        for suite_dir in sorted(runs_root.iterdir()):
            if not suite_dir.is_dir():
                continue
            if suite_filter and suite_dir.name != suite_filter:
                continue
            for run_dir in sorted(suite_dir.iterdir()):
                if run_dir.is_dir() and not run_dir.name.startswith("_"):
                    run_dirs.append(run_dir)

    if not run_dirs:
        print("[aet] No runs found" + (f" for suite '{suite_filter}'" if suite_filter else "") + ".")
        return

    fmt = getattr(args, "format", "table")

    rows = []
    for run_dir in run_dirs:
        suite = run_dir.parent.name
        run_id = run_dir.name
        status = "?"
        model = ""
        cost = None
        turns = None
        tokens_in = None
        trace_id = None
        mlflow_run = None

        report_path = run_dir / "validation_report.json"
        if report_path.exists():
            try:
                r = json.loads(report_path.read_text())
                status = r.get("status", "?")
            except Exception:
                pass
        else:
            # no validation report → fall back to the manifest status (surfaces rate_limited_unfinished
            # / completed for `aet run` runs, so unfinished experiments are discoverable here)
            manifest_path = run_dir / "run_manifest.yaml"
            if manifest_path.exists():
                try:
                    from aet.core.run_manifest import RunManifest
                    status = RunManifest.load(manifest_path).status or status
                except Exception:
                    pass

        params_path = run_dir / "logs" / "params.json"
        if params_path.exists():
            try:
                p = json.loads(params_path.read_text())
                model = p.get("gen_ai.response.model", "")
                mlflow_run = p.get("mlflow_run_id", "")
                trace_id = p.get("aet.otel_trace_id", "")
            except Exception:
                pass

        metrics_path = run_dir / "logs" / "metrics.jsonl"
        if metrics_path.exists():
            try:
                for line in metrics_path.read_text().splitlines():
                    m = json.loads(line)
                    if m["name"] == "aet.agent.cost_usd":
                        cost = m["value"]
                    elif m["name"] == "aet.agent.num_turns":
                        turns = int(m["value"])
                    elif m["name"] == "gen_ai.usage.input_tokens":
                        tokens_in = int(m["value"])
            except Exception:
                pass

        rows.append({
            "suite": suite, "run_id": run_id, "status": status,
            "model": model, "turns": turns, "tokens_in": tokens_in,
            "cost": cost, "trace_id": trace_id, "mlflow_run": mlflow_run,
        })

    if fmt == "json":
        print(json.dumps(rows, indent=2))
        return

    STATUS_SYM = {"pass": "✓", "partial": "~", "error": "✗", "unknown": "?", "?": "?",
                  "completed": "✓", "rate_limited_unfinished": "⏸", "rate_limited_waiting": "⏳",
                  "initialized": "·"}
    col_w = max(len(r["run_id"]) for r in rows)
    print(f"\n  {'':2}  {'suite':<14}  {'run_id':<{col_w}}  {'model':<24}  {'turns':>5}  {'tokens_in':>10}  {'cost':>10}")
    print(f"  {'':2}  {'-'*14}  {'-'*col_w}  {'-'*24}  {'-'*5}  {'-'*10}  {'-'*10}")
    for r in rows:
        sym = STATUS_SYM.get(r["status"], "?")
        cost_s = f"${r['cost']:.4f}" if r["cost"] is not None else "—"
        turns_s = str(r["turns"]) if r["turns"] is not None else "—"
        tokens_s = str(r["tokens_in"]) if r["tokens_in"] is not None else "—"
        model_s = (r["model"] or "—")[:24]
        print(f"  {sym:<2}  {r['suite']:<14}  {r['run_id']:<{col_w}}  {model_s:<24}  {turns_s:>5}  {tokens_s:>10}  {cost_s:>10}")
    print()



def _cmd_spend(args) -> None:
    """Cross-experiment spend rollup over one or more run roots (table or --json)."""
    import json
    import sys as _sys

    from aet.trajectory.rollup import rollup_runs

    roots = [Path(r) for r in args.roots]
    missing = [str(r) for r in roots if not r.exists()]
    if missing:
        print(f"[aet] Error: run root(s) not found: {', '.join(missing)}", file=_sys.stderr)
        _sys.exit(1)

    roll = rollup_runs(roots, budget_usd=args.budget_usd)

    if getattr(args, "json", False):
        print(json.dumps(roll.to_dict(), indent=2))
    else:
        _print_spend_table(roll)

    # Non-zero exit when a hard budget ceiling is exceeded (enforceable in CI / scripts).
    if roll.over_budget:
        print(f"[aet] OVER BUDGET: ${roll.total_cost_usd:.4f} > ${roll.budget_usd:.2f}",
              file=_sys.stderr)
        _sys.exit(2)


def _print_spend_table(roll) -> None:
    tok = roll.tokens
    print(f"\n  Spend rollup — {roll.n_runs} run(s)")
    print("  " + "─" * 58)
    print(f"    total cost           ${roll.total_cost_usd:>12.4f}")
    if roll.unpriced_runs:
        print(f"    unpriced runs        {roll.unpriced_runs:>13}  (cost unavailable — not $0)")
    print(f"    tokens  in/out       {tok.input:>13,} / {tok.output:,}")
    print(f"    tokens  cache        {tok.cache_total:>13,}  "
          f"(read {tok.cache_read:,} / create {tok.cache_creation:,})")

    if roll.per_model:
        print("\n    per-model  (within-run split — every model a run touched, not just its primary)")
        name_w = max(len(m) for m in roll.per_model)
        for model, ms in sorted(roll.per_model.items(),
                                key=lambda kv: kv[1].cost_usd, reverse=True):
            unp = ms.n_runs - ms.n_priced_runs
            unp_s = f"  ({unp} unpriced)" if unp else ""
            print(f"      {model:<{name_w}}  ${ms.cost_usd:>10.4f}  "
                  f"{ms.n_runs:>3} run(s)  {ms.tokens.total:>12,} tok  "
                  f"{ms.activity_share * 100:>5.1f}% activity{unp_s}")

    if roll.budget_usd is not None:
        state = "OVER BUDGET" if roll.over_budget else "within budget"
        print(f"\n    budget               ${roll.budget_usd:>12.2f}")
        print(f"    headroom             ${roll.headroom_usd:>12.4f}  ({state})")
    print()


def _cmd_show(args) -> None:
    import json

    run_path = Path(args.run_path).resolve()
    if not run_path.exists():
        print(f"[aet] Run path not found: {run_path}", file=sys.stderr)
        sys.exit(1)

    fmt = getattr(args, "format", "text")

    # ── load all local data ────────────────────────────────────────────────
    params: dict = {}
    params_path = run_path / "logs" / "params.json"
    if params_path.exists():
        try:
            params = json.loads(params_path.read_text())
        except Exception:
            pass

    metrics: list[dict] = []
    metrics_path = run_path / "logs" / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            try:
                metrics.append(json.loads(line))
            except Exception:
                pass

    events: list[dict] = []
    events_path = run_path / "logs" / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except Exception:
                pass

    report: dict = {}
    report_path = run_path / "validation_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            pass

    if fmt == "json":
        print(json.dumps({"params": params, "metrics": metrics,
                          "events": events, "report": report}, indent=2))
        return

    # ── text display ───────────────────────────────────────────────────────
    SEP = "─" * 60

    print(f"\n{SEP}")
    print(f"  Run: {run_path.name}   suite: {run_path.parent.name}")
    print(SEP)

    # params
    if params:
        print("\n  PARAMS")
        for k, v in params.items():
            print(f"    {k:<40} {v}")

    # metrics grouped
    if metrics:
        print("\n  METRICS")
        groups: dict[str, list] = {}
        for m in metrics:
            cat = m["name"].split(".")[0]
            groups.setdefault(cat, []).append(m)

        # cost block first
        for cat in ["aet", "gen_ai", "cost_usd", "input_tokens", "output_tokens",
                    "cache_read_tokens", "cache_creation_tokens",
                    "turn", "task_achievement_score", "validation_errors"]:
            if cat not in groups:
                continue
            print(f"\n    [{cat}]")
            for m in groups.pop(cat):
                step = f"  step={m['step']}" if m.get("step") is not None else ""
                val = m["value"]
                if isinstance(val, float) and "cost" in m["name"]:
                    val = f"${val:.6f}"
                print(f"      {m['name']:<52} {val}{step}")
        for cat, ms in groups.items():
            print(f"\n    [{cat}]")
            for m in ms:
                step = f"  step={m['step']}" if m.get("step") is not None else ""
                print(f"      {m['name']:<52} {m['value']}{step}")

    # events timeline
    if events:
        print("\n  EVENTS")
        for e in events:
            ts = e.get("ts", "")[:19].replace("T", " ")
            print(f"\n    {ts}  {e['event']}")
            for k, v in (e.get("payload") or {}).items():
                vs = str(v)
                if len(vs) > 120:
                    vs = vs[:120] + "…"
                print(f"      {k}: {vs}")

    # validation
    if report:
        print(f"\n  VALIDATION  →  {report.get('status', '?').upper()}")
        errs = report.get("errors", [])
        if errs:
            for e in errs:
                print(f"    ✗  {e}")
        else:
            print("    ✓  no errors")

    # links
    print("\n  LINKS")
    trace_id = params.get("aet.otel_trace_id", "")
    mlflow_run_id = params.get("mlflow_run_id", "")
    mlflow_uri = params.get("mlflow_tracking_uri", "")
    if trace_id:
        print(f"    signoz   http://localhost:8080/trace/{trace_id}")
    if mlflow_run_id and mlflow_uri:
        print(f"    mlflow   {mlflow_uri}/#/experiments/1/runs/{mlflow_run_id}")
    session_id = params.get("gen_ai.conversation.id", "")
    if session_id:
        print(f"    replay   claude --resume {session_id}")
    print()
