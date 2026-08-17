"""aet CLI — entry point. Builds the argparse table and dispatches to command handlers
defined in :mod:`aet.cli.commands` (grouped by concern)."""
from __future__ import annotations

import argparse
import sys

from aet.core.errors import AetError, SuiteNotFoundError, RunAlreadyExistsError
from aet.cli._common import _add_global_args, _add_tracking_args
from aet.cli.commands.lifecycle import (
    _cmd_init_project, _cmd_init_run, _cmd_validate, _cmd_run_suite,
)
from aet.cli.commands.reporting import (
    _cmd_compare, _cmd_baseline_set, _cmd_baseline_show, _cmd_baseline, _cmd_runs, _cmd_show,
    _cmd_spend,
)
from aet.cli.commands.trajectory import (
    _SESSION_KINDS, _cmd_import, _cmd_plot, _cmd_plot_sessions, _cmd_run, _cmd_monitor,
    _cmd_otel_sink,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aet",
        description="Agentic Eval Tool — research evaluation harness",
    )
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # ------------------------------------------------------------------
    # init-project
    # ------------------------------------------------------------------
    p_init_project = subparsers.add_parser(
        "init-project",
        help="Initialize a new project from a template",
    )
    p_init_project.add_argument(
        "--template",
        choices=["default", "targetgen"],
        required=True,
        help="Template to use",
    )
    p_init_project.add_argument(
        "--project-root",
        metavar="PATH",
        default=None,
        help="Destination directory (default: current working directory)",
    )
    p_init_project.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing files",
    )
    p_init_project.set_defaults(func=_cmd_init_project)

    # ------------------------------------------------------------------
    # init-run
    # ------------------------------------------------------------------
    p_init_run = subparsers.add_parser(
        "init-run",
        help="Initialize a new evaluation run",
    )
    p_init_run.add_argument("--suite", required=True, help="Evaluation suite name")
    p_init_run.add_argument("--method", required=True, help="Method name")
    p_init_run.add_argument("--seed", required=True, type=int, help="Random seed")
    p_init_run.add_argument("--target", default=None, help="Target hardware/platform")
    p_init_run.add_argument("--model", default=None, help="Model identifier")
    p_init_run.add_argument("--dtype", default=None, help="Data type")
    p_init_run.add_argument("--substrate", default=None, help="Substrate identifier")
    p_init_run.add_argument(
        "--smoke", dest="smoke", action="store_true", default=True,
        help="Mark as smoke test run (default)",
    )
    p_init_run.add_argument(
        "--no-smoke", dest="smoke", action="store_false",
        help="Mark as a full (non-smoke) run",
    )
    p_init_run.add_argument(
        "--budget", default="cheap_smoke", help="Budget identifier (default: cheap_smoke)"
    )
    p_init_run.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing run directory",
    )
    _add_global_args(p_init_run)
    _add_tracking_args(p_init_run)
    p_init_run.set_defaults(func=_cmd_init_run)

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------
    p_validate = subparsers.add_parser(
        "validate",
        help="Validate outputs of a completed run",
    )
    p_validate.add_argument("run_path", metavar="RUN_PATH", help="Path to the run directory")
    _add_global_args(p_validate)
    _add_tracking_args(p_validate)
    p_validate.set_defaults(func=_cmd_validate)

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------
    p_compare = subparsers.add_parser(
        "compare",
        help="Aggregate and compare multiple runs for a suite",
    )
    p_compare.add_argument("--suite", required=True, help="Suite name to compare")
    p_compare.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Output directory for reports (default: <project-root>/reports/<suite>)",
    )
    p_compare.add_argument(
        "--no-smoke",
        dest="no_smoke",
        action="store_true",
        default=False,
        help="Exclude smoke-test runs from comparison",
    )
    p_compare.add_argument(
        "--plots",
        action="store_true",
        default=False,
        help="Also render a trajectory comparison plot (requires the [viz] extra)",
    )
    _add_global_args(p_compare)
    _add_tracking_args(p_compare)
    p_compare.set_defaults(func=_cmd_compare)

    # ------------------------------------------------------------------
    # run-suite
    # ------------------------------------------------------------------
    p_run_suite = subparsers.add_parser(
        "run-suite",
        help="Run init-run + validate for all method/seed combos, then compare",
    )
    p_run_suite.add_argument("--suite", required=True, help="Suite name")
    p_run_suite.add_argument("--target", default=None, help="Target hardware/platform")
    p_run_suite.add_argument(
        "--methods",
        required=True,
        metavar="m1,m2,...",
        help="Comma-separated list of methods",
    )
    p_run_suite.add_argument(
        "--seeds",
        required=True,
        metavar="1,2,...",
        help="Comma-separated list of seeds (integers)",
    )
    # run-suite also needs init-run style args
    p_run_suite.add_argument("--model", default=None, help="Model identifier")
    p_run_suite.add_argument("--dtype", default=None, help="Data type")
    p_run_suite.add_argument("--substrate", default=None, help="Substrate identifier")
    p_run_suite.add_argument(
        "--smoke", dest="smoke", action="store_true", default=True,
        help="Mark runs as smoke tests (default)",
    )
    p_run_suite.add_argument(
        "--no-smoke", dest="smoke", action="store_false",
        help="Mark runs as full (non-smoke) runs",
    )
    p_run_suite.add_argument(
        "--budget", default="cheap_smoke", help="Budget identifier (default: cheap_smoke)"
    )
    p_run_suite.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing run directories",
    )
    p_run_suite.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Output directory for compare reports (default: <project-root>/reports/<suite>)",
    )
    _add_global_args(p_run_suite)
    _add_tracking_args(p_run_suite)
    p_run_suite.set_defaults(func=_cmd_run_suite)

    # ------------------------------------------------------------------
    # runs
    # ------------------------------------------------------------------
    p_runs = subparsers.add_parser(
        "runs",
        help="List all runs in a project",
    )
    p_runs.add_argument("--suite", default=None, help="Filter by suite name")
    p_runs.add_argument(
        "--format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )
    _add_global_args(p_runs)
    p_runs.set_defaults(func=_cmd_runs)

    # ------------------------------------------------------------------
    # spend — cross-experiment spend rollup over one or more run roots
    # ------------------------------------------------------------------
    p_spend = subparsers.add_parser(
        "spend",
        help="Aggregate token/cost spend across one or more run roots (enforce a budget ceiling)",
    )
    p_spend.add_argument("roots", nargs="+", metavar="RUN_ROOT",
                         help="Run directories or roots that contain them (e.g. a runs/ tree)")
    p_spend.add_argument("--json", action="store_true", default=False,
                         help="Emit the rollup as JSON instead of a table")
    p_spend.add_argument("--budget-usd", dest="budget_usd", type=float, default=None,
                         metavar="N",
                         help="Hard budget ceiling: print headroom and exit non-zero if total "
                              "spend across all roots exceeds N")
    p_spend.set_defaults(func=_cmd_spend)

    # ------------------------------------------------------------------
    # baseline
    # ------------------------------------------------------------------
    p_baseline = subparsers.add_parser(
        "baseline",
        help="Set or show a baseline for regression detection",
    )
    p_baseline._baseline_parser = p_baseline
    baseline_sub = p_baseline.add_subparsers(dest="baseline_subcommand")

    p_baseline_set = baseline_sub.add_parser("set", help="Set the baseline for a suite")
    p_baseline_set.add_argument("--suite", required=True, help="Suite name")
    p_baseline_set.add_argument("--run-id", dest="run_id", default=None,
                                help="Run ID to use as baseline (omit to pick best)")
    _add_global_args(p_baseline_set)
    p_baseline_set.set_defaults(baseline_func=_cmd_baseline_set)

    p_baseline_show = baseline_sub.add_parser("show", help="Show the current baseline")
    p_baseline_show.add_argument("--suite", required=True, help="Suite name")
    _add_global_args(p_baseline_show)
    p_baseline_show.set_defaults(baseline_func=_cmd_baseline_show)

    p_baseline.set_defaults(func=_cmd_baseline)

    # ------------------------------------------------------------------
    # import — ingest an existing agentic run into a canonical trajectory
    # ------------------------------------------------------------------
    p_import = subparsers.add_parser(
        "import",
        help="Import an existing agentic run into a canonical trajectory",
    )
    p_import.add_argument("--source", default="capsule-bench", metavar="SOURCE",
                          help="Run layout to import. 'transcript' = generic Claude Code stream-json "
                               "*.jsonl (any project); 'otel' = full-fidelity OTLP capture "
                               "(otel_logs.jsonl from `aet otel-sink` — real per-turn tokens/cost/"
                               "duration + cache); 'codex' = Codex-CLI `codex exec --json` stdout "
                               "JSONL (per-turn input/cached/cache-write/output/reasoning tokens + "
                               "tool spans + nullable provenanced cost); 'capsule-bench' = the "
                               "bundled suite layout (default).")
    p_import.add_argument("--format", dest="fmt", default=None, metavar="FORMAT",
                          help="Alias for --source (spec spelling); e.g. `--format codex`.")
    p_import.add_argument("--model", dest="model", default=None, metavar="MODEL",
                          help="[codex] Requested model id, resolved against the OpenAI price "
                               "snapshot (default: gpt-5-codex)")
    p_import.add_argument("--price-snapshot", dest="price_snapshot", default=None, metavar="FILE",
                          help="[codex] Path to a versioned price snapshot (default: bundled openai)")
    p_import.add_argument("--billing-mode", dest="billing_mode", default=None,
                          choices=["per_token", "subscription"],
                          help="[codex] Override billing mode (default: derived from provider)")
    p_import.add_argument("--provider", dest="provider", default=None, metavar="PROVIDER",
                          help="[codex] Provider id for billing classification (default: openai)")
    p_import.add_argument("--raw", required=True, metavar="DIR_OR_FILE",
                          help="Path to the existing run directory, a single transcript file "
                               "(--source transcript), or an otel_logs.jsonl (--source otel)")
    p_import.add_argument("--label", default=None,
                          help="[transcript] Human label for the arm (defaults to the file/dir name)")
    p_import.add_argument("--n-total", dest="n_total", type=int, default=1,
                          help="[transcript] Test-suite size for a terminal --pass/--fail verdict")
    p_import.add_argument("--pass", dest="pass_bool", action="store_true", default=None,
                          help="[transcript] Record a terminal PASS verdict (e.g. functional_pass)")
    p_import.add_argument("--fail", dest="pass_bool", action="store_false",
                          help="[transcript] Record a terminal FAIL verdict")
    p_import.add_argument("--circt", dest="circt", action="store_true", default=None,
                          help="Treat as a CIRCT run (adds RTL-facts long-wait rules)")
    p_import.add_argument("--no-circt", dest="no_circt", action="store_true", default=False,
                          help="Force non-CIRCT classification")
    p_import.add_argument("--classifier-config", metavar="JSON", default=None,
                          help="Path to an ActivityConfig JSON (overrides the source default)")
    p_import.add_argument("--milestone-time", choices=["proportional", "wallclock"],
                          default="proportional",
                          help="Map self-check milestones by proportion of active wall, or raw offset")
    p_import.add_argument("--run-id", dest="run_id", default=None,
                          help="Override the trajectory run_id (default: run dir name)")
    p_import.add_argument("--out", metavar="PATH", default=None,
                          help="Where to write trajectory.json (default: <raw>/trajectory.json)")
    p_import.add_argument("--into", metavar="AET_RUN_DIR", default=None,
                          help="Also materialize a canonical aet run dir (logs/ + trajectory.json)")
    p_import.set_defaults(func=_cmd_import)

    # ------------------------------------------------------------------
    # otel-sink — capture Claude Code telemetry (OTLP/HTTP JSON) to a JSONL file
    # ------------------------------------------------------------------
    p_otel = subparsers.add_parser(
        "otel-sink",
        help="Run a minimal OTLP receiver that captures Claude Code telemetry to otel_logs.jsonl "
             "(then import with `aet import --source otel`).",
    )
    p_otel.add_argument("--port", type=int, required=True, help="TCP port to listen on")
    p_otel.add_argument("--out", required=True, metavar="JSONL",
                        help="Output path — one line per received OTLP envelope")
    p_otel.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p_otel.set_defaults(func=_cmd_otel_sink)

    # ------------------------------------------------------------------
    # plot — render a run's trajectory (requires the [viz] extra)
    # ------------------------------------------------------------------
    p_plot = subparsers.add_parser(
        "plot",
        help="Plot a run's trajectory (requires the [viz] extra)",
    )
    p_plot.add_argument("run", metavar="RUN_OR_JSON",
                        help="A run directory or a trajectory.json file")
    p_plot.add_argument("--kind", default="trajectory",
                        choices=["trajectory", "comparison", "rate-panels",
                                 "cost-vs-time", "tests-facets"],
                        help="Figure kind (default: trajectory). The comparison kinds use "
                             "run + --comparison as the arm list.")
    p_plot.add_argument("--out", metavar="PNG", default=None,
                        help="Output image path (default: <run>.<kind>.png). A .png also writes .svg")
    p_plot.add_argument("--comparison", nargs="+", metavar="RUN", default=None,
                        help="Additional runs/trajectories to stack/compare (the other arms)")
    p_plot.add_argument("--linear-tokens", dest="linear_tokens", action="store_true",
                        default=False, help="Use a linear token axis (default: log)")
    p_plot.add_argument("--no-spend", dest="no_spend", action="store_true", default=False,
                        help="Hide the cumulative-spend twin axis")
    p_plot.add_argument("--split-cache", dest="split_cache", action="store_true", default=False,
                        help="Draw cache reads and cache writes as separate lines instead of "
                             "their sum (--kind trajectory and rate-panels)")
    p_plot.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200)")
    p_plot.set_defaults(func=_cmd_plot)

    # ------------------------------------------------------------------
    # plot-sessions — point at raw Claude sessions → comparison figures
    # ------------------------------------------------------------------
    p_ps = subparsers.add_parser(
        "plot-sessions",
        help="Point at raw Claude session transcripts and render the comparison figures in one step",
    )
    p_ps.add_argument("sessions", nargs="+", metavar="SESSION",
                      help="Transcript *.jsonl files or dirs of session logs (one arm each)")
    p_ps.add_argument("--out", metavar="DIR", default=None,
                      help="Output directory for the figures (default: cwd)")
    p_ps.add_argument("--kinds", nargs="+", choices=list(_SESSION_KINDS), default=None,
                      help="Which figures to render (default: all three)")
    p_ps.add_argument("--pass-all", dest="pass_all", action="store_true", default=False,
                      help="Mark every session as a terminal PASS (for a tests-facets demo)")
    p_ps.add_argument("--n-total", dest="n_total", type=int, default=1,
                      help="Test-suite size for a terminal verdict (default: 1)")
    p_ps.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200)")
    p_ps.set_defaults(func=_cmd_plot_sessions)

    # ------------------------------------------------------------------
    # run — a sandboxed, recorded, rate-limit-resilient agent invocation
    # ------------------------------------------------------------------
    p_run = subparsers.add_parser(
        "run",
        help="Launch a sandboxed, recorded agent run (survives the 5-hour limit; --resume to continue)",
    )
    p_run.add_argument("--task", metavar="FILE_OR_TEXT", default=None,
                       help="Task prompt file (or inline text) fed to the agent on stdin")
    p_run.add_argument("--workspace", metavar="DIR", default=None,
                       help="The agent's writable working dir (the only writable path in the sandbox)")
    p_run.add_argument("--into", metavar="AET_RUN_DIR", default=None,
                       help="Where to materialize the aet run (default: <workspace>_aetrun)")
    p_run.add_argument("--resume", metavar="RUN_DIR", default=None,
                       help="Resume a previously rate-limited run from its recorded session")
    p_run.add_argument("--label", default=None, help="Run label (default: run dir name)")
    p_run.add_argument("--model", default="claude-opus-4-8", help="Model id (default: claude-opus-4-8)")
    p_run.add_argument("--sandbox", choices=["bwrap", "none"], default="bwrap",
                       help="Isolation backend (default: bwrap deny-by-default)")
    p_run.add_argument("--allow", nargs="+", metavar="PATH", default=None,
                       help="Read-only paths granted into the sandbox (inputs + in-repo tools)")
    p_run.add_argument("--deny", nargs="+", metavar="PATH", default=None,
                       help="Paths masked even if under an --allow (answers / sibling runs)")
    p_run.add_argument("--extra-binds", dest="extra_binds", nargs="+", metavar="PATH", default=None,
                       help="Toolchain dirs outside the repo to bind read-only")
    # The three below complete the pass-through: SandboxSpec has supported them since it was written,
    # but `aet run` had no way to reach them, so a caller needing any one of them had to rebuild the
    # bwrap policy itself. Each covers a case that `--allow`/`--deny` provably cannot:
    #   --rw-binds    an agent CLI keeps session state and a config file under $HOME; read-only there
    #                 means it cannot authenticate at all.
    #   --mask-files  withholding a single FILE inside an otherwise-granted directory. `--deny` tmpfs's
    #                 a directory, so it is all-or-nothing for a docs tree with one answer key in it.
    #   --unsetenv    a nested agent session inherits variables that re-route it into the PARENT's
    #                 session, silently joining the run it was meant to be isolated from.
    p_run.add_argument("--rw-binds", dest="rw_binds", nargs="+", metavar="PATH", default=None,
                       help="Read-WRITE binds outside the workspace (an agent CLI's state dir and "
                            "config file under an otherwise read-only home)")
    p_run.add_argument("--mask-files", dest="mask_files", nargs="+", metavar="PATH", default=None,
                       help="Individual files overlaid with /dev/null — present but empty, so "
                            "withholding is not inferable from an ENOENT (per-file answer keys)")
    p_run.add_argument("--unshare-net", dest="unshare_net", action="store_true", default=False,
                       help="Run with NO network namespace. A filesystem allow-list is not an "
                            "information boundary while the network is up: an agent that can reach "
                            "the internet can fetch what the allow-list withheld and publish what "
                            "it protected. Off by default because most agents need the API.")
    p_run.add_argument("--unsetenv", dest="unsetenv", nargs="+", metavar="VAR", default=None,
                       help="Environment variables cleared inside the sandbox (nested-session "
                            "variables that would re-route a child agent into this session)")
    p_run.add_argument("--env-prefix", dest="env_prefix", default="",
                       help="Shell export prefix for the toolchain (PATH/LD_LIBRARY_PATH/...)")
    p_run.add_argument("--allow-unsandboxed", dest="allow_unsandboxed", action="store_true",
                       default=False, help="Permit a real run with --sandbox none (guard override)")
    p_run.add_argument("--agent-cmd", dest="agent_cmd", default=None,
                       help="Override the claude command (a shell cmd emitting stream-json; for "
                            "custom launchers / dummy runs)")
    p_run.add_argument("--poll-seconds", dest="poll_seconds", type=float, default=1200.0,
                       help="Rate-limit poll interval when the reset epoch is unknown (default: 1200)")
    p_run.add_argument("--max-rate-limit-waits", dest="max_rate_limit_waits", type=int, default=3,
                       help="Max five-hour waits before leaving the run unfinished (default: 3)")
    p_run.set_defaults(func=_cmd_run)

    # ------------------------------------------------------------------
    # monitor — live activity view of an in-flight agent session
    # ------------------------------------------------------------------
    p_monitor = subparsers.add_parser(
        "monitor",
        help="Live activity monitor tailing an in-flight stream-json transcript",
    )
    p_monitor.add_argument("--attach", required=True, metavar="TRANSCRIPT",
                           help="Path to the stream-json transcript to tail")
    p_monitor.add_argument("--preset", choices=["generic", "capsule-bench"], default="generic",
                           help="Activity classifier preset (default: generic)")
    p_monitor.add_argument("--circt", action="store_true", default=False,
                           help="With --preset capsule-bench, add the CIRCT long-wait rules")
    p_monitor.add_argument("--classifier-config", metavar="JSON", default=None,
                           help="Path to an ActivityConfig JSON (overrides --preset)")
    p_monitor.add_argument("--selfcheck", metavar="LOG", default=None,
                           help="Path to a selfcheck_log.jsonl for a live tests-passed readout")
    p_monitor.add_argument("--emit-json", dest="emit_json", metavar="PATH", default=None,
                           help="Write a final trajectory snapshot as JSON")
    p_monitor.add_argument("--plot", metavar="PNG", default=None,
                           help="Render a plot at the end (requires the [viz] extra)")
    p_monitor.add_argument("--interval", type=float, default=0.5,
                           help="Poll interval in seconds while following (default: 0.5)")
    p_monitor.add_argument("--flush-every", dest="flush_every", type=int, default=5,
                           help="Refresh the status line every N transcript lines (default: 5)")
    p_monitor.add_argument("--no-follow", dest="no_follow", action="store_true", default=False,
                           help="Parse the existing transcript once and exit (no tailing)")
    p_monitor.add_argument("--max-seconds", dest="max_seconds", type=float, default=None,
                           help="Stop following after this many seconds")
    p_monitor.set_defaults(func=_cmd_monitor)

    # ------------------------------------------------------------------
    # show
    # ------------------------------------------------------------------
    p_show = subparsers.add_parser(
        "show",
        help="Show all captured data for a single run",
    )
    p_show.add_argument("run_path", metavar="RUN_PATH", help="Path to the run directory")
    p_show.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )
    p_show.set_defaults(func=_cmd_show)

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    try:
        args.func(args)
    except (AetError, SuiteNotFoundError, RunAlreadyExistsError) as e:
        print(f"[aet] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
