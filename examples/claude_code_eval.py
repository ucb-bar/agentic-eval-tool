"""
Claude Code end-to-end aet example.

Runs one Claude Code agent invocation through the full aet pipeline:
  init_run → invoke_agent → validate → collect_metrics → tracking

Traces appear in SigNoz under Services → "aet" with the hierarchy:
  invoke_workflow(claude_code_eval)
    invoke_agent(claude-code)
    execute_tool(validate)

Usage:
    # Local only (no services, no API key)
    python examples/claude_code_eval.py --dry-run

    # With SigNoz traces (start SigNoz first via observability/install-signoz.sh)
    python examples/claude_code_eval.py \\
        --otel-endpoint http://localhost:4318 --tracking full --dry-run

    # Full stack with SigNoz + MLflow
    python examples/claude_code_eval.py \\
        --otel-endpoint http://localhost:4318 \\
        --mlflow-uri http://localhost:5001 \\
        --tracking full
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

# Allow running from repo root without installing the package
_SRC = Path(__file__).parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aet.core.run_paths import RunPaths
from aet.core.run_spec import RunSpec
from aet.suites import get_suite
from aet.tracking.run_logger import EvalRunLogger

_DEFAULT_TASK = (
    "List 5 key properties of a well-designed compiler intermediate representation "
    "and explain why each matters. Format as a markdown document with a section per property."
)

_DRY_RUN_OUTPUT = """\
# Compiler IR Design Properties (Dry-run)

> **Note:** Dry-run mode — no `claude` CLI was invoked. This is synthetic output.

## 1. Strong Normalization
A well-designed IR has a canonical form for every program, so analysis passes
can make deterministic structural assumptions without handling equivalent-but-different
representations of the same semantics.

## 2. Explicit Control Flow
All branches, loops, and jumps are first-class IR constructs. This makes
control-flow graphs trivially derivable and enables precise data-flow analysis
without heuristic approximation.

## 3. SSA Form
Single Static Assignment ensures every value has exactly one definition site,
making def-use chains enumerable in O(1). Phi nodes at join points encode the
merging of values without aliasing.

## 4. Type Safety
A typed IR rejects structurally ill-formed programs before codegen. Types
propagate lowering constraints (alignment, width, address space) so mismatches
surface close to their origin rather than as corrupt machine code.

## 5. Source Location Preservation
Debug info (file, line, column) is threaded through every IR transformation.
When compilation fails or a runtime error occurs, the user sees their source,
not a mangled lowered form.
"""


def run_claude_code_eval(
    *,
    project_root: Path,
    tracking_mode: str = "local",
    mlflow_tracking_uri: str | None = None,
    experiment_name: str | None = None,
    otel_endpoint: str | None = None,
    task: str = _DEFAULT_TASK,
    dry_run: bool = False,
    seed: int = 1,
) -> dict:
    """
    Run one Claude Code eval. Returns the validation report dict.

    Always safe to call — never raises. Errors are logged and reflected
    in the returned report's `status` field.
    """
    project_root = Path(project_root)
    run_id = f"{date.today().isoformat()}_claude_code_seed{seed:03d}"
    suite_name = "default"

    spec = RunSpec(
        project=project_root.name or "aet-example",
        suite=suite_name,
        method="claude_code",
        seed=seed,
        run_id=run_id,
        project_root=project_root,
        tracking_mode=tracking_mode,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        otel_endpoint=otel_endpoint,
    )
    paths = RunPaths.from_spec(spec, run_id)
    paths.run_path.mkdir(parents=True, exist_ok=True)

    logger = EvalRunLogger.start(
        project=spec.project,
        suite=spec.suite,
        target="compiler_ir",
        method=spec.method,
        seed=spec.seed,
        run_id=run_id,
        run_path=paths.run_path,
        tracking_mode=tracking_mode,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        otel_endpoint=otel_endpoint,
    )

    suite = get_suite(suite_name)

    with logger.start_run_span("claude_code_eval"):
        # ── 1. init_run ────────────────────────────────────────────────
        suite.init_run(spec, paths, logger)
        logger.log_param("task_length_chars", len(task))
        logger.log_param("dry_run", dry_run)

        # ── 2. invoke agent ────────────────────────────────────────────
        output_text, duration_s = _run_agent(task, dry_run, otel_endpoint, logger)

        paths.generated.mkdir(parents=True, exist_ok=True)
        output_file = paths.generated / "claude_output.md"
        output_file.write_text(output_text)
        logger.log_artifact(output_file)
        logger.log_metric("output_chars", len(output_text))
        logger.log_metric("duration_s", duration_s)
        # Rough token estimate (4 chars ≈ 1 token)
        logger.record_llm_call(
            duration_s,
            input_tokens=len(task) // 4,
            output_tokens=len(output_text) // 4,
        )

        # ── 3. validate ────────────────────────────────────────────────
        with logger.start_tool_span("validate"):
            report = suite.validate(spec, paths, logger)

        # ── 4. collect_metrics ─────────────────────────────────────────
        suite.collect_metrics(spec, paths, logger)

        score = {"pass": 1.0, "partial": 0.5}.get(report.get("status", "fail"), 0.0)
        logger.log_evaluation_result("output_quality", score, report.get("status", "unknown"))
        logger.log_metric("validation_errors", len(report.get("errors", [])))

    logger.finish(status=report.get("status", "unknown"))
    _print_summary(logger, otel_endpoint, run_id, report, dry_run)
    return report


def _run_agent(
    task: str,
    dry_run: bool,
    otel_endpoint: str | None,
    logger: EvalRunLogger,
) -> tuple[str, float]:
    with logger.start_agent_span("claude-code", model="claude", provider="anthropic"):
        if dry_run:
            logger.log_event("agent.dry_run", {"message": "synthetic output, no API call"})
            return _DRY_RUN_OUTPUT, 0.05

        return _invoke_claude_cli(task, otel_endpoint, logger)


def _invoke_claude_cli(
    task: str,
    otel_endpoint: str | None,
    logger: EvalRunLogger,
) -> tuple[str, float]:
    """Call `claude --print --no-verbose <task>` with TRACEPARENT wired in."""
    env = os.environ.copy()

    traceparent = logger.get_traceparent_for_subprocess()
    if traceparent:
        env["TRACEPARENT"] = traceparent

    if otel_endpoint:
        env.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", otel_endpoint)
        env.setdefault("OTEL_SERVICE_NAME", "claude-code")

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            ["claude", "--print", "--no-verbose", task],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "claude CLI not found. "
            "Install: npm install -g @anthropic-ai/claude-code  "
            "or use --dry-run to test without it."
        )
    duration_s = time.monotonic() - t0
    if result.returncode != 0:
        raise RuntimeError(f"claude exited {result.returncode}: {result.stderr[:300]}")
    return result.stdout, duration_s


def _print_summary(
    logger: EvalRunLogger,
    otel_endpoint: str | None,
    run_id: str,
    report: dict,
    dry_run: bool,
) -> None:
    status = report.get("status", "unknown")
    print(f"\n[aet] run: {run_id}  status: {status}")
    if dry_run:
        print("       mode: dry-run (no claude CLI invoked)")
    errors = report.get("errors", [])
    if errors:
        for e in errors:
            print(f"       error: {e}")
    if url := logger.mlflow_run_url:
        print(f"       mlflow:  {url}")
    if trace_id := logger.otel_trace_id:
        print(f"       trace:   {trace_id}")
        if otel_endpoint and "localhost" in otel_endpoint:
            print("       signoz:  http://localhost:3301  →  Services  →  aet")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Claude Code end-to-end aet example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--project-root", type=Path, default=Path("/tmp/aet-claude-example"),
                   help="where run artifacts are written (default: /tmp/aet-claude-example)")
    p.add_argument("--tracking", dest="tracking_mode", default="local",
                   choices=["local", "mlflow", "full"],
                   help="tracking backend (default: local)")
    p.add_argument("--mlflow-uri", dest="mlflow_tracking_uri", default=None,
                   help="MLflow tracking URI, e.g. http://localhost:5001")
    p.add_argument("--experiment-name", default="aet-claude-code",
                   help="MLflow experiment name")
    p.add_argument("--otel-endpoint", default=None,
                   help="OTLP HTTP endpoint, e.g. http://localhost:4318")
    p.add_argument("--task", default=_DEFAULT_TASK,
                   help="prompt to send to claude")
    p.add_argument("--dry-run", action="store_true",
                   help="skip real claude invocation, use synthetic output")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = run_claude_code_eval(
        project_root=args.project_root,
        tracking_mode=args.tracking_mode,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        otel_endpoint=args.otel_endpoint,
        task=args.task,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    sys.exit(0 if report.get("status") in ("pass", "partial") else 1)
