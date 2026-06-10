"""
Claude Code end-to-end aet example — comprehensive instrumentation.

Captures per the Anthropic OTel GenAI semconv (Development, 2025):
  - gen_ai.usage.input_tokens / output_tokens (per turn and total)
  - gen_ai.usage.cache_creation.input_tokens / cache_read.input_tokens
  - gen_ai.conversation.id (session ID for /resume replay)
  - gen_ai.response.model
  - gen_ai.response.finish_reasons
  - gen_ai.user.message  (prompt event, opt-in via --capture-content)
  - gen_ai.assistant.message  (completion event, opt-in)
  - gen_ai.tool.call  (one event per Bash / Read / Write / Edit call)
  - gen_ai.evaluation.result  (task achievement score)
  - aet.agent.num_turns, aet.agent.cost_usd, aet.agent.permission_mode
  - aet.agent.tool_call_count, aet.agent.tool_error_count
  - aet.agent.duration_ms, aet.agent.duration_api_ms

Uses `claude --print --output-format stream-json` which emits one JSON
event per line, giving per-turn token counts, per-tool-call names/inputs/
results, and a final result block with cost_usd and num_turns.

Usage:
    # Local only — no services, no API key needed
    python examples/claude_code_eval.py --dry-run

    # SigNoz traces (start SigNoz first)
    python examples/claude_code_eval.py \\
        --otel-endpoint http://localhost:4318 --tracking full --dry-run

    # Full stack with real Claude Code
    export ANTHROPIC_API_KEY=sk-...
    python examples/claude_code_eval.py \\
        --otel-endpoint http://localhost:4318 \\
        --mlflow-uri http://localhost:5001 \\
        --tracking full \\
        --capture-content \\
        --skip-permissions
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aet.core.run_paths import RunPaths
from aet.core.run_spec import RunSpec
from aet.suites import get_suite
from aet.tracking.claude_stream import ClaudeStreamResult, parse_stream
from aet.tracking.run_logger import EvalRunLogger

_DEFAULT_TASK = (
    "List 5 key properties of a well-designed compiler intermediate representation "
    "and explain why each matters. Format as a markdown document with a section per property."
)

# Synthetic stream-json mimicking real claude --output-format stream-json output.
# Exercises the full parser: 2 turns, 1 Write tool call, token counts, cost.
_DRY_RUN_STREAM = "\n".join([
    json.dumps({"type": "system", "subtype": "init",
                "session_id": "dry-run-session-001",
                "tools": ["Bash", "Read", "Write", "Edit"],
                "mcp_servers": []}),
    json.dumps({"type": "assistant", "message": {
        "id": "msg_dry001", "type": "message", "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll write a markdown document on compiler IR design."},
            {"type": "tool_use", "id": "toolu_dry001", "name": "Write",
             "input": {"file_path": "generated/claude_output.md",
                       "content": "# Compiler IR Design Properties (Dry-run)\n\n> Dry-run synthetic output.\n\n## 1. Strong Normalization\nCanonical form simplifies analysis.\n\n## 2. Explicit Control Flow\nAll branches are first-class, enabling precise DFA.\n\n## 3. SSA Form\nDef-use chains are O(1) enumerable.\n\n## 4. Type Safety\nRejects ill-formed programs before codegen.\n\n## 5. Source Location Preservation\nDebug info threaded through every lowering step.\n"}},
        ],
        "model": "claude-opus-4-5",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 166, "output_tokens": 78,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}}),
    json.dumps({"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_dry001",
         "content": [{"type": "text", "text": "File created successfully at: generated/claude_output.md"}]}]}}),
    json.dumps({"type": "assistant", "message": {
        "id": "msg_dry002", "type": "message", "role": "assistant",
        "content": [{"type": "text",
                     "text": "I've written the document covering 5 key compiler IR properties."}],
        "model": "claude-opus-4-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 280, "output_tokens": 18,
                  "cache_creation_input_tokens": 512, "cache_read_input_tokens": 0}}}),
    json.dumps({"type": "result", "subtype": "success",
                "cost_usd": 0.0,
                "duration_ms": 50,
                "duration_api_ms": 0,
                "num_turns": 2,
                "result": "I've written the document covering 5 key compiler IR properties.",
                "session_id": "dry-run-session-001"}),
])


def run_claude_code_eval(
    *,
    project_root: Path,
    tracking_mode: str = "local",
    mlflow_tracking_uri: str | None = None,
    experiment_name: str | None = None,
    otel_endpoint: str | None = None,
    task: str = _DEFAULT_TASK,
    dry_run: bool = False,
    skip_permissions: bool = False,
    capture_content: bool = False,
    seed: int = 1,
) -> dict:
    """
    Run one Claude Code eval. Returns the validation report dict.
    Never raises — errors are captured in the report status.
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

    permission_mode = "dangerously-skip-permissions" if skip_permissions else "default"

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
        logger.log_permission_mode(permission_mode, approvals_required=0)

        if capture_content:
            logger.log_prompt(task, role="user")

        # ── 2. invoke agent ────────────────────────────────────────────
        with logger.start_agent_span("claude-code", model="claude", provider="anthropic"):
            stream_result = _run_agent(
                task, paths, dry_run, skip_permissions, otel_endpoint, logger
            )

        # ── 3. record comprehensive metrics from stream ────────────────
        _record_stream_metrics(stream_result, logger, capture_content)

        # ── 4. validate ────────────────────────────────────────────────
        with logger.start_tool_span("validate"):
            report = suite.validate(spec, paths, logger)

        # ── 5. collect_metrics ─────────────────────────────────────────
        suite.collect_metrics(spec, paths, logger)

        # ── 6. task achievement score ──────────────────────────────────
        achieved, score = _score_task_achievement(stream_result, report)
        logger.log_task_achievement(achieved, score,
                                    rationale=f"claude_success={stream_result.success}, "
                                              f"validation={report.get('status')}, "
                                              f"turns={stream_result.num_turns}")

        logger.log_metric("validation_errors", len(report.get("errors", [])))

    logger.finish(status=report.get("status", "unknown"))
    _print_summary(logger, otel_endpoint, run_id, report, stream_result, dry_run)
    return report


def _run_agent(
    task: str,
    paths: RunPaths,
    dry_run: bool,
    skip_permissions: bool,
    otel_endpoint: str | None,
    logger: EvalRunLogger,
) -> ClaudeStreamResult:
    if dry_run:
        logger.log_event("agent.dry_run", {"message": "synthetic stream output"})
        result = parse_stream(_DRY_RUN_STREAM)
        # Write the generated file so validation sees content
        paths.generated.mkdir(parents=True, exist_ok=True)
        out = paths.generated / "claude_output.md"
        write_tc = next((tc for tc in result.tool_calls if tc.name == "Write"), None)
        if write_tc:
            content = write_tc.input.get("content", "")
            out.write_text(content)
        return result

    stream_output, _ = _invoke_claude_cli(task, paths, skip_permissions, otel_endpoint, logger)
    result = parse_stream(stream_output)

    # Write the output file if claude didn't (fallback for non-tool-use runs)
    if result.result_text:
        paths.generated.mkdir(parents=True, exist_ok=True)
        out = paths.generated / "claude_output.md"
        if not out.exists():
            out.write_text(result.result_text)

    return result


def _invoke_claude_cli(
    task: str,
    paths: RunPaths,
    skip_permissions: bool,
    otel_endpoint: str | None,
    logger: EvalRunLogger,
) -> tuple[str, float]:
    """Call `claude --print --output-format stream-json` and return (stream_text, duration_s)."""
    env = os.environ.copy()

    traceparent = logger.get_traceparent_for_subprocess()
    if traceparent:
        env["TRACEPARENT"] = traceparent
    if otel_endpoint:
        env.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", otel_endpoint)
        env.setdefault("OTEL_SERVICE_NAME", "claude-code")

    cmd = ["claude", "--print", "--output-format", "stream-json"]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    # Run in the generated dir so file tools land there automatically
    cmd.append(task)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(paths.generated),
            timeout=300,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "claude CLI not found. "
            "Install: npm install -g @anthropic-ai/claude-code  "
            "or use --dry-run to skip."
        )
    duration_s = time.monotonic() - t0

    if proc.returncode not in (0, 1):
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr[:400]}")

    return proc.stdout, duration_s


def _record_stream_metrics(
    r: ClaudeStreamResult,
    logger: EvalRunLogger,
    capture_content: bool,
) -> None:
    """Push all stream-derived metrics, tokens, tool calls to tracking."""
    # ── token usage (all buckets per Anthropic semconv) ────────────────
    logger.log_token_usage(
        input_tokens=r.total_input_tokens,
        output_tokens=r.total_output_tokens,
        cache_creation_tokens=r.total_cache_creation_tokens,
        cache_read_tokens=r.total_cache_read_tokens,
        model=r.model,
    )

    # ── per-turn token series (convergence curves in MLflow) ───────────
    for t in r.turn_usage:
        step = t.turn - 1
        logger.log_metric_step("turn.input_tokens", t.total_input_tokens, step=step)
        logger.log_metric_step("turn.output_tokens", t.output_tokens, step=step)
        if t.cache_creation_input_tokens:
            logger.log_metric_step("turn.cache_creation_tokens", t.cache_creation_input_tokens, step=step)
        if t.cache_read_input_tokens:
            logger.log_metric_step("turn.cache_read_tokens", t.cache_read_input_tokens, step=step)

    # ── cost, turns, duration ──────────────────────────────────────────
    logger.log_cost(r.cost_usd, model=r.model)
    logger.log_agent_turns(r.num_turns)
    logger.log_metric("aet.agent.duration_ms", r.duration_ms)
    logger.log_metric("aet.agent.duration_api_ms", r.duration_api_ms)
    logger.log_metric("aet.agent.tool_call_count", r.tool_call_count)
    logger.log_metric("aet.agent.tool_error_count", r.tool_error_count)

    # ── session ID for replay ─────────────────────────────────────────
    if r.session_id:
        logger.log_session_id(r.session_id)

    # ── model ─────────────────────────────────────────────────────────
    if r.model:
        logger.log_param("gen_ai.response.model", r.model)

    # ── per-tool-call events ──────────────────────────────────────────
    for tc in r.tool_calls:
        logger.log_tool_call_event(
            tool_name=tc.name,
            input_summary=tc.input_summary(),
            result_summary=tc.result_summary(),
            is_error=tc.is_error,
            tool_call_id=tc.tool_use_id,
        )

    # ── completion capture (opt-in — may contain sensitive data) ──────
    if capture_content and r.result_text:
        logger.log_completion(r.result_text)

    # ── unique tools used ─────────────────────────────────────────────
    if r.unique_tools_used:
        logger.log_param("tools_used", ",".join(r.unique_tools_used))


def _score_task_achievement(
    r: ClaudeStreamResult,
    report: dict,
) -> tuple[bool, float]:
    """
    Simple heuristic task achievement score.
    0.0 = claude failed or no output
    0.5 = claude succeeded but validation found issues
    1.0 = claude succeeded and validation passed
    """
    if not r.success or not r.result_text.strip():
        return False, 0.0
    if report.get("status") == "pass":
        return True, 1.0
    if report.get("status") == "partial":
        return True, 0.5
    return False, 0.2


def _print_summary(
    logger: EvalRunLogger,
    otel_endpoint: str | None,
    run_id: str,
    report: dict,
    stream: ClaudeStreamResult,
    dry_run: bool,
) -> None:
    status = report.get("status", "unknown")
    print(f"\n[aet] run: {run_id}  status: {status}")
    if dry_run:
        print("       mode:     dry-run (synthetic stream output)")
    if stream.model:
        print(f"       model:    {stream.model}")
    if stream.num_turns:
        print(f"       turns:    {stream.num_turns}")
    total_in = stream.total_input_tokens
    total_out = stream.total_output_tokens
    cache_read = stream.total_cache_read_tokens
    cache_create = stream.total_cache_creation_tokens
    if total_in or total_out:
        tokens_line = f"       tokens:   {total_in} in / {total_out} out"
        if cache_read:
            tokens_line += f"  (cache_read={cache_read})"
        if cache_create:
            tokens_line += f"  (cache_create={cache_create})"
        print(tokens_line)
    if stream.cost_usd:
        print(f"       cost:     ${stream.cost_usd:.6f}")
    if stream.tool_call_count:
        tools = ", ".join(stream.unique_tools_used)
        print(f"       tools:    {stream.tool_call_count} calls ({tools})")
    if stream.session_id and stream.session_id != "dry-run-session-001":
        print(f"       session:  {stream.session_id}  (replay: claude --resume {stream.session_id})")
    if url := logger.mlflow_run_url:
        print(f"       mlflow:   {url}")
    if trace_id := logger.otel_trace_id:
        print(f"       trace:    {trace_id}")
        if otel_endpoint and "localhost" in otel_endpoint:
            print("       signoz:   http://localhost:8080  →  Services  →  aet")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Claude Code end-to-end aet example — comprehensive instrumentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--project-root", type=Path, default=Path("/tmp/aet-claude-example"))
    p.add_argument("--tracking", dest="tracking_mode", default="local",
                   choices=["local", "mlflow", "full"])
    p.add_argument("--mlflow-uri", dest="mlflow_tracking_uri", default=None)
    p.add_argument("--experiment-name", default="aet-claude-code")
    p.add_argument("--otel-endpoint", default=None,
                   help="OTLP HTTP endpoint, e.g. http://localhost:4318")
    p.add_argument("--task", default=_DEFAULT_TASK)
    p.add_argument("--dry-run", action="store_true",
                   help="Use synthetic stream output instead of calling claude CLI")
    p.add_argument("--skip-permissions", action="store_true",
                   help="Pass --dangerously-skip-permissions to claude (for automated sandboxes)")
    p.add_argument("--capture-content", action="store_true",
                   help="Record prompt and completion text in spans (opt-in; may contain PII)")
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
        skip_permissions=args.skip_permissions,
        capture_content=args.capture_content,
        seed=args.seed,
    )
    sys.exit(0 if report.get("status") in ("pass", "partial") else 1)
