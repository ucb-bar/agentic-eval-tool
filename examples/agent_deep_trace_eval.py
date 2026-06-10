"""agent_deep_trace_eval.py — squeeze every aet feature into one experiment.

What gets recorded
------------------
Per-turn (LLM call):
  - input / output / cache_read / cache_creation / reasoning tokens
  - cost_usd, wall-clock duration, API latency
  - which tools were invoked

Per-tool-call:
  - tool name, type (native / MCP), duration (tool_use → tool_result)
  - file paths touched, input summary, output size
  - error flag

File access map:
  - every file read / written / edited by the agent
  - turn and tool attribution per access

All of the above goes to:
  - OTel spans  (invoke_workflow → invoke_agent + span events)
  - MLflow      (per-turn step metrics, cost, tokens, tool counts)
  - Local JSONL (full audit trail)
  - CLI summary (ASCII tables)
  - Matplotlib  (4-panel analytics figure saved as PNG artifact)

Usage
-----
# local only (dry-run, no API):
.venv/bin/python examples/agent_deep_trace_eval.py --dry-run

# real run (needs claude session auth):
.venv/bin/python examples/agent_deep_trace_eval.py \\
    --tracking full --otel-endpoint http://localhost:4318

# with MLflow + SigNoz:
.venv/bin/python examples/agent_deep_trace_eval.py \\
    --tracking full \\
    --otel-endpoint http://localhost:4318 \\
    --mlflow-uri http://localhost:5001
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── aet imports ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aet.tracking import EvalRunLogger
from aet.tracking.claude_stream import (
    ClaudeStreamResult, ToolCall, TurnUsage,
    parse_timestamped_stream, parse_stream,
)

# ── Task ──────────────────────────────────────────────────────────────────────
TASK = """\
You are performing a deep analysis of the aet Python eval harness at /scratch/agustin/projects/aet.

Complete ALL of these steps in order:

STEP 1 — Find the largest Python source files (exclude __pycache__):
  find /scratch/agustin/projects/aet/src -name "*.py" -not -path "*__pycache__*" | \
    xargs wc -l 2>/dev/null | sort -rn | head -20

STEP 2 — Read the 3 largest source files (from step 1, not test files) and write
  a 2-sentence summary of each module's purpose.

STEP 3 — Find all technical debt markers:
  grep -rn "TODO\\|FIXME\\|HACK\\|XXX\\|NOQA\\|type: ignore" \
    /scratch/agustin/projects/aet/src --include="*.py" | head -30

STEP 4 — Count test functions per test file:
  for f in /scratch/agustin/projects/aet/tests/test_*.py; do \
    echo "$f: $(grep -c "^def test_" "$f" 2>/dev/null || echo 0)"; done

STEP 5 — Check /scratch/agustin/projects/aet/src/aet/__init__.py.
  If it does NOT already contain __version__, add exactly this line:
    __version__ = "0.1.0"

STEP 6 — Use the WebFetch tool to retrieve the OpenTelemetry GenAI semantic conventions
  summary from: https://opentelemetry.io/docs/specs/semconv/gen-ai/
  Extract the 3 most relevant attribute names for LLM observability.

STEP 7 — Write a comprehensive Markdown report to /tmp/aet-deep-analysis.md that includes:
  • A module table: | file | lines | purpose |
  • Technical debt list
  • Test count summary
  • Architecture notes: describe how the tracking layer (local/mlflow/otel backends) is designed
  • The 3 OTel GenAI semconv attributes you fetched
  • One paragraph on what improvements you would suggest

STEP 8 — Call the MCP tool mcp__aet-eval__list_eval_runs with
  project_root="/tmp/aet-deep-trace"
  to discover what aet eval runs have already been recorded there.
  Report the run count and run IDs in your response.

STEP 9 — Invoke the Skill tool with skill="terse" to switch to concise output mode,
  then append a one-sentence "## Exec summary" section to /tmp/aet-deep-analysis.md
  capturing the key finding from all 8 steps above.

Be thorough. Complete every step before finishing.
"""

# ── Dry-run synthetic stream ──────────────────────────────────────────────────
_DRY_STREAM = r"""
{"type":"system","subtype":"init","session_id":"dry-deep-session-001","tools":[],"mcp_servers":[{"name":"aet-eval","status":"connected","tools":["list_eval_runs","get_run_metrics"]}]}
{"type":"assistant","message":{"id":"msg_turn1","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":1850,"output_tokens":120,"cache_creation_input_tokens":1200,"cache_read_input_tokens":0},"stop_reason":null,"content":[{"type":"text","text":"I'll start by finding the largest Python files."},{"type":"tool_use","id":"tu_bash1","name":"Bash","input":{"command":"find /scratch/agustin/projects/aet/src -name '*.py' -not -path '*__pycache__*' | xargs wc -l 2>/dev/null | sort -rn | head -20"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_bash1","is_error":false,"content":[{"type":"text","text":"  668 /scratch/agustin/projects/aet/src/aet/tracking/otel_backend.py\n  415 /scratch/agustin/projects/aet/src/aet/tracking/run_logger.py\n  310 /scratch/agustin/projects/aet/src/aet/tracking/claude_stream.py\n  280 /scratch/agustin/projects/aet/src/aet/cli/main.py\n  234 /scratch/agustin/projects/aet/src/aet/tracking/local_backend.py"}]}]}}
{"type":"assistant","message":{"id":"msg_turn2","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":2100,"output_tokens":85,"cache_creation_input_tokens":0,"cache_read_input_tokens":1850},"stop_reason":null,"content":[{"type":"text","text":"Now reading the 3 largest files."},{"type":"tool_use","id":"tu_read1","name":"Read","input":{"file_path":"/scratch/agustin/projects/aet/src/aet/tracking/otel_backend.py"}},{"type":"tool_use","id":"tu_read2","name":"Read","input":{"file_path":"/scratch/agustin/projects/aet/src/aet/tracking/run_logger.py"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_read1","is_error":false,"content":[{"type":"text","text":"[OTel backend source — 668 lines of OpenTelemetry span/metric setup]"}]},{"type":"tool_result","tool_use_id":"tu_read2","is_error":false,"content":[{"type":"text","text":"[Run logger source — 415 lines delegating to local/mlflow/otel backends]"}]}]}}
{"type":"assistant","message":{"id":"msg_turn3","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":3200,"output_tokens":95,"cache_creation_input_tokens":0,"cache_read_input_tokens":2100},"stop_reason":null,"content":[{"type":"tool_use","id":"tu_read3","name":"Read","input":{"file_path":"/scratch/agustin/projects/aet/src/aet/tracking/claude_stream.py"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_read3","is_error":false,"content":[{"type":"text","text":"[Claude stream parser — 310 lines parsing stream-json JSONL]"}]}]}}
{"type":"assistant","message":{"id":"msg_turn4","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":3800,"output_tokens":70,"cache_creation_input_tokens":0,"cache_read_input_tokens":3200},"stop_reason":null,"content":[{"type":"tool_use","id":"tu_bash2","name":"Bash","input":{"command":"grep -rn \"TODO\\|FIXME\\|HACK\\|XXX\" /scratch/agustin/projects/aet/src --include=\"*.py\" | head -30"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_bash2","is_error":false,"content":[{"type":"text","text":"/scratch/agustin/projects/aet/src/aet/tracking/otel_backend.py:142: # TODO: dedup histogram instrument names\n/scratch/agustin/projects/aet/src/aet/cli/main.py:89: # FIXME: --output flag collides with --output-format"}]}]}}
{"type":"assistant","message":{"id":"msg_turn5","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":4100,"output_tokens":60,"cache_creation_input_tokens":0,"cache_read_input_tokens":3800},"stop_reason":null,"content":[{"type":"tool_use","id":"tu_bash3","name":"Bash","input":{"command":"for f in /scratch/agustin/projects/aet/tests/test_*.py; do echo \"$f: $(grep -c '^def test_' \"$f\" 2>/dev/null || echo 0)\"; done"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_bash3","is_error":false,"content":[{"type":"text","text":"tests/test_mlflow_backend.py: 8\ntests/test_otel_backend.py: 12\ntests/test_run_logger.py: 22\ntests/test_claude_stream.py: 9\ntests/test_step_metrics.py: 5\ntests/test_validation.py: 14"}]}]}}
{"type":"assistant","message":{"id":"msg_turn6","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":4500,"output_tokens":40,"cache_creation_input_tokens":0,"cache_read_input_tokens":4100},"stop_reason":null,"content":[{"type":"tool_use","id":"tu_read4","name":"Read","input":{"file_path":"/scratch/agustin/projects/aet/src/aet/__init__.py"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_read4","is_error":false,"content":[{"type":"text","text":"from aet.tracking import EvalRunLogger\n"}]}]}}
{"type":"assistant","message":{"id":"msg_turn7","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":4800,"output_tokens":55,"cache_creation_input_tokens":0,"cache_read_input_tokens":4500},"stop_reason":null,"content":[{"type":"text","text":"Adding __version__ since it's missing."},{"type":"tool_use","id":"tu_edit1","name":"Edit","input":{"file_path":"/scratch/agustin/projects/aet/src/aet/__init__.py","old_string":"from aet.tracking import EvalRunLogger","new_string":"__version__ = \"0.1.0\"\n\nfrom aet.tracking import EvalRunLogger"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_edit1","is_error":false,"content":[{"type":"text","text":"The file /scratch/agustin/projects/aet/src/aet/__init__.py has been updated successfully."}]}]}}
{"type":"assistant","message":{"id":"msg_turn8","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":5200,"output_tokens":65,"cache_creation_input_tokens":0,"cache_read_input_tokens":4800},"stop_reason":null,"content":[{"type":"text","text":"Fetching OTel semconv page."},{"type":"tool_use","id":"tu_web1","name":"WebFetch","input":{"url":"https://opentelemetry.io/docs/specs/semconv/gen-ai/","prompt":"List the 3 most important GenAI semantic convention attribute names"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_web1","is_error":false,"content":[{"type":"text","text":"Key attributes: gen_ai.system, gen_ai.request.model, gen_ai.usage.input_tokens, gen_ai.usage.output_tokens"}]}]}}
{"type":"assistant","message":{"id":"msg_turn9","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":5600,"output_tokens":55,"cache_creation_input_tokens":0,"cache_read_input_tokens":5200},"stop_reason":null,"content":[{"type":"text","text":"Now writing the final report."},{"type":"tool_use","id":"tu_write1","name":"Write","input":{"file_path":"/tmp/aet-deep-analysis.md","content":"# aet Codebase Analysis\n\n## Module Summary\n\n| File | Lines | Purpose |\n|------|-------|---------|\n| otel_backend.py | 668 | OpenTelemetry span and metrics setup — TracerProvider, MeterProvider, span factories |\n| run_logger.py | 415 | Unified tracking API delegating to local/MLflow/OTel backends |\n| claude_stream.py | 310 | Parser for Claude Code stream-json JSONL output |\n| cli/main.py | 280 | CLI entry-points: aet run, runs, show commands |\n| local_backend.py | 234 | Filesystem JSONL-based tracking (always active, zero deps) |\n\n## Technical Debt\n\n- otel_backend.py:142: TODO: dedup histogram instrument names\n- cli/main.py:89: FIXME: --output flag collision\n\n## Test Coverage\n\n| File | Tests |\n|------|-------|\n| test_run_logger.py | 22 |\n| **Total** | **70** |\n\n## OTel GenAI Semconv (fetched)\n\n- gen_ai.system\n- gen_ai.request.model\n- gen_ai.usage.input_tokens\n\n## Architecture Notes\n\nThree-backend fanout: local JSONL (always), MLflow (optional), OTel (optional).\n\n## Suggested Improvements\n\nAdd prompt fingerprinting and per-token throughput metrics.\n"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_write1","is_error":false,"content":[{"type":"text","text":"File written: /tmp/aet-deep-analysis.md"}]}]}}
{"type":"assistant","message":{"id":"msg_turn10","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":980,"output_tokens":42,"cache_creation_input_tokens":0,"cache_read_input_tokens":5600},"stop_reason":null,"content":[{"type":"text","text":"Calling the aet-eval MCP server to list recorded eval runs."},{"type":"tool_use","id":"tu_mcp1","name":"mcp__aet-eval__list_eval_runs","input":{"project_root":"/tmp/aet-deep-trace"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_mcp1","is_error":false,"content":[{"type":"text","text":"{\"runs\": [\"deep_trace_c3872ebe6f\", \"deep_trace_4ec932ce26\"], \"total\": 2, \"root\": \"/tmp/aet-deep-trace\"}"}]}]}}
{"type":"assistant","message":{"id":"msg_turn11","role":"assistant","model":"claude-sonnet-4-6","usage":{"input_tokens":1120,"output_tokens":38,"cache_creation_input_tokens":0,"cache_read_input_tokens":980},"stop_reason":"end_turn","content":[{"type":"text","text":"Found 2 runs. Now invoking terse skill and appending exec summary."},{"type":"tool_use","id":"tu_skill1","name":"Skill","input":{"skill":"terse"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_skill1","is_error":false,"content":[{"type":"text","text":"Terse mode activated."}]}]}}
{"type":"result","subtype":"success","is_error":false,"result":"Analysis complete. All 9 steps done: codebase survey, debt scan, test count, OTel semconv fetch, report written, 2 eval runs listed via MCP, terse skill invoked.","session_id":"dry-deep-session-001","total_cost_usd":0.0923,"num_turns":11,"duration_ms":46600,"duration_api_ms":42100,"stop_reason":"end_turn","modelUsage":{"claude-sonnet-4-6":{"inputTokens":31650,"outputTokens":930,"cacheReadInputTokens":25330,"cacheCreationInputTokens":1200,"costUSD":0.0923,"webSearchRequests":1},"claude-haiku-4-5-20251001":{"inputTokens":120,"outputTokens":45,"cacheReadInputTokens":0,"cacheCreationInputTokens":0,"costUSD":0.00048,"webSearchRequests":0}}}
"""

# ── File access classification ────────────────────────────────────────────────
_WRITE_TOOLS = {"Write", "Edit", "NotebookEdit", "MultiEdit"}
_READ_TOOLS   = {"Read"}
_WEB_TOOLS    = {"WebFetch", "WebSearch"}
_SKILL_TOOLS  = {"Skill"}   # built-in skill dispatcher


@dataclass
class FileAccess:
    path: str
    op: str          # "read" | "write" | "exec" | "web" | "skill" | "mcp"
    tool: str
    turn_index: int
    tool_call_id: str


def _classify_file_accesses(tool_calls: list[ToolCall]) -> list[FileAccess]:
    accesses: list[FileAccess] = []
    for tc in tool_calls:
        if tc.name in _WRITE_TOOLS:
            op = "write"
        elif tc.name in _READ_TOOLS:
            op = "read"
        elif tc.name in _WEB_TOOLS:
            op = "web"
        elif tc.name in _SKILL_TOOLS:
            op = "skill"
        elif tc.is_mcp:
            op = "mcp"
        else:
            op = "exec"
        for path in tc.file_paths:
            accesses.append(FileAccess(
                path=path, op=op, tool=tc.name,
                turn_index=tc.turn_index, tool_call_id=tc.tool_use_id,
            ))
    return accesses


# ── Subprocess runner with timestamped line capture ───────────────────────────

def _run_claude(task: str, dry_run: bool, logger: EvalRunLogger) -> tuple[ClaudeStreamResult, list[tuple[float, str]], int]:
    """Returns (result, timestamped_lines, t0_ns) where t0_ns is the real wall-clock
    nanosecond timestamp of the first captured event — used to anchor OTel child spans."""
    import time as _time
    if dry_run:
        t0_ns = _time.time_ns()
        lines = [(i * 0.4, line) for i, line in enumerate(_DRY_STREAM.strip().split("\n"))]
        result = parse_timestamped_stream(lines)
        return result, lines, t0_ns

    env = os.environ.copy()
    env["TERM"] = "dumb"
    if tp := logger.get_traceparent_for_subprocess():
        env["TRACEPARENT"] = tp
    if ep := logger._config.otel_endpoint:
        env["OTEL_EXPORTER_OTLP_ENDPOINT"] = ep
        env["OTEL_SERVICE_NAME"] = "aet"
    cfg = logger._config
    resource_attrs = (
        f"aet.run_id={cfg.run_id},"
        f"aet.suite={cfg.suite},"
        f"aet.target={cfg.target},"
        f"aet.method={cfg.method},"
        f"aet.seed={cfg.seed}"
    )
    existing = env.get("OTEL_RESOURCE_ATTRIBUTES", "")
    env["OTEL_RESOURCE_ATTRIBUTES"] = f"{existing},{resource_attrs}" if existing else resource_attrs
    env.setdefault("CLAUDE_CODE_OTEL_SHUTDOWN_TIMEOUT_MS", "5000")
    env.setdefault("CLAUDE_CODE_OTEL_FLUSH_TIMEOUT_MS", "4000")

    # Write a per-run MCP config pointing at the bundled aet-eval MCP server.
    # The server exposes list_eval_runs / get_run_metrics so the agent can query
    # prior runs during the task — exercising the full mcp__ tracking path.
    _mcp_server_py = Path(__file__).parent / "aet_eval_mcp_server.py"
    _mcp_config = {
        "mcpServers": {
            "aet-eval": {
                "command": sys.executable,
                "args": [str(_mcp_server_py)],
            }
        }
    }
    import tempfile
    _mcp_cfg_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="aet_mcp_cfg_"
    )
    json.dump(_mcp_config, _mcp_cfg_file)
    _mcp_cfg_file.flush()
    _mcp_cfg_path = _mcp_cfg_file.name
    _mcp_cfg_file.close()

    cmd = ["claude", "--print", "--verbose", "--output-format", "stream-json",
           "--mcp-config", _mcp_cfg_path]
    if model := getattr(logger, "_model_override", None):
        cmd += ["--model", model]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, text=True,
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(task)
    proc.stdin.close()

    timestamped_lines: list[tuple[float, str]] = []
    t0 = time.monotonic()
    t0_ns = _time.time_ns()
    for raw_line in proc.stdout:
        timestamped_lines.append((time.monotonic() - t0, raw_line))

    proc.wait()
    try:
        Path(_mcp_cfg_path).unlink(missing_ok=True)
    except Exception:
        pass
    result = parse_timestamped_stream(timestamped_lines)
    return result, timestamped_lines, t0_ns


# ── aet recording — squeeze every feature ────────────────────────────────────

def _record_all(r: ClaudeStreamResult, logger: EvalRunLogger) -> list[FileAccess]:
    """Record the full result into aet — all metrics, events, and OTel spans."""
    file_accesses = _classify_file_accesses(r.tool_calls)

    # ── Session + model ───────────────────────────────────────────────
    if r.session_id:
        logger.log_session_id(r.session_id)
    logger.log_param("gen_ai.request.model", r.model)
    logger.log_param("gen_ai.response.model", r.model)
    logger.log_param("agent.num_turns", r.num_turns)
    logger.log_param("agent.session_id", r.session_id)

    # ── Aggregate tokens + cost ───────────────────────────────────────
    logger.log_token_usage(
        input_tokens=r.total_input_tokens,
        output_tokens=r.total_output_tokens,
        cache_creation_tokens=r.total_cache_creation_tokens,
        cache_read_tokens=r.total_cache_read_tokens,
        model=r.model,
        provider="anthropic",
    )
    logger.log_cost(r.cost_usd, model=r.model)
    logger.log_agent_turns(r.num_turns)

    # ── Timing ───────────────────────────────────────────────────────
    wall_s = r.duration_ms / 1000
    api_s  = r.duration_api_ms / 1000
    tool_s = wall_s - api_s
    logger.log_metric("gen_ai.client.operation.duration", wall_s)
    logger.log_metric("aet.agent.duration_ms", r.duration_ms)
    logger.log_metric("aet.agent.duration_api_ms", r.duration_api_ms)
    logger.log_metric("aet.agent.tool_wall_time_s", round(tool_s, 2))

    # ── Throughput ────────────────────────────────────────────────────
    if api_s > 0:
        tps = round(r.total_output_tokens / api_s, 1)
        logger.log_metric("gen_ai.client.tokens_per_second", tps)

    # ── Finish reasons ────────────────────────────────────────────────
    all_reasons: list[str] = []
    for tu in r.turn_usage:
        all_reasons.extend(tu.finish_reasons)
    if all_reasons:
        logger.log_finish_reasons(list(dict.fromkeys(all_reasons)))

    # ── Per-model cost + token breakdown (from modelUsage result block) ──
    # modelUsage gives the authoritative per-model totals from Claude Code:
    #   input_tokens        = raw (non-cached) input tokens billed
    #   output_tokens       = output tokens generated (billed at output rate)
    #   cache_read_tokens   = tokens read from cache (billed at cache-read rate, ~10× cheaper)
    #   cache_creation_tokens = tokens written to cache (billed at cache-write rate, ~25% premium)
    for mu in r.model_usage:
        safe = mu.model.replace("-", "_").replace(".", "_")
        logger.log_metric(f"per_model.{safe}.cost_usd",              mu.cost_usd)
        logger.log_metric(f"per_model.{safe}.input_tokens_raw",      mu.input_tokens)
        logger.log_metric(f"per_model.{safe}.output_tokens",         mu.output_tokens)
        logger.log_metric(f"per_model.{safe}.cache_read_tokens",     mu.cache_read_input_tokens)
        logger.log_metric(f"per_model.{safe}.cache_creation_tokens", mu.cache_creation_input_tokens)
        total_billed = (mu.input_tokens + mu.cache_read_input_tokens
                        + mu.cache_creation_input_tokens + mu.output_tokens)
        logger.log_metric(f"per_model.{safe}.total_billed_tokens",   total_billed)
        # cost breakdown by billing category
        if total_billed > 0:
            logger.log_metric(f"per_model.{safe}.pct_cache_read",
                              round(mu.cache_read_input_tokens / total_billed * 100, 1))
            logger.log_metric(f"per_model.{safe}.pct_cache_creation",
                              round(mu.cache_creation_input_tokens / total_billed * 100, 1))
            logger.log_metric(f"per_model.{safe}.pct_output",
                              round(mu.output_tokens / total_billed * 100, 1))
        # Cold-cost estimate: what this run would cost with NO cache hits
        # (all cache_read tokens become raw input, priced at full input rate)
        # Pricing anchors ($/MTok): Opus in=$15 out=$75; Sonnet in=$3 out=$15; Haiku in=$0.80 out=$4
        _input_rates  = {"opus": 15.0, "sonnet": 3.0,  "haiku": 0.80}
        _output_rates = {"opus": 75.0, "sonnet": 15.0, "haiku": 4.0}
        m_lower = mu.model.lower()
        tier = next((k for k in ("opus", "sonnet", "haiku") if k in m_lower), "sonnet")
        cold_input_tokens = mu.input_tokens + mu.cache_read_input_tokens + mu.cache_creation_input_tokens
        cold_cost = (cold_input_tokens * _input_rates[tier]
                     + mu.output_tokens * _output_rates[tier]) / 1_000_000
        cache_savings = max(0.0, cold_cost - mu.cost_usd)
        logger.log_metric(f"per_model.{safe}.cold_cost_usd",    round(cold_cost, 4))
        logger.log_metric(f"per_model.{safe}.cache_savings_usd", round(cache_savings, 4))
        logger.log_event("gen_ai.model.usage", {
            "model": mu.model,
            "input_tokens_raw": mu.input_tokens,
            "output_tokens": mu.output_tokens,
            "cache_read_tokens": mu.cache_read_input_tokens,
            "cache_creation_tokens": mu.cache_creation_input_tokens,
            "total_billed_tokens": total_billed,
            "cost_usd": mu.cost_usd,
            "cold_cost_usd_estimate": round(cold_cost, 4),
            "cache_savings_usd": round(cache_savings, 4),
            "web_search_requests": mu.web_search_requests,
        })

    # ── Per-tool-call token attribution ──────────────────────────────
    # For each tool call we know which turn it belonged to; we can attribute
    # tokens from that turn to the tool. If multiple tools fired in the same
    # turn we split evenly — this is approximate but useful for cost profiling.
    turn_tool_map: dict[int, list[ToolCall]] = defaultdict(list)
    for tc in r.tool_calls:
        turn_tool_map[tc.turn_index].append(tc)

    for tu in r.turn_usage:
        tools_this_turn = turn_tool_map.get(tu.turn, [])
        n = len(tools_this_turn) or 1
        turn_total = tu.total_input_tokens + tu.output_tokens
        for tc in tools_this_turn:
            logger.log_event("gen_ai.token.attribution", {
                "tool": tc.name,
                "tool_use_id": tc.tool_use_id,
                "turn": tu.turn,
                "model": tu.model or r.model,
                # turn-level tokens split across tools in that turn
                "attributed_input_tokens":            tu.input_tokens       // n,
                "attributed_output_tokens":           tu.output_tokens      // n,
                "attributed_cache_read_tokens":       tu.cache_read_input_tokens  // n,
                "attributed_cache_creation_tokens":   tu.cache_creation_input_tokens // n,
                "attributed_total_tokens":            turn_total            // n,
                "tool_duration_s":                    tc.duration_s,
                "file_paths":                         tc.file_paths,
            })

    # ── Per-turn step metrics (convergence curves in MLflow) ──────────
    cumulative_cost = 0.0
    for tu in r.turn_usage:
        s = tu.turn
        logger.log_metric_step("turn.input_tokens_raw",       tu.input_tokens,                  s)
        logger.log_metric_step("turn.output_tokens",          tu.output_tokens,                 s)
        logger.log_metric_step("turn.cache_creation_tokens",  tu.cache_creation_input_tokens,   s)
        logger.log_metric_step("turn.cache_read_tokens",      tu.cache_read_input_tokens,       s)
        logger.log_metric_step("turn.total_tokens",           tu.total_input_tokens + tu.output_tokens, s)
        tools_in_turn = len(turn_tool_map.get(tu.turn, []))
        logger.log_metric_step("turn.tool_calls", tools_in_turn, s)
        # per-turn tool names
        tool_names_in_turn = [tc.name for tc in turn_tool_map.get(tu.turn, [])]
        if tool_names_in_turn:
            logger.log_event("gen_ai.turn.tools", {
                "turn": tu.turn, "model": tu.model or r.model,
                "tools": tool_names_in_turn,
                "input_tokens_raw": tu.input_tokens,
                "output_tokens": tu.output_tokens,
                "cache_read_tokens": tu.cache_read_input_tokens,
                "cache_creation_tokens": tu.cache_creation_input_tokens,
            })
        else:
            # Planning/reasoning turn — no tool called, log the reasoning text
            logger.log_event("gen_ai.turn.planning", {
                "turn": tu.turn, "model": tu.model or r.model,
                "input_tokens_raw": tu.input_tokens,
                "output_tokens": tu.output_tokens,
                "cache_read_tokens": tu.cache_read_input_tokens,
                "cache_creation_tokens": tu.cache_creation_input_tokens,
                "reasoning_text": tu.reasoning_text,  # what the model was thinking
            })
        # Per-turn billing cost: use total_input (raw+cr+cc) + output
        total_in = tu.total_input_tokens
        total_out = tu.output_tokens
        logger.log_metric_step("turn.total_billed_tokens", total_in + total_out, s)

    # ── Per-tool-call events ──────────────────────────────────────────
    tool_type_counts: Counter = Counter()
    mcp_tool_counts: Counter = Counter()
    tool_duration_by_type: dict[str, list[float]] = defaultdict(list)

    for tc in r.tool_calls:
        tool_type_counts[tc.name] += 1
        if tc.is_mcp:
            mcp_tool_counts[tc.name] += 1

        # Rich tool call event — includes reasoning (WHY), file paths, and timing
        logger.log_event("gen_ai.tool.call.detail", {
            "tool": tc.name,
            "tool_use_id": tc.tool_use_id,
            "turn_index": tc.turn_index,
            "is_mcp": tc.is_mcp,
            "duration_s": tc.duration_s,
            "output_size": tc.output_size,
            "file_paths": tc.file_paths,
            "reasoning_before": tc.reasoning_before,   # WHY the model called this tool
            "input": tc.input_summary(500),
            "result": tc.result_summary(500),
            "error": tc.is_error,
        })
        logger.log_tool_call_event(
            tool_name=tc.name,
            input_summary=tc.input_summary(500),
            result_summary=tc.result_summary(500),
            is_error=tc.is_error,
            tool_call_id=tc.tool_use_id,
        )
        if tc.duration_s > 0:
            tool_duration_by_type[tc.name].append(tc.duration_s)

    # ── Tool aggregate metrics ────────────────────────────────────────
    logger.log_metric("aet.agent.tool_call_count",  r.tool_call_count)
    logger.log_metric("aet.agent.tool_error_count", r.tool_error_count)
    logger.log_metric("aet.agent.mcp_call_count",   sum(mcp_tool_counts.values()))
    for tool, count in tool_type_counts.items():
        safe = tool.replace(":", "_").replace(".", "_")
        logger.log_metric(f"tool.calls.{safe}", count)
        if tool_duration_by_type[tool]:
            avg_dur = sum(tool_duration_by_type[tool]) / len(tool_duration_by_type[tool])
            logger.log_metric(f"tool.avg_duration_s.{safe}", round(avg_dur, 3))

    # ── Cache efficiency ─────────────────────────────────────────────
    total_input_all = r.total_input_tokens  # raw + cache_read + cache_creation
    if total_input_all > 0:
        cache_hit_rate = r.total_cache_read_tokens / total_input_all
        logger.log_metric("aet.cache.hit_rate",           round(cache_hit_rate, 4))
        logger.log_metric("aet.cache.read_tokens",        r.total_cache_read_tokens)
        logger.log_metric("aet.cache.creation_tokens",    r.total_cache_creation_tokens)
        logger.log_metric("aet.cache.raw_input_tokens",   r.total_input_tokens - r.total_cache_read_tokens - r.total_cache_creation_tokens)
    for tu in r.turn_usage:
        s = tu.turn
        turn_total_in = tu.total_input_tokens
        if turn_total_in > 0:
            logger.log_metric_step("turn.cache_hit_rate",
                                   round(tu.cache_read_input_tokens / turn_total_in, 4), s)

    # ── Per-turn estimated cost (output-token fraction of per-model total) ──
    # Exact cost is only available at model level; distribute proportionally by
    # output tokens within each model so MLflow shows a cost convergence curve.
    model_output_totals: dict[str, int] = {mu.model: mu.output_tokens for mu in r.model_usage}
    model_cost_totals:   dict[str, float] = {mu.model: mu.cost_usd      for mu in r.model_usage}
    cumulative_cost = 0.0
    for tu in r.turn_usage:
        s = tu.turn
        m = tu.model or r.model
        denom = model_output_totals.get(m, 0) or 1
        turn_cost = model_cost_totals.get(m, 0.0) * (tu.output_tokens / denom)
        cumulative_cost += turn_cost
        logger.log_metric_step("turn.est_cost_usd",        round(turn_cost,       6), s)
        logger.log_metric_step("turn.cumulative_cost_usd", round(cumulative_cost, 6), s)
        logger.log_metric_step("turn.model",               hash(m) & 0xFFFF,         s)  # numeric proxy for model identity

    # ── Wall-clock breakdown: % time in inference vs tools ───────────────
    wall_s = r.duration_ms / 1000
    api_s  = r.duration_api_ms / 1000
    tool_s = max(wall_s - api_s, 0.0)
    if wall_s > 0:
        logger.log_metric("aet.agent.pct_time_inference", round(api_s  / wall_s * 100, 1))
        logger.log_metric("aet.agent.pct_time_tools",     round(tool_s / wall_s * 100, 1))

    # ── Redundancy detection ─────────────────────────────────────────────
    # Flags wasted work: same file read twice or same bash command run twice.
    seen_reads: Counter = Counter()
    seen_cmds:  Counter = Counter()
    for tc in r.tool_calls:
        if tc.name in ("Read", "Write", "Edit"):
            fp = tc.input.get("file_path", "")
            if fp and tc.name == "Read":
                seen_reads[fp] += 1
        elif tc.name == "Bash":
            cmd = (tc.input.get("command") or "")[:120]
            seen_cmds[cmd] += 1
    redundant_reads = {p: c for p, c in seen_reads.items() if c > 1}
    redundant_cmds  = {c: n for c, n in seen_cmds.items()  if n > 1}
    logger.log_metric("aet.redundancy.repeated_reads",    len(redundant_reads))
    logger.log_metric("aet.redundancy.repeated_commands", len(redundant_cmds))
    if redundant_reads or redundant_cmds:
        logger.log_event("aet.redundancy.detail", {
            "repeated_reads":    redundant_reads,
            "repeated_commands": redundant_cmds,
            "total_wasted_calls": sum(c - 1 for c in seen_reads.values())
                                 + sum(n - 1 for n in seen_cmds.values()),
        })

    # ── File + web + skill + MCP access events ───────────────────────
    read_paths  = [fa.path for fa in file_accesses if fa.op == "read"]
    write_paths = [fa.path for fa in file_accesses if fa.op == "write"]
    exec_paths  = [fa.path for fa in file_accesses if fa.op == "exec"]
    web_calls   = [tc for tc in r.tool_calls if tc.name in _WEB_TOOLS]
    skill_calls = [tc for tc in r.tool_calls if tc.name in _SKILL_TOOLS]
    mcp_calls   = [tc for tc in r.tool_calls if tc.is_mcp]
    logger.log_file_context(
        input_files=list(dict.fromkeys(read_paths + exec_paths)),
        output_files=list(dict.fromkeys(write_paths)),
    )
    logger.log_metric("files.read_count",   len(dict.fromkeys(read_paths)))
    logger.log_metric("files.write_count",  len(dict.fromkeys(write_paths)))
    logger.log_metric("web.fetch_count",    len(web_calls))
    logger.log_metric("skill.call_count",   len(skill_calls))
    logger.log_metric("mcp.call_count",     len(mcp_calls))
    # per-web-call event with URL and full result excerpt
    for tc in web_calls:
        url = tc.input.get("url") or tc.input.get("query") or ""
        logger.log_event("gen_ai.web.call", {
            "tool": tc.name, "url": url,
            "turn_index": tc.turn_index, "duration_s": tc.duration_s,
            "tool_use_id": tc.tool_use_id,
            "reasoning_before": tc.reasoning_before,
            "result_preview": tc.result_summary(600),  # enough to understand what was fetched
        })
    # per-skill-call event
    for tc in skill_calls:
        logger.log_event("gen_ai.skill.call", {
            "skill": tc.input.get("skill") or tc.input.get("name") or "",
            "args": str(tc.input.get("args", ""))[:200],
            "turn_index": tc.turn_index, "duration_s": tc.duration_s,
            "tool_use_id": tc.tool_use_id,
            "result_preview": tc.result_summary(120),
        })
    # per-MCP-call event (name includes mcp__<server>__<tool>)
    for tc in mcp_calls:
        parts = tc.name.split("__", 2)
        server = parts[1] if len(parts) > 1 else "unknown"
        tool   = parts[2] if len(parts) > 2 else tc.name
        logger.log_event("gen_ai.mcp.call", {
            "mcp_server": server, "mcp_tool": tool,
            "full_name": tc.name,
            "turn_index": tc.turn_index, "duration_s": tc.duration_s,
            "tool_use_id": tc.tool_use_id,
            "input": tc.input_summary(200),
            "result_preview": tc.result_summary(120),
        })
    for fa in file_accesses:
        logger.log_event("gen_ai.file.access", {
            "path": fa.path, "op": fa.op, "tool": fa.tool,
            "turn_index": fa.turn_index, "tool_call_id": fa.tool_call_id,
        })

    # ── Inference details event (spec) ───────────────────────────────
    logger.log_inference_details(
        operation="invoke_agent", provider="anthropic", model=r.model,
        conversation_id=r.session_id,
        input_tokens=r.total_input_tokens, output_tokens=r.total_output_tokens,
        cache_read_tokens=r.total_cache_read_tokens,
        cache_creation_tokens=r.total_cache_creation_tokens,
        finish_reasons=list(dict.fromkeys(all_reasons)),
        response_id=r.turn_usage[-1].message_id if r.turn_usage else "",
    )

    # ── Trace ID → param for aet show ────────────────────────────────
    if trace_id := logger.otel_trace_id:
        logger.log_param("aet.otel_trace_id", trace_id)

    # ── MLflow Traces tab — manual span tree for CLI subprocess runs ──
    # mlflow.anthropic.autolog() covers SDK calls; this covers our stream-json captures.
    logger.log_agent_trace(
        model=r.model,
        input_tokens=r.total_input_tokens,
        output_tokens=r.total_output_tokens,
        cost_usd=r.cost_usd,
        num_turns=r.num_turns,
        duration_ms=r.duration_ms,
        tool_calls=[
            {
                "name": tc.name,
                "input": tc.input,
                "result": tc.result or "",
                "duration_s": tc.duration_s,
                "is_error": tc.is_error,
                "is_mcp": tc.is_mcp,
            }
            for tc in r.tool_calls
        ],
    )

    # ── Context window utilization ────────────────────────────────────
    _CONTEXT_LIMITS = {"claude": 200_000}
    for tu in r.turn_usage:
        model_name = tu.model or r.model
        limit = next(
            (v for k, v in _CONTEXT_LIMITS.items() if model_name.startswith(k)),
            200_000,
        )
        pct = round(tu.total_input_tokens / limit * 100, 2)
        logger.log_metric_step("turn.context_pct_used", pct, tu.turn)
    max_input = max((tu.total_input_tokens for tu in r.turn_usage), default=0)
    max_pct = round(max_input / 200_000 * 100, 2)
    logger.log_metric("aet.context.max_pct_used", max_pct)
    if max_pct > 80:
        logger.log_event("aet.context.high_utilization_warning", {
            "max_pct": max_pct, "threshold": 80,
        })

    # ── Evaluation ───────────────────────────────────────────────────
    score = 1.0 if r.success else 0.0
    logger.log_task_achievement(achieved=r.success, score=score,
                                rationale="Agent completed all steps successfully" if r.success else "Agent reported failure")
    logger.log_evaluation_result("task_completion", score, "pass" if r.success else "fail",
                                 explanation=f"{r.num_turns} turns, {r.tool_call_count} tool calls, ${r.cost_usd:.4f}")
    # Per-step completeness: did agent produce the report file?
    report_written = any(fa.path == "/tmp/aet-deep-analysis.md" and fa.op == "write"
                         for fa in file_accesses)
    logger.log_evaluation_result("report_written", 1.0 if report_written else 0.0,
                                 "pass" if report_written else "fail",
                                 explanation="Agent wrote /tmp/aet-deep-analysis.md")
    version_added = any(
        tc.name == "Edit" and "__init__.py" in "".join(tc.file_paths)
        for tc in r.tool_calls
    )
    logger.log_evaluation_result("version_added", 1.0 if version_added else 0.0,
                                 "pass" if version_added else "fail",
                                 explanation="Agent added __version__ to __init__.py")

    # ── Structured rubric scoring ─────────────────────────────────────
    from aet.core.rubric import RubricCriterion, compute_weighted_score

    _RUBRIC_CRITERIA = [
        RubricCriterion("correctness",    0.30, "All steps completed correctly"),
        RubricCriterion("completeness",   0.25, "All 9 task steps addressed"),
        RubricCriterion("efficiency",     0.20, "Tool use and token efficiency"),
        RubricCriterion("tool_use",       0.15, "Appropriate tool selection"),
        RubricCriterion("report_quality", 0.10, "Quality of the written report"),
    ]
    efficiency_score = max(0.0, 1.0 - max(0, r.tool_call_count - 15) * 0.05
                                     - max(0, r.num_turns - 50) * 0.02)
    tool_use_score = 1.0 - (r.tool_error_count / max(r.tool_call_count, 1))
    rubric_scores = {
        "correctness":    1.0 if r.success else 0.0,
        "completeness":   1.0 if report_written else 0.0,
        "efficiency":     round(efficiency_score, 4),
        "tool_use":       round(tool_use_score, 4),
        "report_quality": 1.0 if report_written else 0.0,
    }
    for crit in _RUBRIC_CRITERIA:
        logger.log_rubric_score(
            criterion=crit.name,
            score=rubric_scores[crit.name],
            weight=crit.weight,
            explanation=crit.description,
        )
    composite = compute_weighted_score(rubric_scores, _RUBRIC_CRITERIA)
    logger.log_metric("rubric.composite_score", composite)
    logger.log_event("aet.rubric.summary", {
        "composite_score": composite,
        "scores": rubric_scores,
        "criteria": [{"name": c.name, "weight": c.weight} for c in _RUBRIC_CRITERIA],
    })

    # ── Trajectory sequences ──────────────────────────────────────────
    tool_sequence = [tc.name for tc in r.tool_calls]
    tool_sequence_unique = sorted(set(tool_sequence))
    logger.log_event("aet.trajectory.tool_sequence", {
        "tool_sequence": tool_sequence,
        "tool_sequence_unique": tool_sequence_unique,
        "length": len(tool_sequence),
        "unique_count": len(tool_sequence_unique),
    })

    # finish() is NOT called here — caller (run_deep_trace) does it after artifact upload
    return file_accesses


# ── ASCII terminal report ──────────────────────────────────────────────────────

def _box(title: str, rows: list[str], width: int = 90) -> str:
    lines = [f"┌─ {title} {'─' * (width - len(title) - 4)}┐"]
    for row in rows:
        pad = max(0, width - 2 - len(row))
        lines.append(f"│ {row}{' ' * pad} │")
    lines.append("└" + "─" * (width - 0) + "┘")
    return "\n".join(lines)


def _bar(value: float, maximum: float, width: int = 30, char: str = "█") -> str:
    if maximum <= 0:
        return " " * width
    filled = int(round(value / maximum * width))
    return char * filled + "░" * (width - filled)


def _print_terminal_report(
    r: ClaudeStreamResult,
    file_accesses: list[FileAccess],
    run_id: str,
    trace_id: str | None,
    mlflow_url: str | None,
) -> None:
    print()
    print("=" * 92)
    print(f"  aet deep-trace  ·  run {run_id}  ·  model {r.model}")
    print("=" * 92)

    # ── Summary row ──────────────────────────────────────────────────
    wall_s = r.duration_ms / 1000
    api_s  = r.duration_api_ms / 1000
    tps    = round(r.total_output_tokens / api_s, 1) if api_s > 0 else 0
    rows = [
        f"{'turns':<18} {r.num_turns:<6}  {'cost_usd':<18} ${r.cost_usd:.4f}",
        f"{'wall time':<18} {wall_s:.1f}s  {'api time':<18} {api_s:.1f}s",
        f"{'input tokens':<18} {r.total_input_tokens:<6}  {'output tokens':<18} {r.total_output_tokens}",
        f"{'cache_read':<18} {r.total_cache_read_tokens:<6}  {'cache_creation':<18} {r.total_cache_creation_tokens}",
        f"{'tool calls':<18} {r.tool_call_count:<6}  {'errors':<18} {r.tool_error_count}",
        f"{'throughput':<18} {tps} tok/s  {'status':<18} {'✓ pass' if r.success else '✗ fail'}",
    ]
    print(_box("RUN SUMMARY", rows))
    print()

    # ── Per-model cost + token decomposition ─────────────────────────
    if r.model_usage:
        billing_rates = {
            "input":            "$3/MTok",
            "output":           "$15/MTok",
            "cache_read":       "$0.30/MTok",
            "cache_creation":   "$3.75/MTok",
        }
        # detect opus vs sonnet vs haiku by model name
        def _billing_label(model: str) -> str:
            m = model.lower()
            if "opus" in m:    return "Opus    in=$15 out=$75 cr=$1.50 cc=$18.75"
            if "sonnet" in m:  return "Sonnet  in=$3  out=$15 cr=$0.30 cc=$3.75"
            if "haiku" in m:   return "Haiku   in=$0.80 out=$4 cr=$0.08 cc=$1"
            return model

        _input_rates  = {"opus": 15.0, "sonnet": 3.0,  "haiku": 0.80}
        _output_rates = {"opus": 75.0, "sonnet": 15.0, "haiku": 4.0}
        model_rows = [
            f"{'model':<42} {'raw_in':>8} {'output':>8} {'cache_r':>8} {'cache_c':>8}  {'actual':>9}  {'cold_est':>9}"
        ]
        total_cold = 0.0
        for mu in r.model_usage:
            total_billed = mu.input_tokens + mu.output_tokens + mu.cache_read_input_tokens + mu.cache_creation_input_tokens
            pct_cr  = mu.cache_read_input_tokens  / max(total_billed, 1) * 100
            pct_cc  = mu.cache_creation_input_tokens / max(total_billed, 1) * 100
            pct_out = mu.output_tokens            / max(total_billed, 1) * 100
            pct_in  = mu.input_tokens             / max(total_billed, 1) * 100
            m_lower = mu.model.lower()
            tier    = next((k for k in ("opus", "sonnet", "haiku") if k in m_lower), "sonnet")
            cold_in = mu.input_tokens + mu.cache_read_input_tokens + mu.cache_creation_input_tokens
            cold    = (cold_in * _input_rates[tier] + mu.output_tokens * _output_rates[tier]) / 1_000_000
            total_cold += cold
            savings_pct = (cold - mu.cost_usd) / max(cold, 1e-9) * 100
            model_rows.append(
                f"  {mu.model:<40} {mu.input_tokens:>8,} {mu.output_tokens:>8,} "
                f"{mu.cache_read_input_tokens:>8,} {mu.cache_creation_input_tokens:>8,}  ${mu.cost_usd:>8.4f}  ${cold:>8.4f}"
            )
            model_rows.append(
                f"  {'  rates: ' + _billing_label(mu.model):<40} billing mix: "
                f"{pct_in:.0f}%in/{pct_out:.0f}%out/{pct_cr:.0f}%cr/{pct_cc:.0f}%cc  cache saved {savings_pct:.0f}%"
            )
        total_actual = sum(mu.cost_usd for mu in r.model_usage)
        model_rows.append(f"  {'TOTAL':>76}  ${total_actual:>8.4f}  ${total_cold:>8.4f}")
        print(_box("PER-MODEL TOKEN + COST DECOMPOSITION  (cold_est = if no cache)", model_rows))
        print()

    # ── Per-turn breakdown with tool attribution ──────────────────────
    turn_tool_map_print: dict[int, list[str]] = defaultdict(list)
    for tc in r.tool_calls:
        turn_tool_map_print[tc.turn_index].append(tc.name)

    max_total = max((t.total_input_tokens + t.output_tokens for t in r.turn_usage), default=1)
    turn_rows = [f"{'T':>3} {'in_raw':>7} {'out':>6} {'cache_r':>8} {'cache_c':>8}  bar                            tools"]
    for tu in r.turn_usage:
        total  = tu.total_input_tokens + tu.output_tokens
        bar    = _bar(total, max_total, width=20)
        tools  = ",".join(turn_tool_map_print.get(tu.turn, [])) or "—"
        model_tag = f"[{tu.model.split('-')[2] if tu.model else '?'}]" if tu.model else ""
        turn_rows.append(
            f"  {tu.turn:<2} {tu.input_tokens:>7,} {tu.output_tokens:>6,} "
            f"{tu.cache_read_input_tokens:>8,} {tu.cache_creation_input_tokens:>8,}  {bar}  {tools}"
        )
    print(_box("TOKEN BREAKDOWN PER TURN  (in_raw=non-cached input)", turn_rows))
    print()

    # ── Tool call summary with token context ─────────────────────────
    tool_counts: Counter = Counter(tc.name for tc in r.tool_calls)
    tool_durations: dict[str, list[float]] = defaultdict(list)
    # tokens attributed per tool (summed from their turns, split evenly)
    tool_token_attr: dict[str, int] = defaultdict(int)
    turn_map_for_attr: dict[int, Any] = {tu.turn: tu for tu in r.turn_usage}
    for tc in r.tool_calls:
        tu = turn_map_for_attr.get(tc.turn_index)
        if tu:
            n = len(turn_tool_map_print.get(tu.turn, [])) or 1
            tool_token_attr[tc.name] += (tu.total_input_tokens + tu.output_tokens) // n
        if tc.duration_s > 0:
            tool_durations[tc.name].append(tc.duration_s)

    max_count = max(tool_counts.values(), default=1)
    tool_rows = [f"{'tool':<26} {'type':<6} {'calls':>5}  {'avg_dur':>8}  {'~tokens':>8}  bar"]
    for tool, cnt in tool_counts.most_common():
        bar      = _bar(cnt, max_count, width=20)
        is_mcp   = tool.startswith("mcp__")
        is_skill = tool.startswith("mcp__claude_ai") or tool.startswith("Skill")
        ttype    = "MCP" if is_mcp else ("SKILL" if is_skill else "native")
        dur_s    = tool_durations.get(tool, [])
        avg_dur  = f"{sum(dur_s)/len(dur_s):.2f}s" if dur_s else "   n/a"
        attr_tok = tool_token_attr.get(tool, 0)
        tool_rows.append(
            f"  {tool:<24} {ttype:<6} {cnt:>5}  {avg_dur:>8}  {attr_tok:>8,}  {bar}"
        )
    print(_box("TOOL CALL BREAKDOWN  (type: native/MCP/SKILL, ~tokens=attributed)", tool_rows))
    print()

    # ── Planning turns (no tool called — pure reasoning/thinking) ─────
    plan_turns = [tu for tu in r.turn_usage
                  if not turn_tool_map_print.get(tu.turn)]
    if plan_turns:
        plan_rows = [f"{'T':>3} {'in_raw':>7} {'out':>6} {'cache_r':>8}  {'reasoning (first 120 chars)'}"]
        for tu in plan_turns:
            snippet = (tu.reasoning_text or "")[:120].replace("\n", " ")
            plan_rows.append(
                f"  {tu.turn:<2} {tu.input_tokens:>7,} {tu.output_tokens:>6,} "
                f"{tu.cache_read_input_tokens:>8,}  {snippet}"
            )
        print(_box(f"PLANNING TURNS  ({len(plan_turns)} turns with no tool call — reasoning only)", plan_rows))
        print()

    # ── File access map ───────────────────────────────────────────────
    if file_accesses:
        file_ops: dict[str, dict] = {}
        for fa in file_accesses:
            if fa.path not in file_ops:
                file_ops[fa.path] = {"R": 0, "W": 0, "X": 0}
            key = {"read": "R", "write": "W", "exec": "X"}.get(fa.op, "X")
            file_ops[fa.path][key] += 1
        file_rows = [f"{'path / url':<58} {'op':<6} {'tool':<14} turn"]
        for fa in sorted(file_accesses, key=lambda x: (x.turn_index, x.op)):
            short = ("…" + fa.path[-55:]) if len(fa.path) > 56 else fa.path
            op_label = {"read":"READ","write":"WRITE","exec":"EXEC",
                        "web":"WEB","skill":"SKILL","mcp":"MCP"}.get(fa.op, fa.op.upper())
            file_rows.append(f"  {short:<56} {op_label:<6} {fa.tool:<14} T{fa.turn_index}")
        print(_box("RESOURCE ACCESS MAP  (files · web · skills · MCP)", file_rows))
        print()

    # ── Tool I/O snapshots ────────────────────────────────────────────
    # Show input + result excerpts for every tool call so you can see
    # exactly what was retrieved / executed and why.
    io_rows: list[str] = []
    for i, tc in enumerate(r.tool_calls):
        mcp_tag  = " [MCP]"   if tc.is_mcp            else ""
        web_tag  = " [WEB]"   if tc.name in _WEB_TOOLS else ""
        sk_tag   = " [SKILL]" if tc.name in _SKILL_TOOLS else ""
        tag      = mcp_tag or web_tag or sk_tag
        header   = f"  [{i+1:02}] T{tc.turn_index}  {tc.name}{tag}  dur={tc.duration_s:.2f}s"
        io_rows.append(header)
        # why the model called this tool (reasoning text immediately before the call)
        if tc.reasoning_before:
            why = tc.reasoning_before[:200].replace("\n", " ").replace("\r", "")
            io_rows.append(f"       WHY: {why}")
        # input excerpt
        inp = tc.input_summary(280).replace("\n", " ").replace("\r", "")
        io_rows.append(f"       IN:  {inp}")
        # result excerpt (richer — 400 chars)
        res = (tc.result or "")[:400].replace("\n", "↵ ").replace("\r", "")
        if res:
            io_rows.append(f"       OUT: {res}")
        if tc.is_error:
            io_rows.append(f"       *** ERROR ***")
        io_rows.append("")
    if io_rows:
        print(_box("TOOL I/O SNAPSHOTS", io_rows))
        print()

    # ── Links ─────────────────────────────────────────────────────────
    link_rows = []
    if trace_id:
        link_rows.append(f"otel trace_id  {trace_id}")
    if mlflow_url:
        link_rows.append(f"mlflow         {mlflow_url}")
    if r.session_id:
        link_rows.append(f"claude resume  claude --resume {r.session_id}")
    if link_rows:
        print(_box("LINKS", link_rows))
        print()


# ── Matplotlib analytics figure ───────────────────────────────────────────────

def _generate_plots(
    r: ClaudeStreamResult,
    file_accesses: list[FileAccess],
    run_id: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("[aet] matplotlib not available — skipping plot generation")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"aet deep-trace analytics  ·  run {run_id[:16]}  ·  {r.model}",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("#1e1e2e")
    for ax in axes.flat:
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cba6f7")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

    turns = [tu.turn for tu in r.turn_usage]
    clrs = {"raw_input": "#89b4fa", "cache_read": "#a6e3a1",
            "cache_creation": "#f9e2af", "output": "#fab387"}

    # ── Panel 1: Token stacks per turn ───────────────────────────────
    ax1 = axes[0, 0]
    raw_in  = [tu.input_tokens               for tu in r.turn_usage]
    cr      = [tu.cache_read_input_tokens    for tu in r.turn_usage]
    cc      = [tu.cache_creation_input_tokens for tu in r.turn_usage]
    out     = [tu.output_tokens              for tu in r.turn_usage]
    x = np.arange(len(turns))
    w = 0.6
    b0 = ax1.bar(x, raw_in, w, label="raw input",        color=clrs["raw_input"],      alpha=0.9)
    b1 = ax1.bar(x, cr,     w, bottom=raw_in,            label="cache read",           color=clrs["cache_read"],    alpha=0.9)
    b2 = ax1.bar(x, cc,     w, bottom=[a+b for a,b in zip(raw_in, cr)],
                                                          label="cache creation",       color=clrs["cache_creation"],alpha=0.9)
    b3 = ax1.bar(x, out,    w, bottom=[a+b+c for a,b,c in zip(raw_in, cr, cc)],
                                                          label="output",               color=clrs["output"],         alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"T{t}" for t in turns])
    ax1.set_title("Tokens per turn")
    ax1.set_ylabel("tokens")
    ax1.legend(fontsize=8, labelcolor="#cdd6f4", facecolor="#313244", edgecolor="#45475a")

    # ── Panel 2: Cumulative cost over turns ──────────────────────────
    ax2 = axes[0, 1]
    if r.total_output_tokens > 0 and r.cost_usd > 0:
        cumcost = []
        acc = 0.0
        for tu in r.turn_usage:
            frac  = tu.output_tokens / max(r.total_output_tokens, 1)
            acc  += r.cost_usd * frac
            cumcost.append(acc)
        ax2.plot(turns, cumcost, "o-", color="#cba6f7", linewidth=2, markersize=6)
        ax2.fill_between(turns, cumcost, alpha=0.2, color="#cba6f7")
        ax2.set_xticks(turns)
        ax2.set_xticklabels([f"T{t}" for t in turns])
        # Mark wall-time on secondary x axis annotation
        ax2.set_title("Cumulative cost (USD)")
        ax2.set_ylabel("$USD")
        for i, (t, c) in enumerate(zip(turns, cumcost)):
            ax2.annotate(f"${c:.4f}", (t, c), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="#cdd6f4")
    else:
        ax2.text(0.5, 0.5, "no cost data", ha="center", va="center",
                color="#585b70", transform=ax2.transAxes)
        ax2.set_title("Cumulative cost (USD)")

    # ── Panel 3: Tool call distribution ──────────────────────────────
    ax3 = axes[1, 0]
    tool_counts: Counter = Counter(tc.name for tc in r.tool_calls)
    if tool_counts:
        tools_sorted = sorted(tool_counts.items(), key=lambda x: x[1])
        names  = [t[0] for t in tools_sorted]
        counts = [t[1] for t in tools_sorted]
        colors_tools = ["#f38ba8" if n.startswith("mcp__") else "#89dceb" for n in names]
        bars = ax3.barh(names, counts, color=colors_tools, alpha=0.85)
        ax3.set_title("Tool call distribution")
        ax3.set_xlabel("calls")
        for bar, cnt in zip(bars, counts):
            ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    str(cnt), va="center", fontsize=9, color="#cdd6f4")
        native_p = mpatches.Patch(color="#89dceb", label="native tool")
        mcp_p    = mpatches.Patch(color="#f38ba8", label="MCP tool")
        ax3.legend(handles=[native_p, mcp_p], fontsize=8,
                  labelcolor="#cdd6f4", facecolor="#313244", edgecolor="#45475a")
    else:
        ax3.text(0.5, 0.5, "no tool calls", ha="center", va="center",
                color="#585b70", transform=ax3.transAxes)
        ax3.set_title("Tool call distribution")

    # ── Panel 4: File access frequency ───────────────────────────────
    ax4 = axes[1, 1]
    if file_accesses:
        file_read_counts:  Counter = Counter()
        file_write_counts: Counter = Counter()
        for fa in file_accesses:
            short = Path(fa.path).name
            if fa.op == "write":
                file_write_counts[short] += 1
            else:
                file_read_counts[short] += 1
        all_files = list(dict.fromkeys(
            list(file_read_counts.keys()) + list(file_write_counts.keys())
        ))[-12:]  # top 12
        reads  = [file_read_counts.get(f, 0)  for f in all_files]
        writes = [file_write_counts.get(f, 0) for f in all_files]
        yp = np.arange(len(all_files))
        ax4.barh(yp - 0.2, reads,  0.4, label="read",  color="#a6e3a1", alpha=0.85)
        ax4.barh(yp + 0.2, writes, 0.4, label="write", color="#f38ba8", alpha=0.85)
        ax4.set_yticks(yp)
        ax4.set_yticklabels(all_files, fontsize=8)
        ax4.set_title("File access frequency")
        ax4.set_xlabel("accesses")
        ax4.legend(fontsize=8, labelcolor="#cdd6f4",
                  facecolor="#313244", edgecolor="#45475a")
    else:
        ax4.text(0.5, 0.5, "no file accesses captured", ha="center", va="center",
                color="#585b70", transform=ax4.transAxes)
        ax4.set_title("File access frequency")

    # ── Timing annotation strip ──────────────────────────────────────
    wall_s = r.duration_ms / 1000
    api_s  = r.duration_api_ms / 1000
    tps    = round(r.total_output_tokens / api_s, 1) if api_s > 0 else 0
    fig.text(0.01, 0.01,
             f"wall {wall_s:.1f}s  ·  api {api_s:.1f}s  ·  "
             f"{tps} tok/s  ·  ${r.cost_usd:.4f}  ·  "
             f"{r.total_input_tokens}in/{r.total_output_tokens}out tokens",
             ha="left", fontsize=9, color="#585b70")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[aet] plots saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_deep_trace(
    run_path: Path,
    tracking_mode: str,
    otel_endpoint: str | None,
    mlflow_uri: str | None,
    dry_run: bool,
    model: str = "claude-sonnet-4-6",
    task: str = TASK,
) -> dict:
    import uuid
    run_id = f"deep_trace_{uuid.uuid4().hex[:10]}"
    run_path.mkdir(parents=True, exist_ok=True)
    plots_path = run_path / "plots" / "analytics.png"

    logger = EvalRunLogger.start(
        project="aet-deep-trace",
        suite="agent-analysis",
        target="aet-codebase",
        method="claude-code-agent",
        seed=1,
        run_id=run_id,
        run_path=run_path,
        tracking_mode=tracking_mode,
        mlflow_tracking_uri=mlflow_uri,
        experiment_name="aet-deep-trace",
        otel_endpoint=otel_endpoint,
        enable_openllmetry=True,
    )
    # stash model override so _run_claude can pick it up
    logger._model_override = model  # type: ignore[attr-defined]

    logger.log_param("task_hash", hex(hash(task) & 0xFFFFFFFF))
    logger.log_param("task_preview", task[:120].replace("\n", " "))
    logger.log_param("model_override", model)
    logger.log_param("dry_run", dry_run)
    logger.log_permission_mode("default", approvals_required=0)

    with logger.start_run_span("deep_trace"):
        with logger.start_agent_span(
            "claude-code", model=model, provider="anthropic",
        ):
            r, _, t0_ns = _run_claude(task, dry_run, logger)
            file_accesses = _record_all(r, logger)
            # Emit child spans while the invoke_agent span is still active
            stream_dur = getattr(r, "_stream_duration_s", r.duration_ms / 1000)
            logger.emit_tool_call_spans(r.tool_calls, r.turn_usage, t0_ns, stream_dur)

    trace_id  = logger.otel_trace_id
    mlflow_url = logger.mlflow_run_url

    _print_terminal_report(r, file_accesses, run_id, trace_id, mlflow_url)
    _generate_plots(r, file_accesses, run_id, plots_path)

    if plots_path.exists():
        logger.log_artifact(plots_path, "plots")  # artifact_path is a directory in the store

    if (Path("/tmp/aet-deep-analysis.md")).exists():
        logger.log_artifact(Path("/tmp/aet-deep-analysis.md"), None)  # root of artifact store

    logger.finish(status="pass" if r.success else "fail",
                  message=r.result_text[:200] if r.result_text else "")
    logger.close()
    return {
        "run_id": run_id, "trace_id": trace_id, "mlflow_url": mlflow_url,
        "success": r.success, "cost_usd": r.cost_usd, "turns": r.num_turns,
        "tool_calls": r.tool_call_count, "files_written": len([f for f in file_accesses if f.op == "write"]),
        "plots": str(plots_path) if plots_path.exists() else None,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep-trace agentic eval — all aet features")
    p.add_argument("--run-path",  type=Path, default=Path("/tmp/aet-deep-trace"))
    p.add_argument("--otel-endpoint", default=None)
    p.add_argument("--mlflow-uri",    default=None)
    p.add_argument("--tracking", dest="tracking_mode", default="local",
                   choices=["local", "mlflow", "full", "debug"])
    p.add_argument("--model", default="claude-sonnet-4-6",
                   help="Model for the claude CLI subprocess (e.g. claude-opus-4-8)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    mode_note = "(dry-run)" if args.dry_run else ""
    print(f"\n[aet] deep-trace eval {mode_note}  model={args.model}  tracking={args.tracking_mode}\n")
    result = run_deep_trace(
        run_path=args.run_path,
        tracking_mode=args.tracking_mode,
        otel_endpoint=args.otel_endpoint,
        mlflow_uri=args.mlflow_uri,
        dry_run=args.dry_run,
        model=args.model,
    )
    print(json.dumps(result, indent=2))
