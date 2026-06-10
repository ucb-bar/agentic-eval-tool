# aet examples

End-to-end scripts that demonstrate the full aet pipeline.

## claude_code_eval.py

Calls Claude Code through the aet eval harness: init_run → invoke_agent → validate → tracking.
Produces traces visible in SigNoz and nested MLflow runs when those services are running.

### Scenario 1 — local only (no services, no API key)

```bash
python examples/claude_code_eval.py --dry-run
```

Writes artifacts to `/tmp/aet-claude-example/`, logs everything to local jsonl files.

### Scenario 2 — SigNoz traces

Start SigNoz once (requires Docker):

```bash
bash src/aet/templates/targetgen_project/observability/install-signoz.sh
# wait ~30s for ClickHouse + schema migration, then open http://localhost:8080
```

Run with tracing:

```bash
python examples/claude_code_eval.py \
    --otel-endpoint http://localhost:4318 \
    --tracking full \
    --dry-run
```

Open SigNoz at `http://localhost:8080` → **Services** → `aet` → find the trace.
Drill into: `invoke_workflow` → `invoke_agent(claude-code)` → `execute_tool(validate)`.

The `gen_ai.*` attributes and `gen_ai.evaluation.result` event appear in the span detail pane.
The **LLM Observability** page in SigNoz shows token usage and latency histograms.

### Scenario 3 — full stack (SigNoz + MLflow)

Start MLflow (one-liner if you have it installed):

```bash
docker compose -f src/aet/templates/targetgen_project/observability/docker-compose.mlflow.yml up -d
```

Run:

```bash
python examples/claude_code_eval.py \
    --project-root /tmp/aet-full \
    --otel-endpoint http://localhost:4318 \
    --mlflow-uri http://localhost:5001 \
    --tracking full \
    --dry-run
```

The script prints the MLflow run URL on completion. Open it to see params, metrics, and the
`claude_output.md` artifact.

### Scenario 4 — real Claude Code (API key required)

```bash
export ANTHROPIC_API_KEY=<your key>
python examples/claude_code_eval.py \
    --otel-endpoint http://localhost:4318 \
    --tracking full
```

This calls `claude --print --no-verbose <task>` as a subprocess.
The `TRACEPARENT` env var is injected so Claude Code spans appear nested under
`invoke_agent(claude-code)` in SigNoz when Claude Code OTel instrumentation is enabled.
See `src/aet/templates/targetgen_project/observability/claude_code_otel.md` for setup.
