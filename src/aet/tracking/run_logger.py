"""EvalRunLogger — single tracking API for the aet harness."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from aet.tracking import reports
from aet.tracking.types import TrackingConfig, TRACKING_MODES
from aet.tracking.local_backend import LocalBackend
from aet.tracking.mlflow_backend import MLflowBackend
from aet.tracking.otel_backend import OtelBackend


class EvalRunLogger:
    """
    One canonical tracking abstraction for all harness code.

    Backends:
      local  — always active (pure stdlib, writes to logs/)
      mlflow — active when mode != "local" (optional import)
      otel   — active when mode in ("full", "debug") (optional import)

    All backend failures are caught, warned locally, and never propagate.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self._config = config
        self._local = LocalBackend(config)
        self._mlflow: MLflowBackend | None = (
            MLflowBackend(config, self._local) if config.mode != "local" else None
        )
        self._otel: OtelBackend | None = (
            OtelBackend(config, self._local) if config.mode in ("full", "debug") else None
        )

    # ------------------------------------------------------------------
    @classmethod
    def start(
        cls,
        *,
        project: str = "",
        suite: str = "",
        target: str,
        method: str,
        seed: int,
        run_id: str,
        run_path: Path,
        tracking_mode: str = "local",
        mlflow_tracking_uri: str | None = None,
        experiment_name: str | None = None,
        otel_endpoint: str | None = None,
        parent_run_id: str | None = None,
        enable_openllmetry: bool = True,
    ) -> "EvalRunLogger":
        if tracking_mode not in TRACKING_MODES:
            tracking_mode = "local"
        config = TrackingConfig(
            mode=tracking_mode,
            run_id=run_id,
            project=project,
            suite=suite,
            target=target,
            method=method,
            seed=seed,
            run_path=Path(run_path),
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name,
            otel_endpoint=otel_endpoint,
            parent_run_id=parent_run_id,
            enable_openllmetry=enable_openllmetry,
        )
        return cls(config)

    # ------------------------------------------------------------------
    def log_param(self, name: str, value: Any) -> None:
        self._local.log_param(name, value)
        if self._mlflow:
            self._mlflow.log_param(name, value)

    def log_params(self, params: dict[str, Any]) -> None:
        self._local.log_params(params)
        if self._mlflow:
            self._mlflow.log_params(params)

    # ------------------------------------------------------------------
    def log_metric(
        self,
        name: str,
        value: int | float | bool | None,
        step: int | None = None,
        source: str | None = None,
    ) -> None:
        self._local.log_metric(name, value, step=step, source=source)
        if self._mlflow:
            self._mlflow.log_metric(name, value, step=step, source=source)

    def log_metrics(self, metrics: dict[str, Any], prefix: str | None = None) -> None:
        self._local.log_metrics(metrics, prefix=prefix)
        if self._mlflow:
            self._mlflow.log_metrics(metrics, prefix=prefix)

    def log_metric_step(self, name: str, value: float, step: int) -> None:
        """Log a metric at a specific step (for convergence curves)."""
        self._local.log_metric(name, value, step=step)
        if self._mlflow:
            self._mlflow.log_step_metric(name, value, step)

    # ------------------------------------------------------------------
    def log_event(self, name: str, payload: dict | None = None) -> None:
        self._local.log_event(name, payload)

    # ------------------------------------------------------------------
    # trajectory recording — thin wrappers over the primitives above so an agentic
    # RunTrajectory is reconstructable from canonical logs/ (no new backend/format).
    # ------------------------------------------------------------------
    def log_trajectory_point(self, index: int, t_s: float,
                             cum_input: float, cum_output: float, cum_cache: float,
                             cum_cost: float, round_index: int = 0,
                             provisional_cost: bool = False,
                             cum_cache_read: float = 0.0,
                             cum_cache_creation: float = 0.0) -> None:
        """One cumulative sample as a family of step-metrics keyed by point ``index``.

        ``cum_cache`` is the sum the cost model bills as one class; the read/creation split is
        logged alongside it because the two are priced an order of magnitude apart (a read is
        ~0.1x the input rate, a write ~1.25x), so a curve that only carries the sum cannot show
        where a run's cache spend went. Both split params default to 0.0: callers written against
        the older signature keep working, and a log recorded before this change reconstructs with
        the split fields at zero rather than failing to load.
        """
        self.log_metric_step("aet.traj.t_s", t_s, step=index)
        self.log_metric_step("aet.traj.cum_input_tokens", cum_input, step=index)
        self.log_metric_step("aet.traj.cum_output_tokens", cum_output, step=index)
        self.log_metric_step("aet.traj.cum_cache_tokens", cum_cache, step=index)
        self.log_metric_step("aet.traj.cum_cache_read_tokens", cum_cache_read, step=index)
        self.log_metric_step("aet.traj.cum_cache_creation_tokens", cum_cache_creation, step=index)
        self.log_metric_step("aet.traj.cum_total_tokens", cum_input + cum_output + cum_cache,
                             step=index)
        self.log_metric_step("aet.traj.cum_cost_usd", cum_cost, step=index)
        self.log_metric_step("aet.traj.round_index", round_index, step=index)
        self.log_metric_step("aet.traj.provisional_cost", 1.0 if provisional_cost else 0.0,
                             step=index)

    def log_test_milestone(self, t_s: float, n_passed: int, n_total: int,
                           round_index: int | None = None, scope: str = "all") -> None:
        """An external-oracle test-pass reading (metric for the curve + event for the time)."""
        self._local._log_event_rich(
            "aet.traj.milestone",
            {"t_s": t_s, "n_passed": n_passed, "n_total": n_total,
             "scope": scope, "round_index": round_index},
            stage="eval", actor="oracle")

    def log_round_boundary(self, rb) -> None:
        """One agent round's wall span + billed totals + QA verdict."""
        from dataclasses import asdict, is_dataclass
        payload = asdict(rb) if is_dataclass(rb) else dict(rb)
        self._local._log_event_rich("aet.traj.round", payload,
                                    stage="agent_action", actor="harness")

    # ------------------------------------------------------------------
    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self._local.log_artifact(path, artifact_path)
        if self._mlflow:
            self._mlflow.log_artifact(path, artifact_path)

    def log_artifacts(self, path: Path, artifact_path: str | None = None) -> None:
        self._local.log_artifacts(path, artifact_path)
        if self._mlflow:
            self._mlflow.log_artifacts(path, artifact_path)

    def log_generated_dir(self, generated_dir: Path, target: str) -> None:
        """Log generated/<target>-mlir/ as a tarball artifact (MLflow only)."""
        if self._mlflow:
            self._mlflow.log_generated_dir_as_tarball(generated_dir, target)

    # ------------------------------------------------------------------
    def start_span(self, name: str, attributes: dict[str, Any] | None = None):
        """Context manager. Returns real OTel span or nullcontext."""
        if self._otel:
            return self._otel.start_span(name, attributes)
        return nullcontext()

    def start_run_span(self, workflow_name: str):
        """Context manager wrapping the full eval run as invoke_workflow."""
        if self._otel:
            return self._otel.start_workflow_span(
                workflow_name,
                suite=self._config.suite,
                method=self._config.method,
                seed=self._config.seed,
                target=self._config.target,
                run_id=self._config.run_id,
            )
        return nullcontext()

    def start_agent_span(self, agent_name: str, model: str = "", provider: str = "anthropic"):
        """Context manager for an agent invocation — invoke_agent."""
        if self._otel:
            return self._otel.start_agent_span(agent_name, model=model, provider=provider)
        return nullcontext()

    def start_inference_span(self, model: str, provider: str = "anthropic",
                             server_address: str = "", operation: str = "chat",
                             stream: bool = False):
        """Context manager for a direct LLM API call (Anthropic SDK, OpenAI, Bedrock)."""
        if self._otel:
            return self._otel.start_inference_span(
                model=model, provider=provider, server_address=server_address,
                operation=operation, stream=stream,
            )
        return nullcontext()

    def start_tool_span(self, tool_name: str, validator_name: str | None = None):
        """Context manager for a tool/validator execution — execute_tool."""
        if self._otel:
            return self._otel.start_tool_span(tool_name, validator_name=validator_name)
        return nullcontext()

    def log_rubric_score(
        self,
        criterion: str,
        score: float,
        weight: float = 1.0,
        explanation: str = "",
    ) -> None:
        """Log a single rubric criterion score as metrics + event."""
        self._local.log_metric(f"rubric.{criterion}.score", score)
        self._local.log_metric(f"rubric.{criterion}.weight", weight)
        self._local.log_event("aet.rubric.criterion", {
            "criterion": criterion,
            "score": score,
            "weight": weight,
            "explanation": explanation,
        })
        if self._mlflow:
            self._mlflow.log_metric(f"rubric.{criterion}.score", score)
            self._mlflow.log_metric(f"rubric.{criterion}.weight", weight)

    def log_regression_check(
        self,
        metric: str,
        value: float,
        baseline_value: float,
        threshold_pct: float,
    ) -> None:
        """Emit an aet.regression.{metric} event with delta and regression flag."""
        if baseline_value != 0:
            delta_pct = (value - baseline_value) / abs(baseline_value) * 100
        else:
            delta_pct = 0.0
        is_regression = abs(delta_pct) > threshold_pct
        self._local.log_event(f"aet.regression.{metric}", {
            "metric": metric,
            "value": value,
            "baseline_value": baseline_value,
            "delta_pct": round(delta_pct, 4),
            "threshold_pct": threshold_pct,
            "is_regression": is_regression,
        })

    def log_evaluation_result(self, name: str, score: float, label: str,
                              explanation: str = "") -> None:
        """Emit a gen_ai.evaluation.result event on the current OTel span."""
        self._local.log_event("evaluation.result", {"name": name, "score": score,
                                                     "label": label, "explanation": explanation})
        if self._otel:
            self._otel.log_evaluation_event(name, score, label, explanation=explanation)

    # ------------------------------------------------------------------
    # Claude Code / agent-specific instrumentation

    def log_prompt(self, prompt: str, role: str = "user", max_chars: int = 4000) -> None:
        """Record the input prompt as a span event and local log entry."""
        truncated = prompt[:max_chars]
        self._local.log_event("gen_ai.prompt", {"role": role, "content": truncated,
                                                 "truncated": len(prompt) > max_chars})
        if self._otel:
            self._otel.log_prompt_event(truncated, role=role)

    def log_completion(self, text: str, max_chars: int = 8000) -> None:
        """Record the agent output as a span event and local log entry."""
        truncated = text[:max_chars]
        self._local.log_event("gen_ai.completion", {"content": truncated,
                                                      "truncated": len(text) > max_chars})
        if self._otel:
            self._otel.log_completion_event(truncated)

    def log_tool_call_event(
        self,
        tool_name: str,
        input_summary: str,
        result_summary: str,
        is_error: bool = False,
        tool_call_id: str = "",
    ) -> None:
        """Record a single Claude tool invocation (Bash, Read, Write, Edit, …)."""
        self._local.log_event("gen_ai.tool.call", {
            "tool": tool_name,
            "input": input_summary[:300],
            "result": result_summary[:300],
            "error": is_error,
        })
        if self._otel:
            self._otel.log_tool_call_event(
                tool_name, tool_call_id, input_summary[:300], result_summary[:300], is_error
            )

    def log_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        model: str = "",
        provider: str = "anthropic",
    ) -> None:
        """Record all token counts — includes Anthropic cache buckets."""
        self._local.log_metric("gen_ai.usage.input_tokens", input_tokens)
        self._local.log_metric("gen_ai.usage.output_tokens", output_tokens)
        if cache_creation_tokens:
            self._local.log_metric("gen_ai.usage.cache_creation.input_tokens", cache_creation_tokens)
        if cache_read_tokens:
            self._local.log_metric("gen_ai.usage.cache_read.input_tokens", cache_read_tokens)
        if self._mlflow:
            self._mlflow.log_metric("gen_ai.usage.input_tokens", input_tokens)
            self._mlflow.log_metric("gen_ai.usage.output_tokens", output_tokens)
            if cache_creation_tokens:
                self._mlflow.log_metric("gen_ai.usage.cache_creation.input_tokens", cache_creation_tokens)
            if cache_read_tokens:
                self._mlflow.log_metric("gen_ai.usage.cache_read.input_tokens", cache_read_tokens)
        if self._otel:
            self._otel.log_token_usage(
                input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
                model=model, provider=provider,
            )

    def log_model_usage(self, model_usage) -> None:
        """Record per-model token/cost breakdown from one or more ``ModelUsage`` records.

        Accepts a single :class:`~aet.tracking.claude_stream.ModelUsage` or a list of them;
        ``None`` is a no-op. Writes the ``per_model.<model>.*`` metrics the aet readers and the
        deep-trace example already expect (so existing tooling keeps working) plus a structured
        ``gen_ai.model.usage`` event, fanning out to every configured backend.
        """
        if model_usage is None:
            return
        items = model_usage if isinstance(model_usage, (list, tuple)) else [model_usage]
        for mu in items:
            if mu is None:
                continue
            safe = mu.model.replace("-", "_").replace(".", "_")
            total_billed = (mu.input_tokens + mu.cache_read_input_tokens
                            + mu.cache_creation_input_tokens + mu.output_tokens)
            self.log_metric(f"per_model.{safe}.cost_usd", mu.cost_usd)
            self.log_metric(f"per_model.{safe}.input_tokens_raw", mu.input_tokens)
            self.log_metric(f"per_model.{safe}.output_tokens", mu.output_tokens)
            self.log_metric(f"per_model.{safe}.cache_read_tokens", mu.cache_read_input_tokens)
            self.log_metric(f"per_model.{safe}.cache_creation_tokens", mu.cache_creation_input_tokens)
            self.log_metric(f"per_model.{safe}.total_billed_tokens", total_billed)
            self.log_event("gen_ai.model.usage", {
                "model": mu.model,
                "input_tokens_raw": mu.input_tokens,
                "output_tokens": mu.output_tokens,
                "cache_read_tokens": mu.cache_read_input_tokens,
                "cache_creation_tokens": mu.cache_creation_input_tokens,
                "total_billed_tokens": total_billed,
                "cost_usd": mu.cost_usd,
                "web_search_requests": mu.web_search_requests,
            })

    def log_ttft(self, ttft_s: float, provider: str = "anthropic", model: str = "") -> None:
        """Record time-to-first-chunk (streaming latency) as metric + span attribute + histogram."""
        self._local.log_metric("gen_ai.response.time_to_first_chunk", ttft_s)
        if self._mlflow:
            self._mlflow.log_metric("gen_ai.response.time_to_first_chunk", ttft_s)
        if self._otel:
            self._otel.log_ttft(ttft_s, provider=provider, model=model)

    def log_finish_reasons(self, reasons: list[str]) -> None:
        """Record gen_ai.response.finish_reasons on the current span."""
        if reasons:
            self._local.log_event("gen_ai.response.finish_reasons", {"reasons": reasons})
        if self._otel:
            self._otel.log_finish_reasons(reasons)

    def log_exception(self, exc_type: str, message: str, stacktrace: str = "") -> None:
        """Emit gen_ai.client.operation.exception event and log locally."""
        self._local.log_event("gen_ai.client.operation.exception", {
            "exception.type": exc_type,
            "exception.message": message,
        })
        if self._otel:
            self._otel.log_exception_event(exc_type, message, stacktrace)

    def log_inference_details(
        self,
        operation: str,
        provider: str,
        model: str = "",
        conversation_id: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        finish_reasons: list[str] | None = None,
        response_id: str = "",
        ttft_s: float = 0.0,
    ) -> None:
        """Emit gen_ai.client.inference.operation.details event (spec opt-in)."""
        if self._otel:
            self._otel.log_inference_details_event(
                operation=operation, provider=provider, model=model,
                conversation_id=conversation_id, stream=True,
                input_tokens=input_tokens, output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens, cache_creation_tokens=cache_creation_tokens,
                finish_reasons=finish_reasons or [], response_id=response_id, ttft_s=ttft_s,
            )

    def log_cost(self, cost_usd: float, model: str = "") -> None:
        """Record API cost in USD."""
        self._local.log_metric("aet.agent.cost_usd", cost_usd)
        if self._mlflow:
            self._mlflow.log_metric("aet.agent.cost_usd", cost_usd)
        if self._otel:
            self._otel.set_span_attributes({"aet.agent.cost_usd": cost_usd})
            if model:
                self._otel.set_span_attributes({"gen_ai.response.model": model})

    def log_agent_turns(self, num_turns: int) -> None:
        """Record number of agent turns (round-trips to the LLM)."""
        self._local.log_metric("aet.agent.num_turns", num_turns)
        if self._mlflow:
            self._mlflow.log_metric("aet.agent.num_turns", num_turns)
        if self._otel:
            self._otel.set_span_attribute("aet.agent.num_turns", num_turns)

    def log_session_id(self, session_id: str) -> None:
        """Record Claude Code session ID (enables /resume for replay)."""
        self._local.log_param("gen_ai.conversation.id", session_id)
        if self._mlflow:
            self._mlflow.log_param("gen_ai.conversation.id", session_id)
        if self._otel:
            self._otel.set_span_attribute("gen_ai.conversation.id", session_id)

    def log_permission_mode(self, mode: str, approvals_required: int = 0) -> None:
        """Record permission mode and whether human approvals were needed."""
        self._local.log_param("aet.agent.permission_mode", mode)
        self._local.log_metric("aet.agent.human_approvals_required", approvals_required)
        if self._mlflow:
            self._mlflow.log_param("aet.agent.permission_mode", mode)
        if self._otel:
            self._otel.set_span_attribute("aet.agent.permission_mode", mode)

    def log_file_context(self, input_files: list[str], output_files: list[str]) -> None:
        """Record files provided to / produced by the agent."""
        self._local.log_event("agent.file_context", {
            "input_files": input_files,
            "output_files": output_files,
            "input_count": len(input_files),
            "output_count": len(output_files),
        })
        if self._mlflow:
            self._mlflow.log_metric("input_file_count", len(input_files))
            self._mlflow.log_metric("output_file_count", len(output_files))

    def log_task_achievement(self, achieved: bool, score: float, rationale: str = "") -> None:
        """Record whether the agent achieved the given task."""
        label = "achieved" if achieved else "not_achieved"
        self._local.log_metric("task_achievement_score", score)
        self._local.log_event("task.achievement", {"achieved": achieved, "score": score, "rationale": rationale})
        if self._mlflow:
            self._mlflow.log_metric("task_achievement_score", score)
        self.log_evaluation_result("task_achievement", score, label)

    def record_llm_call(self, duration_s: float, input_tokens: int, output_tokens: int, model: str = "") -> None:
        """Manually record an LLM call's metrics (for cases openllmetry doesn't cover)."""
        if self._otel:
            self._otel.record_metric(
                op_duration=duration_s,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
            )

    def emit_tool_call_spans(
        self,
        tool_calls: list,
        turn_usage: list,
        t0_ns: int,
        stream_duration_s: float,
    ) -> None:
        """Create OTel child spans (tool calls + inference turns) — call while invoke_agent span is active."""
        if self._otel:
            self._otel.emit_tool_call_spans(tool_calls, turn_usage, t0_ns, stream_duration_s)

    def log_agent_trace(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        num_turns: int,
        duration_ms: int,
        tool_calls: list[dict],
    ) -> None:
        """Push a complete CLI agent invocation to the MLflow Traces tab.

        Each tool call becomes a child span, mirroring the SigNoz span tree
        without requiring the Anthropic Python SDK.
        """
        if not self._mlflow:
            return
        self._mlflow.log_agent_trace(
            run_id=self._config.run_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            num_turns=num_turns,
            duration_ms=duration_ms,
            tool_calls=tool_calls,
        )

    def get_traceparent_for_subprocess(self) -> str | None:
        """Return TRACEPARENT value for injecting into a subprocess environment."""
        if self._otel:
            return self._otel.get_traceparent_for_subprocess()
        return None

    # ------------------------------------------------------------------
    # Hardware benchmark / event-sourced trace methods

    def log_run_start(
        self,
        benchmark: str = "",
        variant: str = "",
        spec_version_hash: str = "",
        tb_version_hash: str = "",
        repo_initial_commit: str = "",
        tool_tier: str = "",
    ) -> None:
        self._local._log_event_rich("run.start", {
            "benchmark": benchmark,
            "variant": variant,
            "spec_version_hash": spec_version_hash,
            "tb_version_hash": tb_version_hash,
            "repo_initial_commit": repo_initial_commit,
            "tool_tier": tool_tier,
        }, stage="setup", actor="harness")

    def log_run_end(
        self,
        status: str,
        repo_final_commit: str = "",
        wall_time_s: float = 0.0,
    ) -> None:
        self._local._log_event_rich("run.end", {
            "status": status,
            "repo_final_commit": repo_final_commit,
            "wall_time_s": wall_time_s,
        }, stage="teardown", actor="harness")
        self._local.log_metric("run.wall_time_s", wall_time_s)

    def log_run_abort(self, reason: str, detail: str = "") -> None:
        self._local._log_event_rich("run.abort", {
            "reason": reason, "detail": detail,
        }, stage="teardown", actor="harness")

    def log_prompt_sent(
        self,
        prompt_path: str,
        prompt_hash: str = "",
        stage: str = "dispatch",
    ) -> str:
        """Emit prompt.sent event. Returns the event_id."""
        return self._local._log_event_rich("prompt.sent", {
            "prompt_path": prompt_path,
            "prompt_hash": prompt_hash,
        }, stage=stage, actor="harness", output_refs=[prompt_path])

    def log_llm_response(
        self,
        model: str,
        turn: int,
        tok_in: int,
        tok_out: int,
        tok_cached: int = 0,
        finish_reason: str = "",
        prompt_event_id: str | None = None,
    ) -> str:
        """Emit llm.response event. Returns the event_id."""
        return self._local._log_event_rich("llm.response", {
            "model": model,
            "turn": turn,
            "tok_in": tok_in,
            "tok_out": tok_out,
            "tok_cached": tok_cached,
            "finish_reason": finish_reason,
        }, stage="inference", actor=model,
        input_refs=[prompt_event_id] if prompt_event_id else None)

    def log_file_diff(
        self,
        path: str,
        op: str,
        sha256_before: str | None = None,
        sha256_after: str | None = None,
        iteration: int | None = None,
    ) -> None:
        """Emit file.diff event for agent file writes/edits/reads."""
        self._local._log_event_rich("file.diff", {
            "path": path,
            "op": op,
            "sha256_before": sha256_before,
            "sha256_after": sha256_after,
            "iteration": iteration,
        }, stage="agent_action", actor="agent")

    def log_tool_start(
        self,
        tool_name: str,
        tool_use_id: str,
        input_summary: str = "",
        turn: int = 0,
    ) -> None:
        self._local._log_event_rich("tool.start", {
            "tool": tool_name,
            "tool_use_id": tool_use_id,
            "input": input_summary[:300],
            "turn": turn,
        }, stage="agent_action", actor="agent")

    def log_tool_end(
        self,
        tool_name: str,
        tool_use_id: str,
        result_summary: str = "",
        is_error: bool = False,
        duration_s: float = 0.0,
    ) -> None:
        self._local._log_event_rich("tool.end", {
            "tool": tool_name,
            "tool_use_id": tool_use_id,
            "result": result_summary[:300],
            "is_error": is_error,
            "duration_s": duration_s,
        }, stage="agent_action", actor="agent")

    def log_tool_error(
        self,
        tool_name: str,
        tool_use_id: str,
        error_message: str,
    ) -> None:
        self._local._log_event_rich("tool.error", {
            "tool": tool_name,
            "tool_use_id": tool_use_id,
            "error": error_message[:500],
        }, stage="agent_action", actor="agent")

    def log_artifact_created(
        self,
        path: str,
        sha256: str | None,
        origin: str,
        size_bytes: int | None = None,
        protected: bool = False,
    ) -> None:
        self._local._log_event_rich("artifact.created", {
            "path": path,
            "sha256": sha256,
            "origin": origin,
            "size_bytes": size_bytes,
            "protected": protected,
        }, stage="artifact", actor="harness", output_refs=[path])

    def log_eval_test_result(
        self,
        test_name: str,
        passed: bool,
        iteration: int | None = None,
        failure_category: str | None = None,
        failure_detail: str = "",
        oracle_output: str = "",
    ) -> None:
        self._local._log_event_rich("eval.test_result", {
            "test_name": test_name,
            "passed": passed,
            "iteration": iteration,
            "failure_category": failure_category,
            "failure_detail": failure_detail[:500],
            "oracle_output": oracle_output[:1000],
        }, stage="eval", actor="oracle")
        self._local.log_metric(f"eval.{test_name}.passed", int(passed), step=iteration)

    def log_eval_score(
        self,
        testbench_pass: bool,
        localization_recall: float,
        localization_precision: float,
        regression_count: int = 0,
        tainted: bool = False,
        first_elaboration_iter: int | None = None,
        first_public_pass_iter: int | None = None,
    ) -> None:
        self._local._log_event_rich("eval.score", {
            "testbench_pass": testbench_pass,
            "localization_recall": localization_recall,
            "localization_precision": localization_precision,
            "regression_count": regression_count,
            "tainted": tainted,
            "first_elaboration_iter": first_elaboration_iter,
            "first_public_pass_iter": first_public_pass_iter,
        }, stage="eval", actor="harness")
        self._local.log_metric("hw.testbench_pass", int(testbench_pass))
        self._local.log_metric("hw.localization_recall", localization_recall)
        self._local.log_metric("hw.localization_precision", localization_precision)
        self._local.log_metric("hw.regression_count", regression_count)
        if first_elaboration_iter is not None:
            self._local.log_metric("hw.first_elaboration_iter", first_elaboration_iter)
        if first_public_pass_iter is not None:
            self._local.log_metric("hw.first_public_pass_iter", first_public_pass_iter)

    def log_synth_end(
        self,
        status: str,
        iteration: int | None = None,
        verilator_output: str = "",
        failure_category: str | None = None,
    ) -> None:
        self._local._log_event_rich("synth.end", {
            "status": status,
            "iteration": iteration,
            "verilator_output": verilator_output[:2000],
            "failure_category": failure_category,
        }, stage="eval", actor="oracle")

    def log_human_intervention(self, reason: str, detail: str = "") -> None:
        self._local._log_event_rich("human.intervention", {
            "reason": reason, "detail": detail,
        }, stage="human", actor="human")

    # ------------------------------------------------------------------
    # Milestone helpers

    def record_elaboration(self, iteration: int) -> None:
        """Mark the first iteration at which Verilator compile succeeded."""
        self._local._log_event_rich("eval.milestone", {
            "milestone": "first_elaboration", "iteration": iteration,
        }, stage="eval", actor="oracle")
        self._local.log_metric("hw.first_elaboration_iter", iteration)

    def record_public_pass(self, iteration: int) -> None:
        """Mark the first iteration at which the public testbench passed."""
        self._local._log_event_rich("eval.milestone", {
            "milestone": "first_public_pass", "iteration": iteration,
        }, stage="eval", actor="oracle")
        self._local.log_metric("hw.first_public_pass_iter", iteration)

    def record_regression(self, iteration: int, detail: str = "") -> None:
        """Increment regression counter (a previously passing run broke)."""
        self._local._log_event_rich("eval.regression", {
            "iteration": iteration, "detail": detail,
        }, stage="eval", actor="oracle")

    def log_eval_score_gen(
        self,
        functional_pass: bool,
        cycles: int | None = None,
        c_ref: int | None = None,
        ratio: float | None = None,
        perf: str = "",
        improved: bool = False,
        cheat_suspected: bool = False,
        cheat_flags: list | None = None,
        holdout_present: bool = True,
        holdout_file: str | None = None,
        tainted: bool = False,
    ) -> None:
        """Emit eval.score event for spec-to-rtl (generative) runs."""
        self._local._log_event_rich("eval.score", {
            "task": "spec-to-rtl",
            "functional_pass": functional_pass,
            "cycles": cycles,
            "c_ref": c_ref,
            "ratio": ratio,
            "perf": perf,
            "improved": improved,
            "cheat_suspected": cheat_suspected,
            "cheat_flags": cheat_flags or [],
            "holdout_present": holdout_present,
            "holdout_file": holdout_file,
            "tainted": tainted,
        }, stage="eval", actor="harness")
        self._local.log_metric("hw.functional_pass", int(functional_pass))
        self._local.log_metric("hw.tainted", int(tainted))
        self._local.log_metric("hw.cheat_suspected", int(cheat_suspected))
        if cycles is not None:
            self._local.log_metric("hw.cycles", cycles)
        if c_ref is not None:
            self._local.log_metric("hw.c_ref", c_ref)
        if ratio is not None:
            self._local.log_metric("hw.perf_ratio", ratio)
        if perf:
            self._local.log_param("hw.perf", perf)

    def log_integrity_check(
        self,
        passed: bool,
        violation_type: str = "",
        flags: list | None = None,
        detail: str = "",
    ) -> None:
        """Emit integrity.check event (cheat detection, taint scan, etc.)."""
        self._local._log_event_rich("integrity.check", {
            "passed": passed,
            "violation_type": violation_type,
            "flags": flags or [],
            "detail": detail,
        }, stage="eval", actor="harness")

    def log_iteration_result(
        self,
        iteration: int,
        oracle_output: str = "",
        passed: bool = False,
        failure_category: str | None = None,
        tok_in: int | None = None,
        elapsed_s: float | None = None,
    ) -> None:
        """Emit iter.result event for one oracle invocation during an agent run."""
        self._local._log_event_rich("iter.result", {
            "iteration": iteration,
            "passed": passed,
            "failure_category": failure_category,
            "oracle_output": oracle_output[:500],
            "tok_in": tok_in,
            "elapsed_s": elapsed_s,
        }, stage="eval", actor="oracle")

    # ------------------------------------------------------------------
    # Eval lifecycle events

    def log_eval_start(
        self,
        eval_type: str = "public",
        iteration: int | None = None,
    ) -> None:
        self._local._log_event_rich("eval.start", {
            "eval_type": eval_type, "iteration": iteration,
        }, stage="evaluation", actor="evaluator")

    def log_eval_stage_result(
        self,
        stage: str,
        passed: bool,
        score: float | None = None,
        max_score: float | None = None,
        iteration: int | None = None,
    ) -> None:
        self._local._log_event_rich("eval.stage_result", {
            "stage": stage, "passed": passed,
            "score": score, "max_score": max_score, "iteration": iteration,
        }, stage="evaluation", actor="evaluator")

    def log_eval_assertion_result(
        self,
        assertion: str,
        passed: bool,
        cycle: int | None = None,
        expected: str | None = None,
        observed: str | None = None,
        contract_id: str | None = None,
    ) -> None:
        self._local._log_event_rich("eval.assertion_result", {
            "assertion": assertion, "passed": passed,
            "cycle": cycle, "expected": expected,
            "observed": observed, "contract_id": contract_id,
        }, stage="evaluation", actor="evaluator")

    def log_eval_coverage_result(
        self,
        coverage_point: str,
        covered: int,
        total: int,
        missing: list | None = None,
    ) -> None:
        self._local._log_event_rich("eval.coverage_result", {
            "coverage_point": coverage_point,
            "covered": covered, "total": total,
            "coverage_rate": covered / total if total > 0 else 0.0,
            "missing": missing or [],
        }, stage="evaluation", actor="evaluator")

    def log_eval_end(
        self,
        total_passed: int,
        total_tests: int,
        score: float | None = None,
        eval_type: str = "public",
    ) -> None:
        pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        self._local._log_event_rich("eval.end", {
            "eval_type": eval_type,
            "total_passed": total_passed, "total_tests": total_tests,
            "pass_rate": pass_rate, "score": score,
        }, stage="evaluation", actor="evaluator")
        self._local.log_metric(f"eval.{eval_type}.pass_rate", pass_rate)
        if score is not None:
            self._local.log_metric(f"eval.{eval_type}.score", score)

    # ------------------------------------------------------------------
    # Synthesis / PPA events

    def log_synth_start(
        self,
        tool: str = "yosys",
        target_library: str | None = None,
        clock_target_ns: float | None = None,
    ) -> None:
        self._local._log_event_rich("synth.start", {
            "tool": tool,
            "target_library": target_library,
            "clock_target_ns": clock_target_ns,
        }, stage="synthesis", actor="tool")

    def log_ppa(
        self,
        area_um2: float | None = None,
        cell_count: int | None = None,
        critical_path_ns: float | None = None,
        slack_ns: float | None = None,
        power_uw: float | None = None,
        frequency_mhz: float | None = None,
        tool: str | None = None,
        warnings: int = 0,
        ppa_validity: str = "functional_pass_required",
    ) -> None:
        payload = {
            "area_um2": area_um2, "cell_count": cell_count,
            "critical_path_ns": critical_path_ns, "slack_ns": slack_ns,
            "power_uw": power_uw, "frequency_mhz": frequency_mhz,
            "tool": tool, "warnings": warnings, "ppa_validity": ppa_validity,
        }
        self._local._log_event_rich("ppa.report", payload,
                                    stage="synthesis", actor="tool")
        for key, val in [("ppa.area_um2", area_um2), ("ppa.cell_count", cell_count),
                         ("ppa.critical_path_ns", critical_path_ns),
                         ("ppa.slack_ns", slack_ns), ("ppa.power_uw", power_uw),
                         ("ppa.frequency_mhz", frequency_mhz)]:
            if val is not None:
                self._local.log_metric(key, val)

    # ------------------------------------------------------------------
    # Run error / timeout events

    def log_run_timeout(self, reason: str = "", elapsed_s: float | None = None) -> None:
        self._local._log_event_rich("run.timeout", {
            "reason": reason, "elapsed_s": elapsed_s,
        }, stage="teardown", actor="system")

    def log_run_error(self, error_type: str = "", detail: str = "") -> None:
        self._local._log_event_rich("run.error", {
            "error_type": error_type, "detail": detail,
        }, stage="teardown", actor="system")

    # ------------------------------------------------------------------
    # File lifecycle events

    def log_file_read(
        self,
        path: str,
        sha256: str | None = None,
        size_bytes: int | None = None,
    ) -> None:
        self._local._log_event_rich("file.read", {
            "path": path, "sha256": sha256, "size_bytes": size_bytes,
        }, stage="editing", actor="agent", input_refs=[path])

    def log_file_delete(
        self,
        path: str,
        sha256_before: str | None = None,
    ) -> None:
        self._local._log_event_rich("file.delete", {
            "path": path, "sha256_before": sha256_before,
        }, stage="editing", actor="agent")

    # ------------------------------------------------------------------
    # Git events

    def log_git_commit(
        self,
        sha: str,
        message: str = "",
        files_changed: int = 0,
        additions: int = 0,
        deletions: int = 0,
    ) -> None:
        self._local._log_event_rich("git.commit", {
            "sha": sha, "message": message,
            "files_changed": files_changed,
            "additions": additions, "deletions": deletions,
        }, stage="editing", actor="agent")

    def log_git_diff(
        self,
        files_changed: int = 0,
        additions: int = 0,
        deletions: int = 0,
        ref_before: str | None = None,
        ref_after: str | None = None,
    ) -> None:
        self._local._log_event_rich("git.diff", {
            "files_changed": files_changed,
            "additions": additions, "deletions": deletions,
            "ref_before": ref_before, "ref_after": ref_after,
        }, stage="editing", actor="system")

    # ------------------------------------------------------------------
    # Human intervention sub-types

    def log_human_clarification(self, question: str, response: str = "") -> None:
        self._local._log_event_rich("human.clarification", {
            "question": question, "response": response,
        }, stage="planning", actor="human")

    def log_human_patch(
        self,
        file_path: str,
        description: str = "",
        iteration: int | None = None,
    ) -> None:
        self._local._log_event_rich("human.patch", {
            "file_path": file_path, "description": description,
            "iteration": iteration,
        }, stage="editing", actor="human")
        self._local.log_metric("human.patch_count", 1)

    def log_human_override(
        self,
        what: str,
        why: str = "",
        artifact_affected: str | None = None,
    ) -> None:
        self._local._log_event_rich("human.override", {
            "what": what, "why": why,
            "artifact_affected": artifact_affected,
        }, stage="evaluation", actor="human")

    # ------------------------------------------------------------------
    # Utility writers

    def write_run_record(self, extra: dict | None = None) -> Path:
        """Write `run_record.json` at the run root (see :mod:`aet.tracking.reports`)."""
        c = self._config
        return reports.write_run_record(
            c.run_path, run_id=c.run_id, project=c.project, suite=c.suite, target=c.target,
            method=c.method, seed=c.seed, mode=c.mode, extra=extra)

    def write_summary_metrics(self, extra: dict | None = None) -> Path:
        """Write `metrics/summary_metrics.json` (see :mod:`aet.tracking.reports`)."""
        c = self._config
        return reports.write_summary_metrics(
            c.run_path, run_id=c.run_id, project=c.project, suite=c.suite, method=c.method,
            seed=c.seed, target=c.target, extra=extra)

    def write_eval_report(
        self,
        tests: list[dict],
        contracts: list[dict] | None = None,
        assertions: list[dict] | None = None,
        coverage: list[dict] | None = None,
        extra: dict | None = None,
    ) -> Path:
        """Write `eval_report.json` (see :mod:`aet.tracking.reports`)."""
        return reports.write_eval_report(
            self._config.run_path, run_id=self._config.run_id, tests=tests, contracts=contracts,
            assertions=assertions, coverage=coverage, extra=extra)

    def write_metrics_structured(
        self,
        cost: dict,
        quality: dict,
        process: dict,
        extra: dict | None = None,
    ) -> Path:
        """Write structured `metrics.json` (see :mod:`aet.tracking.reports`)."""
        return reports.write_metrics_structured(
            self._config.run_path, run_id=self._config.run_id, cost=cost, quality=quality,
            process=process, extra=extra)

    # ------------------------------------------------------------------
    def finish(self, status: str, message: str | None = None) -> None:
        self._local.log_event("run.finished", {"status": status, "message": message})
        if self._mlflow:
            self._mlflow.finish(status, message)

    def close(self) -> None:
        """Flush and close all backends. Call after finish()."""
        if self._otel:
            self._otel.uninstrument_openllmetry()

    def patch_manifest(self, manifest_path: Path) -> None:
        """Write tracking IDs back into run_manifest.yaml."""
        if self._mlflow:
            self._mlflow.patch_manifest(manifest_path)

    # ------------------------------------------------------------------
    @property
    def mlflow_run_id(self) -> str | None:
        return self._mlflow.run_id if self._mlflow else None

    @property
    def mlflow_run_url(self) -> str | None:
        run_id = self.mlflow_run_id
        if not run_id:
            return None
        uri = (self._config.mlflow_tracking_uri or "http://localhost:5000").rstrip("/")
        exp_id = self._mlflow.experiment_id if self._mlflow else "0"
        return f"{uri}/#/experiments/{exp_id}/runs/{run_id}"

    @property
    def otel_trace_id(self) -> str | None:
        return self._otel.trace_id if self._otel else None

    @property
    def mode(self) -> str:
        return self._config.mode

    @property
    def logs_dir(self) -> Path:
        return self._config.run_path / "logs"

    def print_summary(self) -> None:
        """Print tracking status to stdout."""
        mlf_status = "enabled" if (self._mlflow and self._mlflow._enabled) else "disabled"
        otel_status = "enabled" if (self._otel and self._otel._enabled) else "disabled"
        print(f"  tracking mode:  {self._config.mode}")
        print(f"  MLflow:         {mlf_status}" +
              (f" (run_id: {self.mlflow_run_id})" if self.mlflow_run_id else ""))
        print(f"  OTel:           {otel_status}")
        print(f"  local logs:     {self.logs_dir}")
