"""EvalRunLogger — single tracking API for the aet harness."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

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

    def start_tool_span(self, tool_name: str, validator_name: str | None = None):
        """Context manager for a tool/validator execution — execute_tool."""
        if self._otel:
            return self._otel.start_tool_span(tool_name, validator_name=validator_name)
        return nullcontext()

    def log_evaluation_result(self, name: str, score: float, label: str) -> None:
        """Emit a gen_ai.evaluation.result event on the current OTel span."""
        self._local.log_event("evaluation.result", {"name": name, "score": score, "label": label})
        if self._otel:
            self._otel.log_evaluation_event(name, score, label)

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
                input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, model
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

    def get_traceparent_for_subprocess(self) -> str | None:
        """Return TRACEPARENT value for injecting into a subprocess environment."""
        if self._otel:
            return self._otel.get_traceparent_for_subprocess()
        return None

    # ------------------------------------------------------------------
    def finish(self, status: str, message: str | None = None) -> None:
        self._local.log_event("run.finished", {"status": status, "message": message})
        if self._mlflow:
            self._mlflow.finish(status, message)

    def close(self) -> None:
        """Flush and close all backends. Call after finish()."""
        pass  # local backend has no buffers; MLflow run already ended in finish()

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
        return f"{uri}/#/experiments/0/runs/{run_id}"

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
