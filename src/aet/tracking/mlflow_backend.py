"""MLflow backend — optional. Degrades gracefully if mlflow is not installed or server is down."""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

from aet.tracking.types import TrackingConfig


class MLflowBackend:
    def __init__(self, config: TrackingConfig, local) -> None:
        self._enabled = False
        self._run = None
        self._mlflow = None
        self._local = local
        self._config = config

        try:
            import mlflow as _mlflow
            self._mlflow = _mlflow
        except ImportError:
            local.warn(
                "mlflow not installed; mlflow tracking disabled. "
                "Install with: uv pip install 'aet[tracking]'"
            )
            return

        self._setup(config, local)

    def _setup(self, config: TrackingConfig, local) -> None:
        try:
            import os
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
            uri = config.mlflow_tracking_uri or "http://localhost:5000"
            self._mlflow.set_tracking_uri(uri)
            exp_name = config.experiment_name or f"targetgen-evals-{config.target}"
            self._mlflow.set_experiment(exp_name)
            run_kwargs: dict = dict(
                run_name=config.run_id,
                tags={
                    "target": config.target,
                    "method": config.method,
                    "seed": str(config.seed),
                    "tracking_mode": config.mode,
                },
            )
            if config.parent_run_id:
                run_kwargs["parent_run_id"] = config.parent_run_id
            self._run = self._mlflow.start_run(**run_kwargs)
            self._enabled = True
            mlflow_run_id = self._run.info.run_id
            local.log_param("mlflow_run_id", mlflow_run_id)
            local.log_param("mlflow_tracking_uri", uri)
            local.log_param("mlflow_experiment_name", exp_name)
            # Enable auto-tracing for direct Anthropic SDK calls made within this process.
            # No-ops if anthropic is not installed; does not affect CLI subprocess captures.
            try:
                self._mlflow.anthropic.autolog()
            except Exception:
                pass
        except Exception as e:
            local.warn(
                f"MLflow setup failed ({e}); falling back to local-only tracking. "
                "Is the MLflow server running?"
            )
            self._enabled = False

    @property
    def run_id(self) -> str | None:
        if self._enabled and self._run:
            return self._run.info.run_id
        return None

    @property
    def experiment_id(self) -> str | None:
        if self._enabled and self._run:
            return self._run.info.experiment_id
        return None

    # ------------------------------------------------------------------
    def log_param(self, name: str, value: Any) -> None:
        if not self._enabled:
            return
        try:
            self._mlflow.log_param(name, value)
        except Exception as e:
            self._local.warn(f"MLflow log_param failed: {e}")

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            # MLflow has a 100-param limit per call; chunk if needed
            items = list(params.items())
            for i in range(0, len(items), 100):
                self._mlflow.log_params(dict(items[i : i + 100]))
        except Exception as e:
            self._local.warn(f"MLflow log_params failed: {e}")

    # ------------------------------------------------------------------
    def log_metric(self, name: str, value: Any, step: int | None = None, source: str | None = None) -> None:
        if not self._enabled or value is None:
            return
        if not isinstance(value, (int, float, bool)):
            return
        try:
            self._mlflow.log_metric(name, float(value), step=step)
        except Exception as e:
            self._local.warn(f"MLflow log_metric({name}) failed: {e}")

    def log_metrics(self, metrics: dict[str, Any], prefix: str | None = None) -> None:
        if not self._enabled:
            return
        numeric = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, bool)) and v is not None:
                name = f"{prefix}.{k}" if prefix else k
                numeric[name] = float(v)
        if not numeric:
            return
        try:
            self._mlflow.log_metrics(numeric)
        except Exception as e:
            self._local.warn(f"MLflow log_metrics failed: {e}")

    def log_step_metric(self, name: str, value: float, step: int) -> None:
        if not self._enabled:
            return
        try:
            self._mlflow.log_metric(name, float(value), step=step)
        except Exception as e:
            self._local.warn(f"MLflow log_step_metric({name}) failed: {e}")

    # ------------------------------------------------------------------
    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if not self._enabled:
            return
        p = Path(path)
        if not p.exists():
            return
        try:
            self._mlflow.log_artifact(str(p), artifact_path)
        except Exception as e:
            self._local.warn(f"MLflow log_artifact({p.name}) failed: {e}")

    def log_artifacts(self, path: Path, artifact_path: str | None = None) -> None:
        if not self._enabled:
            return
        p = Path(path)
        if not p.exists():
            return
        if p.is_dir():
            try:
                self._mlflow.log_artifacts(str(p), artifact_path)
            except Exception as e:
                self._local.warn(f"MLflow log_artifacts({p.name}) failed: {e}")
        else:
            self.log_artifact(p, artifact_path)

    def log_generated_dir_as_tarball(self, generated_dir: Path, target: str) -> None:
        """Log the generated target directory as a single tarball artifact."""
        if not self._enabled or not generated_dir.exists():
            return
        try:
            artifacts_dir = generated_dir.parent.parent / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            tarball = artifacts_dir / f"generated_{target}_mlir.tar.gz"
            with tarfile.open(tarball, "w:gz") as tf:
                tf.add(generated_dir, arcname=generated_dir.name)
            self._mlflow.log_artifact(str(tarball), "generated")
        except Exception as e:
            self._local.warn(f"MLflow tarball artifact failed: {e}")

    # ------------------------------------------------------------------
    def finish(self, status: str, message: str | None = None) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            mlflow_status = "FINISHED" if status in ("pass", "success") else "FAILED"
            self._mlflow.end_run(status=mlflow_status)
        except Exception as e:
            self._local.warn(f"MLflow end_run failed: {e}")

    def log_agent_trace(
        self,
        run_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        num_turns: int,
        duration_ms: int,
        tool_calls: list[dict],
    ) -> None:
        """Record a complete CLI agent invocation as an MLflow trace span.

        Creates one root span (invoke_agent) with one child span per tool call,
        so the MLflow Traces tab shows the same structure as SigNoz.
        """
        if not self._enabled:
            return
        try:
            with self._mlflow.start_span(name="invoke_agent", span_type="AGENT") as root:
                root.set_inputs({"run_id": run_id, "model": model, "num_turns": num_turns})
                root.set_outputs({
                    "cost_usd": cost_usd,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "duration_ms": duration_ms,
                })
                root.set_attribute("gen_ai.request.model", model)
                root.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                root.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                root.set_attribute("aet.cost_usd", cost_usd)
                root.set_attribute("aet.num_turns", num_turns)
                for tc in tool_calls:
                    with self._mlflow.start_span(
                        name=tc.get("name", "tool"),
                        span_type="TOOL",
                        parent_span=root,
                    ) as ts:
                        ts.set_inputs(tc.get("input", {}))
                        ts.set_outputs({"result": tc.get("result", "")[:200]})
                        ts.set_attribute("gen_ai.tool.name", tc.get("name", ""))
                        ts.set_attribute("aet.tool.duration_s", tc.get("duration_s", 0.0))
                        ts.set_attribute("aet.tool.is_error", tc.get("is_error", False))
                        if tc.get("is_mcp"):
                            ts.set_attribute("aet.tool.is_mcp", True)
        except Exception as e:
            self._local.warn(f"MLflow log_agent_trace failed: {e}")

    def patch_manifest(self, manifest_path: Path) -> None:
        """Write mlflow run_id back into run_manifest.yaml observability block."""
        if not self._enabled or not manifest_path.exists() or self._run is None:
            return
        try:
            import yaml
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f) or {}
            obs = manifest.setdefault("observability", {})
            mlf = obs.setdefault("mlflow", {})
            mlf["enabled"] = True
            mlf["run_id"] = self._run.info.run_id
            with open(manifest_path, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=True, allow_unicode=True)
        except Exception as e:
            self._local.warn(f"MLflow manifest patch failed: {e}")
