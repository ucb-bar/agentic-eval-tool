"""multi_provider_eval.py — aet instrumentation for Anthropic SDK, OpenAI/Codex, and AWS Bedrock.

Runs the same task through multiple providers and compares cost/latency/quality.
Every call produces fully spec-compliant OTel GenAI semconv spans and metrics.

Usage (dry-run, no API keys needed):
    .venv/bin/python examples/multi_provider_eval.py --dry-run

Usage (real, needs ANTHROPIC_API_KEY and/or OPENAI_API_KEY):
    .venv/bin/python examples/multi_provider_eval.py \\
        --providers anthropic,openai \\
        --otel-endpoint http://localhost:4318 \\
        --mlflow-uri http://localhost:5001 \\
        --tracking full

Each provider run creates:
  - An invoke_workflow span (the whole comparison)
    - An invoke_agent span per provider (one child per provider)
      - A chat span for the direct SDK call
      - A gen_ai.client.inference.operation.details event
      - A gen_ai.evaluation.result event with explanation
  - SigNoz: all spans linked by trace ID for waterfall comparison
  - MLflow: metrics per provider as separate child runs (nested)
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from aet.tracking import EvalRunLogger
from aet.tracking.semconv import (
    PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_AWS_BEDROCK,
    SERVER_ADDRESS_ANTHROPIC, SERVER_ADDRESS_OPENAI,
    TOKEN_TYPE_INPUT, TOKEN_TYPE_OUTPUT,
)

TASK = (
    "In exactly three bullet points, name the key tradeoffs between "
    "static and dynamic compilation in ML compilers."
)

# ── Per-provider default server addresses ──────────────────────────────────────
_SERVER_ADDR = {
    PROVIDER_ANTHROPIC: SERVER_ADDRESS_ANTHROPIC,
    PROVIDER_OPENAI: SERVER_ADDRESS_OPENAI,
    PROVIDER_AWS_BEDROCK: "bedrock-runtime.us-east-1.amazonaws.com",
}

# ── Synthetic dry-run responses (so the example runs without API keys) ─────────
_DRY_RESPONSES = {
    PROVIDER_ANTHROPIC: {
        "text": (
            "• **Optimization depth**: Static compilers do exhaustive graph rewrites "
            "at compile time; dynamic (JIT) compilers defer until runtime when shapes are known.\n"
            "• **Startup latency**: Static compilation pays cost upfront; JIT incurs "
            "per-invocation overhead on first execution.\n"
            "• **Flexibility**: Dynamic compilation adapts to runtime data (batch size, "
            "dtype), while static compilation locks in assumptions made at export time."
        ),
        "input_tokens": 312, "output_tokens": 98, "cache_read": 14500, "cache_create": 800,
        "model": "claude-sonnet-4-6", "latency_s": 2.1, "ttft_s": 0.45, "cost_usd": 0.018,
        "response_id": "msg_dry_anthropic_001",
    },
    PROVIDER_OPENAI: {
        "text": (
            "• **Shape flexibility**: AOT compilers require fixed shapes; JIT compilers "
            "like torch.compile can specialize at runtime for variable inputs.\n"
            "• **Recompilation cost**: Dynamic compilation may trigger retracing when "
            "input shapes change, adding latency; static avoids this.\n"
            "• **Deployment simplicity**: Static artifacts (ONNX, TorchScript) are "
            "portable and version-stable; dynamic requires the full runtime stack."
        ),
        "input_tokens": 290, "output_tokens": 105, "cache_read": 0, "cache_create": 0,
        "model": "gpt-4o-mini", "latency_s": 1.7, "ttft_s": 0.38, "cost_usd": 0.003,
        "response_id": "chatcmpl-dry-openai-001",
    },
    PROVIDER_AWS_BEDROCK: {
        "text": (
            "• **Portability**: Static compilation produces self-contained artifacts; "
            "Bedrock dynamic inference adapts per request.\n"
            "• **Cold-start latency**: Static compiled models load faster after warm-up; "
            "dynamic compilation pays shape-specialization cost on first call.\n"
            "• **Operator coverage**: Static compilers may reject unsupported ops at "
            "export time; dynamic compilation falls back to eager execution."
        ),
        "input_tokens": 305, "output_tokens": 102, "cache_read": 0, "cache_create": 0,
        "model": "us.anthropic.claude-sonnet-4-6-20251015-v1:0",
        "latency_s": 2.3, "ttft_s": 0.52, "cost_usd": 0.014,
        "response_id": "bedrock-dry-001",
    },
}

# Bedrock Sonnet model — requires inference profile prefix "us." for on-demand
_BEDROCK_SONNET_CANDIDATES = [
    "us.anthropic.claude-sonnet-4-6-20251015-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
]
BEDROCK_REGION = "us-east-1"

_BULLET_PREFIXES = ("•", "-", "*", "–", "—", "▸", "›")


def _is_bullet(line: str) -> bool:
    s = line.strip()
    if any(s.startswith(p + " ") or s.startswith(p + "\t") for p in _BULLET_PREFIXES):
        return True
    # Numbered: "1. " or "1) "
    parts = s.split(None, 1)
    return bool(parts and parts[0].rstrip(".):").isdigit())


@dataclass
class ProviderResult:
    provider: str
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: float
    latency_s: float
    ttft_s: float
    response_id: str
    error: str = ""

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    def quality_score(self) -> float:
        """Heuristic: did we get three bullet points (any common format)?"""
        bullets = [l for l in self.text.splitlines() if _is_bullet(l)]
        return 1.0 if len(bullets) >= 3 else (0.5 if len(bullets) >= 1 else 0.0)

    def quality_label(self) -> str:
        s = self.quality_score()
        return "pass" if s == 1.0 else ("partial" if s == 0.5 else "fail")

    def quality_explanation(self) -> str:
        bullets = [l for l in self.text.splitlines() if _is_bullet(l)]
        return f"{len(bullets)} bullet point(s) found; expected 3."


def _dry_result(provider: str) -> ProviderResult:
    d = _DRY_RESPONSES[provider]
    return ProviderResult(
        provider=provider, model=d["model"], text=d["text"],
        input_tokens=d["input_tokens"], output_tokens=d["output_tokens"],
        cache_read_tokens=d.get("cache_read", 0),
        cache_creation_tokens=d.get("cache_create", 0),
        cost_usd=d["cost_usd"], latency_s=d["latency_s"],
        ttft_s=d["ttft_s"], response_id=d["response_id"],
    )


def _call_anthropic(task: str, dry_run: bool) -> ProviderResult:
    if dry_run:
        return _dry_result(PROVIDER_ANTHROPIC)
    try:
        import anthropic
        t0 = time.monotonic()
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": task}],
        )
        latency_s = time.monotonic() - t0
        usage = msg.usage
        return ProviderResult(
            provider=PROVIDER_ANTHROPIC,
            model=msg.model,
            text=msg.content[0].text,
            input_tokens=usage.input_tokens + getattr(usage, "cache_read_input_tokens", 0)
                         + getattr(usage, "cache_creation_input_tokens", 0),
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            cost_usd=0.0,
            latency_s=latency_s,
            ttft_s=latency_s,
            response_id=msg.id,
        )
    except Exception as e:
        return ProviderResult(provider=PROVIDER_ANTHROPIC, model="claude-sonnet-4-6",
                              text="", input_tokens=0, output_tokens=0,
                              cache_read_tokens=0, cache_creation_tokens=0,
                              cost_usd=0.0, latency_s=0.0, ttft_s=0.0,
                              response_id="", error=str(e))


def _call_openai(task: str, dry_run: bool) -> ProviderResult:
    if dry_run:
        return _dry_result(PROVIDER_OPENAI)
    try:
        import openai
        t0 = time.monotonic()
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": task}],
            max_tokens=512,
        )
        latency_s = time.monotonic() - t0
        usage = resp.usage
        return ProviderResult(
            provider=PROVIDER_OPENAI,
            model=resp.model,
            text=resp.choices[0].message.content,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            cost_usd=0.0,
            latency_s=latency_s,
            ttft_s=latency_s,
            response_id=resp.id,
        )
    except Exception as e:
        return ProviderResult(provider=PROVIDER_OPENAI, model="gpt-4o-mini",
                              text="", input_tokens=0, output_tokens=0,
                              cache_read_tokens=0, cache_creation_tokens=0,
                              cost_usd=0.0, latency_s=0.0, ttft_s=0.0,
                              response_id="", error=str(e))


def _call_bedrock(task: str, dry_run: bool) -> ProviderResult:
    if dry_run:
        return _dry_result(PROVIDER_AWS_BEDROCK)
    import os
    import requests as _requests
    token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not token:
        return ProviderResult(
            provider=PROVIDER_AWS_BEDROCK, model=_BEDROCK_SONNET_CANDIDATES[0],
            text="", input_tokens=0, output_tokens=0,
            cache_read_tokens=0, cache_creation_tokens=0,
            cost_usd=0.0, latency_s=0.0, ttft_s=0.0,
            response_id="",
            error="AWS_BEARER_TOKEN_BEDROCK not set — pass --env-file or export it",
        )
    base = f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com"
    body = {
        "messages": [{"role": "user", "content": [{"text": task}]}],
        "inferenceConfig": {"maxTokens": 512, "temperature": 0.0},
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    last_err = ""
    for model_id in _BEDROCK_SONNET_CANDIDATES:
        try:
            t0 = time.monotonic()
            r = _requests.post(
                f"{base}/model/{model_id}/converse",
                headers=headers, data=json.dumps(body), timeout=60.0,
            )
            latency_s = time.monotonic() - t0
            if r.status_code >= 400:
                last_err = f"Bedrock {r.status_code} for {model_id}: {r.text[:200]}"
                continue
            data = r.json()
            text = "".join(p.get("text", "") for p in data["output"]["message"]["content"])
            usage = data.get("usage", {})
            return ProviderResult(
                provider=PROVIDER_AWS_BEDROCK, model=model_id, text=text,
                input_tokens=usage.get("inputTokens", 0),
                output_tokens=usage.get("outputTokens", 0),
                cache_read_tokens=usage.get("cacheReadInputTokenCount", 0),
                cache_creation_tokens=usage.get("cacheWriteInputTokenCount", 0),
                cost_usd=0.0, latency_s=latency_s, ttft_s=latency_s,
                response_id=r.headers.get("x-amzn-requestid", ""),
            )
        except Exception as e:
            last_err = str(e)
    return ProviderResult(
        provider=PROVIDER_AWS_BEDROCK, model=_BEDROCK_SONNET_CANDIDATES[0],
        text="", input_tokens=0, output_tokens=0,
        cache_read_tokens=0, cache_creation_tokens=0,
        cost_usd=0.0, latency_s=0.0, ttft_s=0.0,
        response_id="", error=last_err,
    )


_CALLERS = {
    PROVIDER_ANTHROPIC: _call_anthropic,
    PROVIDER_OPENAI: _call_openai,
    PROVIDER_AWS_BEDROCK: _call_bedrock,
}


def _record_provider(result: ProviderResult, logger: EvalRunLogger, capture_content: bool) -> None:
    """Record all semconv-compliant metrics/events for one provider call."""
    # ── token usage (all four buckets) ────────────────────────────────
    logger.log_token_usage(
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_creation_tokens=result.cache_creation_tokens,
        cache_read_tokens=result.cache_read_tokens,
        model=result.model,
        provider=result.provider,
    )

    # ── TTFT + operation duration ──────────────────────────────────────
    if result.ttft_s:
        logger.log_ttft(result.ttft_s, provider=result.provider, model=result.model)
    logger.log_metric("gen_ai.client.operation.duration", result.latency_s)

    # ── cost ──────────────────────────────────────────────────────────
    if result.cost_usd:
        logger.log_cost(result.cost_usd, model=result.model)

    # ── inference details event (spec: gen_ai.client.inference.operation.details)
    logger.log_inference_details(
        operation="chat",
        provider=result.provider,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_tokens=result.cache_read_tokens,
        cache_creation_tokens=result.cache_creation_tokens,
        finish_reasons=["end_turn"] if result.text else ["error"],
        response_id=result.response_id,
        ttft_s=result.ttft_s,
    )

    # ── exception event if call failed ────────────────────────────────
    if result.error:
        logger.log_exception(
            exc_type="ProviderError",
            message=result.error,
        )

    # ── evaluation ────────────────────────────────────────────────────
    score = result.quality_score()
    label = result.quality_label()
    explanation = result.quality_explanation()
    logger.log_task_achievement(achieved=score >= 1.0, score=score, rationale=explanation)
    logger.log_evaluation_result(
        name="bullet_point_format",
        score=score,
        label=label,
        explanation=explanation,
    )

    # ── content capture (opt-in) ──────────────────────────────────────
    if capture_content:
        logger.log_prompt(TASK, role="user")
        if result.text:
            logger.log_completion(result.text)


def run_comparison(
    providers: list[str],
    run_path: Path,
    otel_endpoint: str | None,
    mlflow_uri: str | None,
    tracking_mode: str,
    dry_run: bool,
    capture_content: bool,
    enable_openllmetry: bool = True,
) -> dict:
    import uuid
    run_id = f"multi_provider_{uuid.uuid4().hex[:8]}"
    run_path.mkdir(parents=True, exist_ok=True)

    logger = EvalRunLogger.start(
        project="aet-multi-provider",
        suite="comparison",
        target="compiler_ir",
        method="multi_provider",
        seed=1,
        run_id=run_id,
        run_path=run_path,
        tracking_mode=tracking_mode,
        mlflow_tracking_uri=mlflow_uri,
        experiment_name="aet-provider-comparison",
        otel_endpoint=otel_endpoint,
        enable_openllmetry=enable_openllmetry,
    )

    results: dict[str, ProviderResult] = {}

    with logger.start_run_span("provider_comparison"):
        logger.log_param("task", TASK[:120])
        logger.log_param("providers", ",".join(providers))
        logger.log_param("dry_run", dry_run)

        for provider in providers:
            caller = _CALLERS.get(provider)
            if not caller:
                print(f"[aet] Unknown provider: {provider}, skipping.")
                continue

            # Each provider gets its own invoke_agent span
            with logger.start_agent_span(
                agent_name=f"{provider}-agent",
                model=_DRY_RESPONSES.get(provider, {}).get("model", provider),
                provider=provider,
            ):
                # Direct API call gets its own chat span nested inside invoke_agent
                with logger.start_inference_span(
                    model=_DRY_RESPONSES.get(provider, {}).get("model", provider),
                    provider=provider,
                    server_address=_SERVER_ADDR.get(provider, ""),
                    operation="chat",
                    stream=False,
                ):
                    t0 = time.monotonic()
                    result = caller(TASK, dry_run)
                    result.latency_s = result.latency_s or (time.monotonic() - t0)

                _record_provider(result, logger, capture_content)
                results[provider] = result

                status = "pass" if result.quality_score() == 1.0 else ("partial" if result.error == "" else "error")
                sym = "✓" if status == "pass" else ("~" if status == "partial" else "✗")
                print(
                    f"  {sym}  {provider:<20}  model={result.model:<40}  "
                    f"tokens={result.input_tokens}+{result.output_tokens}  "
                    f"cost=${result.cost_usd:.4f}  latency={result.latency_s:.2f}s  "
                    f"quality={result.quality_label()}"
                )

        # ── cross-provider summary metrics ─────────────────────────────
        best_cost = min((r.cost_usd for r in results.values() if r.cost_usd), default=0)
        best_latency = min((r.latency_s for r in results.values() if r.latency_s), default=0)
        best_quality = max(r.quality_score() for r in results.values()) if results else 0
        logger.log_metric("comparison.best_cost_usd", best_cost)
        logger.log_metric("comparison.best_latency_s", best_latency)
        logger.log_metric("comparison.best_quality_score", best_quality)
        logger.log_metric("comparison.num_providers", len(results))

        logger.finish(status="pass" if best_quality == 1.0 else "partial")

    trace_id = logger.otel_trace_id
    print(f"\n[aet] run: {run_id}")
    if trace_id and otel_endpoint:
        base = otel_endpoint.rstrip("/").replace(":4318", ":8080").replace(":4317", ":8080")
        print(f"       signoz:   {base}/trace/{trace_id}")
    if url := logger.mlflow_run_url:
        print(f"       mlflow:   {url}")

    return {"run_id": run_id, "trace_id": trace_id,
            "results": {p: {"score": r.quality_score(), "cost": r.cost_usd,
                            "latency_s": r.latency_s, "error": r.error}
                        for p, r in results.items()}}


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no override)."""
    import os
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-provider LLM comparison with full OTel semconv")
    p.add_argument("--providers", default="anthropic,openai,aws.bedrock",
                   help="Comma-separated list: anthropic,openai,aws.bedrock")
    p.add_argument("--run-path", type=Path, default=Path("/tmp/aet-multi-provider"))
    p.add_argument("--otel-endpoint", default=None)
    p.add_argument("--mlflow-uri", default=None)
    p.add_argument("--tracking", dest="tracking_mode", default="local",
                   choices=["local", "mlflow", "full", "debug"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--capture-content", action="store_true")
    p.add_argument("--env-file", type=Path, default=None,
                   help="Path to a .env file to load API keys from (e.g. /path/to/.env)")
    p.add_argument("--no-openllmetry", dest="openllmetry", action="store_false", default=True,
                   help="Disable OpenLLMetry auto-instrumentation of Anthropic/OpenAI SDKs")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.env_file:
        _load_env_file(args.env_file)
    providers = [p.strip() for p in args.providers.split(",")]
    ollm_note = "" if args.openllmetry else " (OpenLLMetry disabled)"
    print(f"\n[aet] Comparing {providers} {'(dry-run)' if args.dry_run else ''}{ollm_note}\n")
    result = run_comparison(
        providers=providers,
        run_path=args.run_path,
        otel_endpoint=args.otel_endpoint,
        mlflow_uri=args.mlflow_uri,
        tracking_mode=args.tracking_mode,
        dry_run=args.dry_run,
        capture_content=args.capture_content,
        enable_openllmetry=args.openllmetry,
    )
    print(json.dumps(result, indent=2))
