"""OTel GenAI semantic convention attribute and event name constants.

Covers the Anthropic GenAI semconv (status: Development, 2025) as provided
by the official OTel spec, plus aet-specific extensions.

Set OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental to emit the
latest experimental version of these conventions.
"""

from __future__ import annotations

# ── Operation ────────────────────────────────────────────────────────────────
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

# ── Provider / model ─────────────────────────────────────────────────────────
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"

# ── Request parameters ───────────────────────────────────────────────────────
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"
GEN_AI_REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"

# ── Token usage (Anthropic semconv note: input_tokens excludes cached tokens) ─
# gen_ai.usage.input_tokens = input_tokens + cache_read + cache_creation
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"

# ── Conversation / session ───────────────────────────────────────────────────
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"

# ── Agentic ──────────────────────────────────────────────────────────────────
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_VERSION = "gen_ai.agent.version"
GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
GEN_AI_WORKFLOW_NAME = "gen_ai.workflow.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_IS_ERROR = "gen_ai.tool.is_error"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
GEN_AI_DATA_SOURCE_ID = "gen_ai.data_source.id"

# ── Token type values ─────────────────────────────────────────────────────────
GEN_AI_TOKEN_TYPE = "gen_ai.token.type"
TOKEN_TYPE_INPUT = "input"
TOKEN_TYPE_OUTPUT = "output"
TOKEN_TYPE_CACHE_READ = "cache_read"
TOKEN_TYPE_CACHE_CREATION = "cache_creation"

# ── Provider name values ──────────────────────────────────────────────────────
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_AWS_BEDROCK = "aws.bedrock"
PROVIDER_AZURE_OPENAI = "azure.ai.openai"
PROVIDER_GCP_VERTEX = "gcp.vertex_ai"

# ── Server ────────────────────────────────────────────────────────────────────
SERVER_ADDRESS_ANTHROPIC = "api.anthropic.com"
SERVER_ADDRESS_OPENAI = "api.openai.com"
SERVER_ADDRESS_BEDROCK = "bedrock-runtime.{region}.amazonaws.com"

# ── Content (opt-in — may contain sensitive data) ────────────────────────────
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"

# ── Evaluation ───────────────────────────────────────────────────────────────
GEN_AI_EVAL_NAME = "gen_ai.evaluation.name"
GEN_AI_EVAL_SCORE_VALUE = "gen_ai.evaluation.score.value"
GEN_AI_EVAL_SCORE_LABEL = "gen_ai.evaluation.score.label"
GEN_AI_EVAL_EXPLANATION = "gen_ai.evaluation.explanation"

# ── Event names ──────────────────────────────────────────────────────────────
GEN_AI_EVAL_RESULT_EVENT = "gen_ai.evaluation.result"
GEN_AI_INFERENCE_DETAILS_EVENT = "gen_ai.client.inference.operation.details"
GEN_AI_OPERATION_EXCEPTION_EVENT = "gen_ai.client.operation.exception"
GEN_AI_USER_MESSAGE_EVENT = "gen_ai.user.message"
GEN_AI_ASSISTANT_MESSAGE_EVENT = "gen_ai.assistant.message"
GEN_AI_TOOL_CALL_EVENT = "gen_ai.tool.call"
GEN_AI_SYSTEM_EVENT = "gen_ai.system"

# ── Metric names ──────────────────────────────────────────────────────────────
METRIC_TOKEN_USAGE = "gen_ai.client.token.usage"
METRIC_OPERATION_DURATION = "gen_ai.client.operation.duration"
METRIC_TIME_TO_FIRST_CHUNK = "gen_ai.client.operation.time_to_first_chunk"
METRIC_TIME_PER_OUTPUT_CHUNK = "gen_ai.client.operation.time_per_output_chunk"

# ── Operation name values ─────────────────────────────────────────────────────
OP_INVOKE_WORKFLOW = "invoke_workflow"
OP_INVOKE_AGENT = "invoke_agent"
OP_EXECUTE_TOOL = "execute_tool"
OP_CHAT = "chat"
OP_CREATE_AGENT = "create_agent"
OP_EMBEDDINGS = "embeddings"
OP_RETRIEVAL = "retrieval"

# ── aet-specific run attributes ───────────────────────────────────────────────
AET_SUITE = "aet.suite"
AET_METHOD = "aet.method"
AET_SEED = "aet.seed"
AET_TARGET = "aet.target"
AET_RUN_ID = "aet.run_id"
AET_RUN_ERRORS = "aet.run.total_errors"
AET_RUN_WARNINGS = "aet.run.total_warnings"

# ── aet-specific validator attributes ────────────────────────────────────────
AET_VALIDATOR_NAME = "aet.validator.name"

# ── aet-specific agent/cost attributes ───────────────────────────────────────
AET_AGENT_NUM_TURNS = "aet.agent.num_turns"
AET_AGENT_COST_USD = "aet.agent.cost_usd"
AET_AGENT_TOOL_CALL_COUNT = "aet.agent.tool_call_count"
AET_AGENT_TOOL_ERROR_COUNT = "aet.agent.tool_error_count"
AET_AGENT_PERMISSION_MODE = "aet.agent.permission_mode"
AET_AGENT_DURATION_MS = "aet.agent.duration_ms"
AET_AGENT_DURATION_API_MS = "aet.agent.duration_api_ms"

# ── aet-specific compilation/MLIR pass attributes ────────────────────────────
AET_COMPILATION_PASS = "aet.compilation.pass_name"
AET_COMPILATION_DIALECT_FROM = "aet.compilation.dialect_from"
AET_COMPILATION_DIALECT_TO = "aet.compilation.dialect_to"
