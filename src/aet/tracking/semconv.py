"""OTel GenAI semantic convention attribute name constants.

Covers the GenAI semconv (status: Development, 2025) plus aet-specific extensions.
"""

from __future__ import annotations

# OTel GenAI semconv — operation
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

# OTel GenAI semconv — provider / model
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"

# OTel GenAI semconv — agentic
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_WORKFLOW_NAME = "gen_ai.workflow.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

# OTel GenAI semconv — token usage
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# OTel GenAI semconv — evaluation
GEN_AI_EVAL_NAME = "gen_ai.evaluation.name"
GEN_AI_EVAL_SCORE_VALUE = "gen_ai.evaluation.score.value"
GEN_AI_EVAL_SCORE_LABEL = "gen_ai.evaluation.score.label"

# OTel GenAI semconv — event names
GEN_AI_EVAL_RESULT_EVENT = "gen_ai.evaluation.result"

# aet-specific run attributes
AET_SUITE = "aet.suite"
AET_METHOD = "aet.method"
AET_SEED = "aet.seed"
AET_TARGET = "aet.target"
AET_RUN_ID = "aet.run_id"
AET_RUN_ERRORS = "aet.run.total_errors"
AET_RUN_WARNINGS = "aet.run.total_warnings"

# aet-specific validator attributes
AET_VALIDATOR_NAME = "aet.validator.name"

# aet-specific compilation/MLIR pass attributes
AET_COMPILATION_PASS = "aet.compilation.pass_name"
AET_COMPILATION_DIALECT_FROM = "aet.compilation.dialect_from"
AET_COMPILATION_DIALECT_TO = "aet.compilation.dialect_to"

# OTel GenAI operation.name values
OP_INVOKE_WORKFLOW = "invoke_workflow"
OP_INVOKE_AGENT = "invoke_agent"
OP_EXECUTE_TOOL = "execute_tool"
OP_CHAT = "chat"
