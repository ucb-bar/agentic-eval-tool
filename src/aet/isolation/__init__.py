"""aet.isolation — reusable filesystem-isolation + integrity tooling for agentic runs.

Project-agnostic primitives extracted from the gemmini agentic A/B harness:

  * sandbox.SandboxSpec / bwrap_argv / wrap_command — deny-by-default bwrap allow-list (the agent sees
    only granted files + tools; answers, other agents' work, and other projects are masked).
  * audit.AuditPolicy / audit_run — post-run allow-list check over the transcripts (hard cheats vs soft
    out-of-scope vs review-warnings), an independent backstop to the sandbox.
  * ledger.file_access_ledger — exhaustive "every file touched + what was done" record for review.

The caller supplies all paths/patterns; nothing target- or experiment-specific lives here.
"""
from __future__ import annotations

from .sandbox import SandboxSpec, bwrap_argv, wrap_command  # noqa: F401
from .audit import AuditPolicy, audit_run                   # noqa: F401
from .ledger import file_access_ledger                      # noqa: F401
