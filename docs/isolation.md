# Isolation & Integrity (`aet.isolation`)

Reusable filesystem-isolation and post-run integrity tooling for agentic code-generation experiments. The
goal: an agent should start with **nothing but** the files you grant it plus the tools it needs, must not
read anything else on the machine (answer keys, other agents' work, other projects), and you should be able
to **prove** afterwards that it didn't.

These primitives are project-agnostic — you supply the paths and patterns; nothing target-specific lives in
the module.

## 1. Sandbox — a deny-by-default filesystem view (`SandboxSpec`, `bwrap_argv`, `wrap_command`)

Built on bubblewrap (`bwrap`). You declare an allow-list; everything else is masked.

```python
from pathlib import Path
from aet.isolation import SandboxSpec, wrap_command

spec = SandboxSpec(
    workspace=Path("/run/ws"),                 # the ONLY writable dir (the agent's cwd)
    allow=[Path("/repo/inputs"), Path("/repo/tools")],   # ro-bind: granted inputs + in-repo tools
    deny=[Path("/repo/inputs/answers")],       # tmpfs-mask AFTER allow (deny wins over a broad allow)
    mask_files=[Path("/repo/inputs/golden.yaml")],       # /dev/null overlay (per-FILE answer)
    extra_binds=[Path("/opt/toolchain")],      # ro-bind tools that live OUTSIDE the repo
    tmpfs=["/tmp", "/scratch"],                # blank these (e.g. the parent of sibling runs)
    unsetenv=["CLAUDE_CODE_SSE_PORT"],         # clear vars that would mis-route a child agent
    dns=True,                                  # bind /run/systemd/resolve so networking resolves
)
cmd = wrap_command("python build.py && ./run", spec, env_prefix="export PATH=/opt/toolchain/bin:$PATH;")
subprocess.run(["bash", "-c", cmd])
```

Notes learned in production:
- **Order matters** — `deny`/`mask_files` are applied *after* `allow`/`extra_binds`, so a broad allow can't
  re-expose a denied sub-path.
- **DNS** — `/etc/resolv.conf` usually symlinks into `/run`, which bwrap doesn't bind; `dns=True` binds the
  resolver stub so an in-sandbox process can reach the network.
- **Nested sessions** — when launching a child agent from inside another agent session, clear the parent's
  session env vars (`unsetenv`) or the child mis-routes through a dead local relay.
- **Permission-safe** — a `chmod 000` lock on an answer dir won't crash the builder; it's still masked.

## 2. Audit — post-run allow-list check (`AuditPolicy`, `audit_run`)

Independent backstop to the sandbox: reads the agent's round transcripts and flags any out-of-scope access.

```python
import re
from aet.isolation import AuditPolicy, audit_run

policy = AuditPolicy(
    cheats={"golden": re.compile(r"golden\.yaml"),            # HARD -> disqualified
            "oracle": re.compile(r"import\s+solver\.oracle")},
    contaminants={"other_proj": re.compile(r"/other/projects/[^ \"']+")},  # SOFT -> isolation not clean
    warns={"oracle_src": re.compile(r"oracle\.py")},          # review only
)
r = audit_run(Path("runs/agent_01"), policy)
# r["disqualified"], r["isolation_clean"], r["cheat_hits"], r["out_of_scope_reads"], r["warnings"]
```

Severities: **cheats** = reading an answer/foreign source (any hit disqualifies); **contaminants** = any
other out-of-scope path (flagged, not auto-fatal); **warns** = borderline, for human review.

## 3. Ledger — exhaustive file-access record (`file_access_ledger`)

Every file the agent touched and what it did with it (reads + bytes, writes + bytes, bash commands + the
paths they referenced), each classified by your callback.

```python
from aet.isolation import file_access_ledger
L = file_access_ledger(Path("runs/agent_01"),
                       classify=lambda p: "out_of_scope" if p.startswith("/other") else "in_scope")
# L["files_read"], L["files_written"], L["n_bash"], L["out_of_scope_events"], L["events"]
```

**Limitation:** tool-level capture only — a path opened *inside* a subprocess the agent spawns is not
individually listed (only the bash command text + visible paths). Syscall-level capture needs
strace/auditd; the **sandbox** is what actually prevents out-of-scope opens, the audit/ledger are the
record and the backstop.
```
