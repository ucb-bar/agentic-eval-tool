# ADR-0003 — Deny-by-default bubblewrap for run isolation, not Docker

**Status:** accepted

## Context
An agent given `bypassPermissions` can run arbitrary shell and read anything on disk — including the
reference implementation / golden vectors / answer keys that would defeat "implement from spec".
`--add-dir`-style flags only scope the agent's *edit* surface, not OS reads. We need real filesystem
isolation for untrusted "run-wild" agent runs.

## Decision
Use **bubblewrap (`bwrap`)** namespaces via `aet.isolation` — a deny-by-default allow-list: the agent
sees only granted inputs + tools; answers, sibling runs, and other projects are simply not bound.

## Why
- **Deny-by-default** — the agent gets *nothing* unless explicitly granted (`allow`/`extra_binds`);
  the reference is hidden by omission, not by a blocklist that can miss a path.
- **Lightweight** — no image build, no daemon; a per-run namespace over the existing filesystem.
  Setuid-root `bwrap` runs unprivileged.
- **Composable with recording** — the sandbox wraps the same `claude` command the runner streams, so
  isolation and trajectory-capture are one flow.
- A post-run `audit_run` + `file_access_ledger` provide an independent backstop (the sandbox can be
  disabled for debugging; the audit still flags out-of-scope access).

## Consequences
- Requires `bwrap` on the host (setuid or relaxed unprivileged-userns). Not a hard cross-platform
  container; that is an acceptable trade for research hosts.
- Callers supply the allow/deny paths — nothing project-specific lives in `aet.isolation`.
