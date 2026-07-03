# ADR-0002 — Session-log cost is a flagged provisional estimate

**Status:** accepted

## Context
`aet` ingests two on-disk shapes of a Claude Code run:
- **CLI `stream-json`** — ends with a `result` event carrying the authoritative `total_cost_usd`.
- **Desktop/app session log** — has per-turn `message.usage` + timestamps, but **no `result` event**,
  so no billed number.

## Decision
When a transcript has a terminal `result`, the trajectory's cost is the **billed** number. When it
does not, cost is a **list-price estimate** and every point is flagged `provisional_cost` (and the
trajectory `provisional`). Figures render provisional cost with a `~$` prefix.

## Why
- Never silently present an estimate as a billed number.
- Still lets desktop-recovered sessions produce faithful token/activity/rate figures — only the cost
  axis is an estimate, clearly marked.

## Consequences
- Consumers must check `traj.provisional` before treating cost as authoritative.
- A single file may concatenate several invocations (several `result` events); the importer splits at
  each so each invocation's billed cost is counted (not just the last).
