# aet — Agentic Eval Tool

Repo-agnostic harness to **run, record, and visualize** agentic coding runs (UC Berkeley BAR / SLICE).

`aet` turns an agent session (any Claude Code / `stream-json` transcript) into a canonical
[**trajectory**](trajectory.md) — cumulative tokens, cost, an activity timeline, and test-pass
milestones over time — then compares and plots many runs in a consistent house style. It also runs
agents in a [deny-by-default sandbox](isolation.md), survives the Claude 5-hour usage limit
unattended, and drives structured evaluation suites.

## Start here
- **[Trajectories](trajectory.md)** — record an agentic run and plot it.
- **[Architecture](ARCHITECTURE.md)** — the one-spine design.
- **[Isolation](isolation.md)** — the bwrap sandbox for untrusted agent runs.
- **[Decisions (ADRs)](adr/0001-transcript-oracle-mining.md)** — why the non-obvious calls were made.
- **API reference** — auto-generated from docstrings (see the nav).

Install: `pip install "aet[viz]"`. See the [README](https://github.com/ucb-bar/agentic-eval-tool)
for the full CLI, and [AGENTS.md](https://github.com/ucb-bar/agentic-eval-tool/blob/main/AGENTS.md)
to work *on* aet.
