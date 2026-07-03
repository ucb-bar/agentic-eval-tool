# ADR-0001 — Mine the tests-passing climb from the transcript, not a harness hook

**Status:** accepted

## Context
The presentation "tests passing over time" figure needs a *progression* (e.g. 0/182 → 182/182 across
a run), not just a terminal pass/fail. In oscar-merlin that climb came from a dedicated periodic
grader that appended a `selfcheck_log.jsonl`. abc-testing (and most single-shot tasks) grades **once
at the end** — there is no periodic log.

Two ways to recover the climb:
1. **Hook the oracle** (`run_test.py`) to append `{wall, n_passed, n_total}` each time it runs.
2. **Mine the transcript** — the agent already runs the testbench (`./run.sh`) repeatedly, and each
   result is captured in the transcript as a tool result.

## Decision
Mine the transcript (`aet.trajectory.oracle.extract_oracle_progression`), gated by
`import_transcript(oracle_markers=…)`.

## Why
- **No harness change** — nothing to add to the shared `run_test.py`; it stays a pure oracle.
- **Retroactive** — works on runs that already happened.
- Robust enough: parse the testbench summary (`*** PASSED *** … N cases`), key each reading on the
  tool call's wall time, and **reject a reading whose suite size ≠ the known one** (the agent's own
  sub-module unit tests report different counts). Marker defaults to `./run.sh` (the task's oracle
  entrypoint), deliberately not bare `verilator`/`run_test.py`.

## Consequences
- The climb is only as granular as the agent's own oracle runs (usually fail→…→pass steps, not a
  smooth k/N curve unless the testbench reports partial counts). That is honest and sufficient.
- If a future task suppresses oracle output, a structured `run_test.py` log (option 1) is the
  fallback — left as a documented follow-up.
