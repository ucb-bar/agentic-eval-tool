# AGENT.md — skills/

## Purpose

Agent skill definitions for the targetgen evaluation harness. Each skill is an
instruction set that constrains what an LLM agent may do during a method run.

## What belongs here

- One sub-directory per skill, each with its own AGENT.md describing scope and constraints.
- Skills are read-only during method runs; they are not modified by the agent.

## Available skills

| Skill | Description |
|---|---|
| `gemmini_source_curator` | Curates the frozen source snapshot from the Gemmini repository |
| `dialect_planner` | Writes schema-constrained planning artifacts (YAML/JSON only) |
| `rtl_analyst` | Extracts hardware facts from RTL using grep/regex tools |
| `kernel_miner` | Mines abstraction candidates from kernel source files |
| `dialect_generator` | Deterministic code generator: reads dialect_plan.yaml, emits xDSL |

## Adding a new skill

1. Create a sub-directory with the skill name.
2. Add an `AGENT.md` describing: purpose, allowed actions, forbidden actions, outputs.
3. Reference the skill in the relevant `method.yaml` under `allowed_skills`.
