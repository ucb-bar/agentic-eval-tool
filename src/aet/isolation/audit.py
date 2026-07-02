"""Post-run transcript audit — an *allow-list* check that the agent only touched what it was granted.

Under a filesystem sandbox the agent *can't* reach out-of-scope paths. But sandboxes can be bypassed,
disabled for debugging, or simply not used — so this independent, post-hoc check reads the agent's round
transcripts (the recorded tool calls) and flags any access outside its granted scope. Three severities,
all caller-configured (no project specifics baked in):

  * cheats       — HARD: reading an answer (golden, oracle, prior solution, a foreign implementation's
                   source). Any hit sets ``disqualified``.
  * contaminants — SOFT: any other out-of-scope path (a foreign project tree, another agent's run dir,
                   the experimenter's notes). Sets ``isolation_clean=False``; surfaced for review.
  * warns        — borderline, for human review only (e.g. reading an oracle's source without using it).

The caller passes regex patterns describing *its* answer/foreign surfaces. The transcript format is the
common Claude Code stream-json: ``{"type":"assistant","message":{"content":[{"type":"tool_use",...}]}}``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AuditPolicy:
    cheats: dict[str, re.Pattern] = field(default_factory=dict)        # hard -> disqualified
    contaminants: dict[str, re.Pattern] = field(default_factory=dict)  # soft -> isolation not clean
    warns: dict[str, re.Pattern] = field(default_factory=dict)         # review only
    transcript_glob: str = "round_*.transcript.jsonl"
    rounds_subdir: str = "rounds"


def _iter_tool_inputs(run_dir: Path, policy: AuditPolicy):
    """Yield the JSON-serialized input of every assistant tool_use across the run's round transcripts."""
    rdir = run_dir / policy.rounds_subdir
    for tp in sorted(rdir.glob(policy.transcript_glob)) if rdir.is_dir() else []:
        for ln in tp.read_text(errors="ignore").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get("type") != "assistant":
                continue
            for b in (o.get("message", {}) or {}).get("content", []) or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    yield b.get("name", "?"), json.dumps(b.get("input", {}) or {})


def audit_run(run_dir: Path, policy: AuditPolicy) -> dict:
    """Scan one run's transcripts. -> dict with cheat_hits / contaminants / warnings + verdict flags."""
    run_dir = Path(run_dir)
    cheats = {k: set() for k in policy.cheats}
    contam = {k: set() for k in policy.contaminants}
    warns = {k: 0 for k in policy.warns}
    tools: dict[str, int] = {}
    for name, blob in _iter_tool_inputs(run_dir, policy):
        tools[name] = tools.get(name, 0) + 1
        for k, rx in policy.cheats.items():
            for m in rx.findall(blob):
                cheats[k].add((m if isinstance(m, str) else m[0])[:120])
        for k, rx in policy.contaminants.items():
            for m in rx.findall(blob):
                contam[k].add((m if isinstance(m, str) else m[0])[:120])
        for k, rx in policy.warns.items():
            if rx.search(blob):
                warns[k] += 1
    cheats = {k: sorted(v) for k, v in cheats.items() if v}
    contam = {k: sorted(v) for k, v in contam.items() if v}
    warns = {k: n for k, n in warns.items() if n}
    return {
        "run_id": run_dir.name,
        "tools": tools,
        "cheat_hits": cheats,
        "disqualified": bool(cheats),
        "out_of_scope_reads": contam,
        "isolation_clean": not contam,
        "warnings": warns,
    }
