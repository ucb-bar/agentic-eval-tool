"""Complete file-access LEDGER — every file an agent touched and what it did with it.

Where ``audit`` is the pass/fail gate (does any access break the allow-list?), this is the exhaustive
record for human review: it walks a run's round transcripts and enumerates EVERY file-touching tool call —
reads (path + bytes returned), writes/edits (path + bytes written), and bash commands (the verbatim command
+ the path-like tokens it referenced). Each path is classified by a caller-supplied ``classify`` callback
(e.g. in_scope / out_of_scope / system), so the ledger and the audit can agree.

HONEST LIMITATION: this is TOOL-level capture. A path opened *inside* a subprocess the agent spawns is not
individually listed — only the bash command text + the paths visible in it. Syscall-level capture needs
strace/auditd, which this does not do; the sandbox is what actually prevents out-of-scope opens.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

# absolute path, or a relative path with a slash + an extension-ish tail
_PATH_RX = re.compile(r"(/[\w./+-]+|(?:[\w.+-]+/)+[\w.+-]+)")


def _results_by_id(records) -> dict:
    out = {}
    for o in records:
        for b in (o.get("message", {}) or {}).get("content", []) or []:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                c = b.get("content")
                s = c if isinstance(c, str) else (json.dumps(c) if c else "")
                out[b.get("tool_use_id", "")] = {"bytes": len(s), "head": s[:120]}
    return out


def file_access_ledger(run_dir: Path,
                       classify: Callable[[str], str] = lambda p: "unclassified",
                       rounds_subdir: str = "rounds",
                       transcript_glob: str = "round_*.transcript.jsonl") -> dict:
    """Enumerate every file-touching tool call in a run. ``classify(path)->str`` tags each path."""
    run_dir = Path(run_dir)
    rdir = run_dir / rounds_subdir
    records = []
    for tp in sorted(rdir.glob(transcript_glob)) if rdir.is_dir() else []:
        for ln in tp.read_text(errors="ignore").splitlines():
            try:
                records.append(json.loads(ln))
            except Exception:
                continue
    results = _results_by_id(records)
    events = []
    for o in records:
        if o.get("type") != "assistant":
            continue
        for b in (o.get("message", {}) or {}).get("content", []) or []:
            if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                continue
            name, inp = b.get("name"), (b.get("input", {}) or {})
            res = results.get(b.get("id", ""), {})
            if name == "Read":
                p = inp.get("file_path", "")
                events.append({"op": "read", "path": p, "scope": classify(p),
                               "result_bytes": res.get("bytes")})
            elif name in ("Write", "Edit", "MultiEdit"):
                p = inp.get("file_path", "")
                events.append({"op": name.lower(), "path": p, "scope": classify(p),
                               "wrote_bytes": len(inp.get("content", "") or inp.get("new_string", "") or "")})
            elif name == "Bash":
                cmd = inp.get("command", "")
                refs = []
                for t in dict.fromkeys(_PATH_RX.findall(cmd)):
                    if "/" in t and (t.startswith("/") or "." in t):
                        refs.append({"path": t, "scope": classify(t)})
                events.append({"op": "exec", "cmd": cmd[:240], "refs": refs,
                               "stdout_bytes": res.get("bytes")})
    reads = sorted({e["path"] for e in events if e.get("op") == "read" and e.get("path")})
    writes = sorted({e["path"] for e in events if e.get("op") in ("write", "edit", "multiedit") and e.get("path")})
    oos = [e for e in events
           if str(e.get("scope", "")).startswith("out") or
           any(str(r["scope"]).startswith("out") for r in e.get("refs", []))]
    return {"run_id": run_dir.name, "n_events": len(events),
            "files_read": reads, "files_written": writes,
            "n_bash": sum(1 for e in events if e.get("op") == "exec"),
            "out_of_scope_events": oos, "events": events}
