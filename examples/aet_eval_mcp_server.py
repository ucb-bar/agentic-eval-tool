"""Minimal MCP server for the aet deep-trace experiment.

Exposes two tools the agent can call during the experiment:
  - list_eval_runs   — list run directories under <project_root>/runs/
  - get_run_metrics  — return the last-N metrics from a run's metrics.jsonl

Run as a subprocess by Claude Code via --mcp-config; communicates over stdio.
"""
from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("aet-eval")


@mcp.tool()
def list_eval_runs(project_root: str = "/tmp/aet-deep-trace") -> str:
    """List all aet eval runs stored under <project_root>/runs/.

    Returns a JSON object with keys:
      runs  — list of run directory names
      total — count
      root  — the resolved project_root that was searched
    """
    runs_dir = Path(project_root) / "runs"
    if not runs_dir.exists():
        # Also try the path itself as the runs dir directly
        direct = Path(project_root)
        if direct.exists() and direct.is_dir():
            run_names = [
                d.name for d in sorted(direct.iterdir())
                if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
            ]
            return json.dumps({"runs": run_names, "total": len(run_names), "root": str(direct)})
        return json.dumps({"runs": [], "total": 0, "root": str(runs_dir),
                           "note": "directory not found"})
    run_names = [
        d.name for d in sorted(runs_dir.iterdir())
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]
    return json.dumps({"runs": run_names, "total": len(run_names), "root": str(runs_dir)})


@mcp.tool()
def get_run_metrics(run_id: str, project_root: str = "/tmp/aet-deep-trace") -> str:
    """Return the recorded metrics for an aet eval run.

    Searches for <project_root>/runs/<run_id>/logs/metrics.jsonl or
    <project_root>/<run_id>/logs/metrics.jsonl.

    Returns a JSON object with:
      run_id  — the requested run id
      metrics — list of metric entries (last 30)
      total   — total number of metric entries
    """
    candidates = [
        Path(project_root) / "runs" / run_id / "logs" / "metrics.jsonl",
        Path(project_root) / run_id / "logs" / "metrics.jsonl",
        Path(project_root) / "logs" / "metrics.jsonl",  # project_root IS the run dir
    ]
    for mf in candidates:
        if mf.exists():
            entries = [
                json.loads(line) for line in mf.read_text().splitlines() if line.strip()
            ]
            return json.dumps({
                "run_id": run_id,
                "metrics": entries[-30:],
                "total": len(entries),
                "source": str(mf),
            })
    return json.dumps({
        "run_id": run_id,
        "error": f"metrics.jsonl not found (searched {[str(c) for c in candidates]})",
    })


if __name__ == "__main__":
    mcp.run()
