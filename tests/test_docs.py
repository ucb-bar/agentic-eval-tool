"""Docs anti-drift: keep README/AGENTS in lockstep with the real CLI + a working quickstart.

These tests exist so documentation *cannot* silently drift from the code:
  * every CLI subcommand the README/AGENTS name must actually resolve;
  * every `aet <cmd>` used in a README code block must be a real subcommand;
  * the README's record→plot promise runs end to end.
If you rename/remove a command, one of these fails until the docs are updated.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text()
AGENTS = (REPO / "AGENTS.md").read_text()

# The commands the README/AGENTS promise. If the CLI drops one, docs must too.
DOCUMENTED = {
    "import", "plot", "plot-sessions", "run", "monitor",
    "init-project", "init-run", "validate", "run-suite", "compare", "baseline", "runs", "show",
}


def _aet(*args):
    return subprocess.run([sys.executable, "-m", "aet.cli.main", *args],
                          capture_output=True, text=True)


def test_top_help_lists_documented_commands():
    r = _aet("--help")
    assert r.returncode == 0
    missing = [c for c in DOCUMENTED if c not in r.stdout]
    assert not missing, f"CLI no longer exposes {missing} but README/AGENTS still name them"


def test_each_documented_command_resolves():
    for cmd in sorted(DOCUMENTED):
        r = _aet(cmd, "--help")
        assert r.returncode == 0, f"`aet {cmd} --help` failed:\n{r.stderr}"


def _code_block_commands(md: str) -> set[str]:
    blocks = re.findall(r"```(?:bash)?\n(.*?)```", md, re.S)
    cmds = set()
    for b in blocks:
        for m in re.findall(r"(?:^|\s)aet\s+([a-z][a-z-]+)", b):
            cmds.add(m)
    return cmds


def test_readme_code_blocks_use_real_commands():
    avail = _aet("--help").stdout
    used = _code_block_commands(README)
    assert used, "expected at least one `aet <cmd>` example in the README"
    unreal = [c for c in used if c not in avail]
    assert not unreal, f"README code blocks use non-existent commands: {unreal}"


def test_agents_recipes_point_at_real_files():
    # the "how to add X" recipes name real seams — catch a moved registry/module
    for rel in ["src/aet/trajectory/importers/__init__.py",
                "src/aet/viz/comparison.py",
                "src/aet/suites/__init__.py",
                "src/aet/cli/commands/trajectory.py"]:
        assert (REPO / rel).is_file(), f"AGENTS.md references {rel} which no longer exists"
    assert "IMPORTER_REGISTRY" in AGENTS and "get_suite" in AGENTS


def test_readme_quickstart_import_runs(tmp_path):
    # the README's record→plot promise actually works (core path, no [viz] needed)
    from aet.trajectory.importers.transcript import import_transcript
    import json
    events = [
        {"timestamp": "2026-06-20T16:00:00Z", "type": "system", "session_id": "s"},
        {"timestamp": "2026-06-20T16:00:00Z", "type": "assistant", "message": {
            "id": "m1", "model": "claude-opus-4-8", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 100, "output_tokens": 10,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}},
        {"timestamp": "2026-06-20T16:00:20Z", "type": "result", "subtype": "success",
         "total_cost_usd": 0.07, "duration_ms": 20000, "num_turns": 1, "session_id": "s"},
    ]
    f = tmp_path / "transcript.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    traj = import_transcript(f, run_id="run-a")
    assert traj.num_rounds == 1
    assert abs(traj.final_cost_usd - 0.07) < 1e-9 and not traj.provisional


def test_readme_plot_promise(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from aet.trajectory.model import RunTrajectory, TrajectoryPoint
    from aet.viz.comparison import plot_cost_vs_time
    t = RunTrajectory(run_id="a", duration_s=120, num_rounds=1, final_cost_usd=1.0,
                      points=[TrajectoryPoint(t_s=0, cum_cost_usd=0.0),
                              TrajectoryPoint(t_s=120, cum_cost_usd=1.0)])
    out = tmp_path / "cost.png"
    plot_cost_vs_time([t], ["a"]).savefig(out)
    assert out.stat().st_size > 2000
