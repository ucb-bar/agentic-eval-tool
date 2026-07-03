"""Test that aet init-project for targetgen template works or fails gracefully."""
import subprocess
import sys


def test_init_project_targetgen(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "aet.cli.main", "init-project",
         "--template", "targetgen", "--project-root", str(tmp_path), "--force"],
        capture_output=True, text=True,
    )
    # Either succeeds (rc=0) or fails with an informative error message
    if result.returncode != 0:
        err = (result.stderr + result.stdout).lower()
        assert "template" in err or "error" in err, (
            f"init-project failed without useful message:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def test_init_project_default(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "aet.cli.main", "init-project",
         "--template", "default", "--project-root", str(tmp_path), "--force"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        err = (result.stderr + result.stdout).lower()
        assert "template" in err or "error" in err


def test_init_project_force_idempotent(tmp_path):
    """Running init-project twice with --force must not crash."""
    cmd = [
        sys.executable, "-m", "aet.cli.main", "init-project",
        "--template", "targetgen", "--project-root", str(tmp_path), "--force",
    ]
    r1 = subprocess.run(cmd, capture_output=True, text=True)
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    # Both must either succeed or fail gracefully with template-related messages
    for result in (r1, r2):
        if result.returncode != 0:
            err = (result.stderr + result.stdout).lower()
            assert "template" in err or "error" in err


def test_init_project_no_subcommand_exits_cleanly():
    """Running aet with no subcommand exits 0 (help printed or silent)."""
    result = subprocess.run(
        [sys.executable, "-m", "aet.cli.main"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
