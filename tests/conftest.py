import pytest
import subprocess
import sys


@pytest.fixture
def tmp_project(tmp_path):
    """Initialize a default aet project in a temp dir."""
    return tmp_path


@pytest.fixture
def tmp_targetgen_project(tmp_path):
    """Initialize a targetgen project via aet init-project."""
    subprocess.run(
        [sys.executable, "-m", "aet.cli.main", "init-project",
         "--template", "targetgen", "--project-root", str(tmp_path), "--force"],
        capture_output=True, text=True
    )
    return tmp_path
