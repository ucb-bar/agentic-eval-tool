import subprocess
from pathlib import Path

def run_git(args: list[str], cwd: Path) -> str:
    """Run a git command. Returns stdout string or empty string on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""

def git_head(repo_root: Path) -> str:
    """Return current git HEAD hash, or 'unknown' on failure."""
    result = run_git(["rev-parse", "HEAD"], cwd=repo_root)
    return result if result else "unknown"

def git_diff_names(repo_root: Path, since_hash: str, path_filter: str = "") -> list[str]:
    """Return list of changed filenames since given hash."""
    args = ["diff", "--name-only", since_hash, "--"]
    if path_filter:
        args.append(path_filter)
    result = run_git(args, cwd=repo_root)
    if not result:
        return []
    return [line for line in result.splitlines() if line.strip()]
