import shutil
from pathlib import Path

def copy_template(src_dir: Path, dst_dir: Path, force: bool = False) -> list[Path]:
    """Copy template directory tree to dst_dir. Skip existing files unless force=True.
    Returns list of created paths."""
    created = []
    for src_file in src_dir.rglob("*"):
        if src_file.is_dir():
            continue
        rel = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel
        if dst_file.exists() and not force:
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        created.append(dst_file)
    return created

def ensure_gitkeep(directory: Path) -> None:
    """Create directory with a .gitkeep if it doesn't exist."""
    directory.mkdir(parents=True, exist_ok=True)
    gitkeep = directory / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("")
