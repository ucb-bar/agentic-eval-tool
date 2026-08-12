"""Content hashes for immutable run inputs.

``RunSpec`` has declared ``spec_version_hash`` / ``tb_version_hash`` / ``hidden_eval_version`` since
it was written, and nothing ever populated them. Declared-and-unread fields are worse than absent
ones: they read like a control while providing none, so a comparison across runs whose inputs
silently diverged looks exactly like one where they did not.

:func:`hash_inputs` produces the mapping, and :func:`assert_comparable` is the half that makes it a
control rather than decoration — it refuses to aggregate runs whose inputs disagree.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

_CHUNK = 1 << 20   # 1 MiB

#: Never contributes to a hash: churn that does not change what a run reads.
DEFAULT_EXCLUDE = frozenset({
    ".git", ".hg", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "node_modules", ".DS_Store",
})


def sha256_file(path: Path) -> str | None:
    """Streaming sha256, or ``None`` when the file is absent.

    Streams rather than ``read_bytes()`` so hashing a multi-GB elaboration or liberty file does not
    load it into memory — the ASAP7 merged library alone is 46 MB, and a candidate RTL tree can be
    far larger.
    """
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(directory: Path, exclude: Iterable[str] = DEFAULT_EXCLUDE) -> str | None:
    """Deterministic sha256 over a tree: sorted relative paths, each with its content.

    Sorted rather than ``rglob`` order, which is filesystem-dependent — two machines hashing the
    same tree must agree, or ``assert_comparable`` starts rejecting runs for being on a different
    host. Paths are hashed alongside contents, so a rename is a change.
    """
    root = Path(directory)
    if not root.is_dir():
        return None
    skip = set(exclude)
    h = hashlib.sha256()
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip)
        for name in sorted(filenames):
            if name in skip:
                continue
            f = Path(dirpath) / name
            h.update(str(f.relative_to(root)).encode("utf-8"))
            h.update(b"\0")
            digest = sha256_file(f)
            h.update(bytes.fromhex(digest) if digest else b"<unreadable>")
            h.update(b"\0")
    return h.hexdigest()


def sha256_path(path: Path, exclude: Iterable[str] = DEFAULT_EXCLUDE) -> str | None:
    """File or directory, whichever it is. ``None`` when absent."""
    p = Path(path)
    if p.is_dir():
        return sha256_dir(p, exclude=exclude)
    return sha256_file(p)


def hash_inputs(inputs: Mapping[str, Path | str | None],
                exclude: Iterable[str] = DEFAULT_EXCLUDE) -> dict[str, str | None]:
    """``{name: sha256 | None}`` for every declared input.

    A missing input hashes to ``None`` and is KEPT in the mapping. Dropping the key would make a run
    that lacked an input compare equal to one that had it, which is the failure this whole module
    exists to prevent.
    """
    return {name: (None if p is None else sha256_path(Path(p), exclude=exclude))
            for name, p in inputs.items()}


def compare_inputs(a: Mapping[str, str | None], b: Mapping[str, str | None]) -> list[str]:
    """Names on which two input-hash mappings disagree, including presence."""
    out = []
    for name in sorted(set(a) | set(b)):
        if a.get(name) != b.get(name):
            out.append(name)
    return out


def assert_comparable(manifests: Mapping[str, Mapping[str, str | None]],
                      *, ignore: Iterable[str] = ()) -> None:
    """Raise unless every run in ``manifests`` was built from identical inputs.

    ``manifests`` maps run id -> its ``input_hashes``. ``ignore`` skips inputs that are *supposed*
    to differ — in an arm comparison the scaffold differs between arms by design, and listing it
    here is how that difference gets declared rather than assumed.

    This is the function that turns recorded hashes into a control. Without it they are a field
    somebody might read.
    """
    ids = list(manifests)
    if len(ids) < 2:
        return
    skip = set(ignore)
    base_id = ids[0]
    base = {k: v for k, v in manifests[base_id].items() if k not in skip}
    problems: list[str] = []
    for rid in ids[1:]:
        other = {k: v for k, v in manifests[rid].items() if k not in skip}
        diff = compare_inputs(base, other)
        if diff:
            problems.append(f"{base_id} vs {rid}: {', '.join(diff)}")
    if problems:
        raise InputsDiffer(
            "refusing to aggregate runs built from different inputs:\n  "
            + "\n  ".join(problems)
            + "\n(pass ignore=[...] for inputs that are supposed to differ, e.g. a per-arm scaffold)")


class InputsDiffer(RuntimeError):
    """Runs under comparison were not built from the same inputs."""
