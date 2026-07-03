"""Deny-by-default filesystem sandbox for agentic runs, built on bubblewrap (``bwrap``).

The model: an agent should start with **nothing but** the files you grant it + the tools it needs, and
must not be able to read anything else on the machine (answer keys, other agents' work, other projects).
This builds a ``bwrap`` argv that gives exactly that — an *allow-list* filesystem view:

  * ``system_dirs``  bound read-only (the OS: /usr, /bin, /lib, ...)
  * ``workspace``    bound read-WRITE (the agent's cwd — the one place it may write)
  * ``allow``        bound read-only (granted inputs + tools)
  * ``extra_binds``  bound read-only (toolchain dirs that live OUTSIDE the repo, re-added over tmpfs masks)
  * ``tmpfs``        blanked (e.g. /tmp, and the project parent so siblings disappear)
  * ``deny``         tmpfs-masked AFTER the allow binds (deny-wins: a broad allow can't expose a denied
                     sub-path) — for answer dirs, other agents' run dirs, prior solutions
  * ``mask_files``   overlaid with /dev/null (per-FILE masking — e.g. individual golden files inside an
                     otherwise-granted corpus dir)
  * ``unsetenv``     cleared in the sandbox (e.g. nested-session vars that would mis-route a child agent)
  * ``dns``          bind /run/systemd/resolve so name resolution works (``/etc/resolv.conf`` often
                     symlinks into /run, which bwrap does not bind by default)

Nothing here is project-specific: the caller supplies the paths/patterns. Gemmini/Merlin specifics (which
dirs are tools vs answers, the toolchain locations) live in the caller's config, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SYSTEM_DIRS = ["/usr", "/bin", "/lib", "/lib64", "/etc"]
DNS_DIR = "/run/systemd/resolve"   # /etc/resolv.conf -> here on systemd-resolved hosts


@dataclass
class SandboxSpec:
    workspace: Path                                          # the only writable dir (agent cwd)
    allow: list[Path] = field(default_factory=list)          # ro-bind (granted inputs + in-repo tools)
    rw_binds: list[Path] = field(default_factory=list)       # read-WRITE bind (agent state dirs outside
                                                             #   the workspace, e.g. a CLI's ~/.claude
                                                             #   session/cache under an otherwise-ro home)
    deny: list[Path] = field(default_factory=list)           # tmpfs-mask (answers / other agents' work)
    mask_files: list[Path] = field(default_factory=list)     # /dev/null overlay (per-file answers)
    extra_binds: list[Path] = field(default_factory=list)    # ro-bind toolchain dirs outside the repo
    system_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_SYSTEM_DIRS))
    tmpfs: list[str] = field(default_factory=lambda: ["/tmp"])  # blanked dirs (parents of siblings, /tmp)
    unsetenv: list[str] = field(default_factory=list)        # env vars to clear inside the sandbox
    dns: bool = True                                         # bind the resolver stub for networking
    die_with_parent: bool = True
    unshare_pid: bool = True


def _kind(p: Path) -> str:
    """Permission-safe path classification: a chmod-000 lock on a parent makes stat() raise — treat that
    as a present dir so a locked answer surface is still masked, and the builder never crashes."""
    try:
        if p.is_dir():
            return "dir"
        if p.exists():
            return "file"
        return "missing"
    except PermissionError:
        return "dir"


def bwrap_argv(spec: SandboxSpec) -> list[str]:
    """Build the deny-by-default ``bwrap`` argv prefix (everything up to, but not including, the command).

    Order matters: system + tmpfs masks first, then the workspace + allow binds, then deny masks (so deny
    wins over a broader allow), then per-file overlays, then extra toolchain binds, then env unsets."""
    parts: list[str] = ["bwrap"]
    if spec.die_with_parent:
        parts += ["--die-with-parent"]
    if spec.unshare_pid:
        parts += ["--unshare-pid"]
    for d in spec.system_dirs:
        if Path(d).exists():
            parts += ["--ro-bind", d, d]
    for d in spec.tmpfs:
        parts += ["--tmpfs", d]
    parts += ["--bind", str(spec.workspace), str(spec.workspace)]
    parts += ["--proc", "/proc", "--dev", "/dev", "--chdir", str(spec.workspace)]
    if spec.dns and Path(DNS_DIR).exists():
        parts += ["--ro-bind", DNS_DIR, DNS_DIR]
    for p in spec.allow:
        if _kind(Path(p)) != "missing":
            parts += ["--ro-bind", str(p), str(p)]
    for p in spec.extra_binds:
        if _kind(Path(p)) != "missing":
            parts += ["--ro-bind", str(p), str(p)]
    # rw binds AFTER the ro allow/extra binds so a writable state dir overrides a broad ro allow
    # (e.g. ~/.claude writable under an otherwise read-only $HOME), but BEFORE the deny masks below.
    for p in spec.rw_binds:
        if _kind(Path(p)) != "missing":
            parts += ["--bind", str(p), str(p)]
    # deny wins: mask AFTER allow/extra binds
    for p in spec.deny:
        if _kind(Path(p)) == "dir":
            parts += ["--tmpfs", str(p)]
    for f in spec.mask_files:
        if _kind(Path(f)) in ("file", "dir"):
            parts += ["--ro-bind", "/dev/null", str(f)]   # present-but-empty -> content withheld
    for v in spec.unsetenv:
        parts += ["--unsetenv", v]
    return parts


def wrap_command(inner: str, spec: SandboxSpec, env_prefix: str = "") -> str:
    """Wrap a shell command to run inside the sandbox: ``bwrap ... bash -c '<env_prefix> <inner>'``.

    ``env_prefix`` is shell ``export``s for the toolchain (PATH/LD_LIBRARY_PATH/PYTHONPATH/...), kept by the
    caller because tool locations are project-specific. Returns a single string for ``bash -c``."""
    argv = " ".join(bwrap_argv(spec))
    body = f"{env_prefix} {inner}".strip()
    return f"{argv} bash -c '{body}'"
