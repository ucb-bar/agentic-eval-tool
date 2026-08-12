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
  * ``unshare_net``  no network namespace at all — the filesystem allow-list means nothing if the agent
                     can fetch the same content over HTTP
  * ``clearenv``     start from an EMPTY environment and re-export only ``env_allow``, rather than
                     inheriting the launcher's and subtracting known-bad names
  * ``mask_git``     tmpfs over ``.git`` directories, so history cannot serve content the working tree
                     no longer has

Nothing here is project-specific: the caller supplies the paths/patterns. Gemmini/Merlin specifics (which
dirs are tools vs answers, the toolchain locations) live in the caller's config, not here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SYSTEM_DIRS = ["/usr", "/bin", "/lib", "/lib64", "/etc"]
DNS_DIR = "/run/systemd/resolve"   # /etc/resolv.conf -> here on systemd-resolved hosts

#: Enough to run a shell and a toolchain, and nothing that identifies the launcher or carries a
#: credential. The caller adds what its tools need; the default is deliberately austere, because the
#: point of ``clearenv`` is that a variable arrives only if someone named it.
DEFAULT_ENV_ALLOW = ["PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL", "USER"]

#: How deep to look for nested ``.git`` (submodules) under each granted root. Bounded on purpose: an
#: unbounded rglob over a granted toolchain tree is slow enough to be noticed, and the deep case is
#: rare enough that a caller who has one can pass it in ``deny``.
_GIT_SCAN_DEPTH = 3


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
    unshare_net: bool = False                                # no network namespace (see note below)
    clearenv: bool = False                                   # empty env, then re-export `env_allow`
    env_allow: list[str] | None = None                       # None -> DEFAULT_ENV_ALLOW when clearenv
    mask_git: bool = False                                   # tmpfs over .git under granted roots


def _git_dirs(roots: list[Path]) -> list[Path]:
    """``.git`` directories at or under each granted root, bounded to ``_GIT_SCAN_DEPTH``.

    Masking these matters whenever a granted tree is a checkout: withholding a file from the working
    tree while leaving its history readable withholds nothing. Deleting ``.git`` would work too, but
    it mutates the caller's tree — a tmpfs mask is reversible and leaves the directory present, so a
    tool that stats it does not take a different branch than it would outside the sandbox."""
    found: list[Path] = []
    for root in roots:
        try:
            if not root.is_dir():
                continue
        except PermissionError:
            continue
        candidates = [root / ".git"]
        for depth in range(1, _GIT_SCAN_DEPTH + 1):
            candidates += list(root.glob("/".join(["*"] * depth) + "/.git"))
        for c in candidates:
            try:
                if c.exists() and c not in found:
                    found.append(c)
            except PermissionError:
                continue
    return found


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
    if spec.unshare_net:
        # A filesystem allow-list is not an information boundary on its own: an agent denied a file
        # can often fetch the same bytes from the network. Off by default because most callers want
        # a working package index; on for anything measured.
        parts += ["--unshare-net"]
    for d in spec.system_dirs:
        if Path(d).exists():
            parts += ["--ro-bind", d, d]
    for d in spec.tmpfs:
        parts += ["--tmpfs", d]
    parts += ["--bind", str(spec.workspace), str(spec.workspace)]
    parts += ["--proc", "/proc", "--dev", "/dev", "--chdir", str(spec.workspace)]
    # Binding the resolver stub into a netns with no interfaces resolves nothing; skip it rather than
    # leave a bind that suggests the sandbox has networking when it does not.
    if spec.dns and not spec.unshare_net and Path(DNS_DIR).exists():
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
    # .git masks sit with the other deny masks: after every allow bind, so a granted repo root cannot
    # re-expose the history a broader allow just bound.
    if spec.mask_git:
        roots = [Path(spec.workspace), *map(Path, spec.allow),
                 *map(Path, spec.extra_binds), *map(Path, spec.rw_binds)]
        for g in _git_dirs(roots):
            parts += ["--tmpfs", str(g)]
    for f in spec.mask_files:
        if _kind(Path(f)) in ("file", "dir"):
            parts += ["--ro-bind", "/dev/null", str(f)]   # present-but-empty -> content withheld
    # Env comes last so nothing above can be undone by it. `--clearenv` starts from empty and
    # re-exports only what was named, which is the opposite posture from `--unsetenv`: the latter
    # requires you to have anticipated every bad variable, the former requires you to have
    # anticipated every needed one. Only the second failure mode is loud.
    if spec.clearenv:
        parts += ["--clearenv"]
        for name in (spec.env_allow if spec.env_allow is not None else DEFAULT_ENV_ALLOW):
            val = os.environ.get(name)
            if val is not None:
                parts += ["--setenv", name, val]
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
