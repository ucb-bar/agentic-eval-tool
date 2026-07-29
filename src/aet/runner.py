"""Minimal sandboxed agent runner — the vehicle that closes *sandbox-run → record → plot*.

``run_agent`` launches a single Claude Code invocation (``claude --print --output-format
stream-json``) inside a deny-by-default :mod:`aet.isolation` sandbox, streams its stdout to a
transcript while recording a live :class:`~aet.trajectory.model.RunTrajectory`, and on exit
materialises a canonical aet run (manifest + ``logs/`` + ``metrics/trajectory.json``) that
``aet runs``/``aet show``/``aet plot`` read directly.

It is **rate-limit resilient** (the key ask for unattended overnight runs): if an invocation comes
back rejected-with-no-work by the Claude five-hour usage window, it checkpoints, waits to the exact
reset (or polls every ~20 min up to a 5h20m cap when the reset epoch is missing), and **resumes the
same session** — never burning the attempt. On the *weekly* limit, or when the poll cap / wait budget
is exhausted, it stops honestly: writes ``UNFINISHED.md`` + sets manifest ``status:
rate_limited_unfinished`` with a ``resume_cmd`` so a person — or another session — can pick it up with
``aet run --resume <run>``.

The watchdog is **daemon-free** (checkpoint + relaunch/resume, not a background service) and every
external dependency (the spawn, ``sleep``, wall-clock ``now``) is injectable, so the full
wait→resume and weekly→unfinished paths are exercised in milliseconds in CI with no real ``claude``,
no ``bwrap``, and no 5-hour wait.
"""
from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from aet.isolation import SandboxSpec, wrap_command
from aet.ratelimit import (
    DEFAULT_POLL_S, FIVE_HOUR_CAP_S, RateLimitState,
    parse_rate_limit, seconds_until_reset,
)

STATUS_COMPLETED = "completed"
STATUS_UNFINISHED = "rate_limited_unfinished"
STATUS_ERROR = "error"


@dataclass
class AgentRunResult:
    run_dir: Path
    status: str
    session_id: str = ""
    rate_limit_waits: int = 0
    limit_type: str = ""
    resets_at: int | None = None
    trajectory_path: Path | None = None
    exit_code: int | None = None
    attempts: int = 0


# --------------------------------------------------------------------- command building
def _claude_inner(task_file: Path, model: str, *, resume_session: str = "",
                  extra_flags: tuple[str, ...] = ()) -> str:
    """The ``claude --print`` stream-json command (prompt piped from ``task_file`` on stdin)."""
    parts = ["claude", "--print", "--output-format", "stream-json", "--verbose", "--model", model]
    if resume_session:
        parts += ["--resume", resume_session]
    parts += list(extra_flags)
    return " ".join(shlex.quote(p) for p in parts) + f" < {shlex.quote(str(task_file))}"


def _wrap(inner: str, spec: SandboxSpec | None, env_prefix: str) -> str:
    """Wrap ``inner`` for the sandbox (or run it directly with an optional env prefix)."""
    if spec is None:
        body = f"{env_prefix} {inner}".strip()
        return f"bash -c {shlex.quote(body)}"
    return wrap_command(inner, spec, env_prefix=env_prefix)


# --------------------------------------------------------------------- streaming spawn
def _stream_invocation(command: str, transcript_path: Path,
                       on_line: Callable[[str], None] | None = None) -> int:
    """Run ``command`` (shell), tee each stdout line to ``transcript_path``, return the exit code."""
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    with open(transcript_path, "w") as tf:
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            tf.write(line)
            tf.flush()
            if on_line is not None:
                on_line(line)
        proc.stdout.close()
        return proc.wait()


# --------------------------------------------------------------------- watchdog
def _plan_wait(state: RateLimitState, *, now: float, poll_seconds: float) -> float:
    """How long to sleep before the next attempt: to the exact reset when known, else one poll tick
    (bounded so an unknown/stale reset still resumes within one interval)."""
    exact = seconds_until_reset(state, now)
    if exact is not None:
        return exact
    return poll_seconds


# --------------------------------------------------------------------- materialize
def _materialize(run_dir: Path, *, label: str, model: str,
                 circt: bool | None = None) -> "tuple[Path, str]":
    """Build the trajectory from the streamed round transcripts + write the canonical run.

    Returns ``(trajectory_json_path, session_id)``."""
    from aet.trajectory.importers.transcript import import_transcript
    from aet.trajectory.recording import materialize_run

    rounds = run_dir / "rounds"
    traj = import_transcript(rounds, label=label, circt=circt, run_id=label)
    traj.source = "aet-run"
    traj.model = traj.model or model
    materialize_run(traj, run_dir)   # writes run_manifest.yaml + logs/ + metrics/trajectory.json
    session_id = traj.rounds[-1].session_id if traj.rounds else ""
    return run_dir / "metrics" / "trajectory.json", session_id


def _set_status(run_dir: Path, status: str, **meta) -> None:
    """Patch the run's manifest ``status`` + resume metadata (best-effort)."""
    from aet.core.run_manifest import RunManifest
    mpath = run_dir / "run_manifest.yaml"
    if not mpath.is_file():
        return
    m = RunManifest.load(mpath)
    m.status = status
    m.metadata = {**(m.metadata or {}), **{k: v for k, v in meta.items() if v is not None}}
    m.dump(mpath)


def _write_unfinished(run_dir: Path, state: RateLimitState, session_id: str) -> None:
    resume = f"aet run --resume {run_dir}"
    reset = f"epoch {state.resets_at}" if state.resets_at else "unknown"
    (run_dir / "UNFINISHED.md").write_text(
        "# Run unfinished — rate limited\n\n"
        f"This run stopped against the **{state.limit_type or 'usage'}** limit before finishing.\n\n"
        f"- limit type: `{state.limit_type or 'unknown'}`\n"
        f"- resets at: {reset}\n"
        f"- session id: `{session_id or 'unknown'}`\n\n"
        "It was (almost certainly) the weekly limit, or the five-hour wait budget was exhausted. "
        "Resume it later — a person, or another Claude Code session — with:\n\n"
        f"```\n{resume}\n```\n")


# --------------------------------------------------------------------- the runner
def run_agent(task: str | Path, workspace: str | Path, *,
              into: str | Path | None = None,
              model: str = "claude-opus-4-8",
              label: str = "",
              sandbox: str = "bwrap",
              allow: list | None = None,
              deny: list | None = None,
              extra_binds: list | None = None,
              rw_binds: list | None = None,
              mask_files: list | None = None,
              unsetenv: list | None = None,
              env_prefix: str = "",
              allow_unsandboxed: bool = False,
              circt: bool | None = None,
              # watchdog
              poll_seconds: float = DEFAULT_POLL_S,
              cap_seconds: float = FIVE_HOUR_CAP_S,
              max_rate_limit_waits: int = 3,
              # injectables (tests / custom launchers)
              agent_cmd: str | None = None,
              resume_session: str = "",
              on_line: Callable[[str], None] | None = None,
              sleep: Callable[[float], None] = time.sleep,
              now: Callable[[], float] = time.time,
              inject_states: "list[RateLimitState] | None" = None) -> AgentRunResult:
    """Launch a sandboxed, recorded, rate-limit-resilient agent invocation.

    ``agent_cmd`` overrides the ``claude`` command (a shell command that emits stream-json to
    stdout) — used by custom launchers and by the dummy-agent test path. ``inject_states`` (tests)
    forces the per-attempt rate-limit verdict without parsing, so the watchdog is unit-testable.
    """
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    run_dir = Path(into).resolve() if into else workspace.parent / f"{workspace.name}_aetrun"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = label or run_dir.name

    # task text → a file the agent reads on stdin (kept alongside the run for provenance)
    task_text = Path(task).read_text() if (isinstance(task, (str, Path)) and Path(str(task)).is_file()) else str(task)
    task_file = run_dir / "TASK.md"
    task_file.write_text(task_text)

    # sandbox policy
    spec: SandboxSpec | None = None
    if sandbox == "bwrap":
        spec = SandboxSpec(
            workspace=workspace,
            allow=[Path(p) for p in (allow or [])],
            deny=[Path(p) for p in (deny or [])],
            extra_binds=[Path(p) for p in (extra_binds or [])],
            rw_binds=[Path(p) for p in (rw_binds or [])],
            mask_files=[Path(p) for p in (mask_files or [])],
            unsetenv=list(unsetenv or []),
        )
    elif sandbox == "none" and not (allow_unsandboxed or agent_cmd is not None):
        raise ValueError(
            "refusing to run unsandboxed: pass allow_unsandboxed=True (or --allow-unsandboxed) "
            "to run a real agent without a sandbox")

    waits = 0
    attempt = 0
    session = resume_session
    exit_code = None
    last_state = RateLimitState()
    elapsed_wait = 0.0

    while True:
        transcript = run_dir / "rounds" / f"round_{attempt:02d}.transcript.jsonl"
        if agent_cmd is not None:
            inner = agent_cmd
        else:
            inner = _claude_inner(task_file, model, resume_session=session)
        command = _wrap(inner, spec, env_prefix)
        exit_code = _stream_invocation(command, transcript, on_line)
        attempt += 1

        # rate-limit verdict for this attempt (injectable for tests)
        if inject_states:
            state = inject_states[min(attempt - 1, len(inject_states) - 1)]
        else:
            state = parse_rate_limit(transcript.read_text(errors="ignore").splitlines())
        last_state = state

        if not state.rejected:
            break   # completed (or errored) with real work — done

        # rejected with no work → this attempt was burned. Decide: wait+resume, or give up.
        session = _session_from(transcript) or session
        if state.is_weekly or waits >= max_rate_limit_waits or elapsed_wait >= cap_seconds:
            # weekly limit, or the five-hour wait budget / poll cap exhausted → leave a note
            traj_path, sid = _materialize(run_dir, label=run_id, model=model, circt=circt)
            session = session or sid
            _set_status(run_dir, STATUS_UNFINISHED, limit_type=state.limit_type,
                        resets_at=state.resets_at, session_id=session,
                        resume_cmd=f"aet run --resume {run_dir}")
            _write_unfinished(run_dir, state, session)
            return AgentRunResult(run_dir=run_dir, status=STATUS_UNFINISHED, session_id=session,
                                  rate_limit_waits=waits, limit_type=state.limit_type,
                                  resets_at=state.resets_at, trajectory_path=traj_path,
                                  exit_code=exit_code, attempts=attempt)

        # five-hour limit with budget left → checkpoint, wait to the reset (or one poll tick), resume
        _materialize(run_dir, label=run_id, model=model, circt=circt)
        _set_status(run_dir, "rate_limited_waiting", limit_type=state.limit_type,
                    resets_at=state.resets_at, session_id=session)
        wait_s = _plan_wait(state, now=now(), poll_seconds=poll_seconds)
        sleep(wait_s)
        elapsed_wait += wait_s
        waits += 1
        # loop: resume the same session

    traj_path, sid = _materialize(run_dir, label=run_id, model=model, circt=circt)
    _set_status(run_dir, STATUS_COMPLETED, session_id=session or sid)
    return AgentRunResult(run_dir=run_dir, status=STATUS_COMPLETED, session_id=session or sid,
                          rate_limit_waits=waits, limit_type=last_state.limit_type,
                          trajectory_path=traj_path, exit_code=exit_code, attempts=attempt)


def _session_from(transcript: Path) -> str:
    """The session_id from a transcript (for ``claude --resume``)."""
    import json
    for line in transcript.read_text(errors="ignore").splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        sid = ev.get("session_id") or (ev.get("message", {}) or {}).get("session_id")
        if sid:
            return str(sid)
    return ""


def resume_run(run_dir: str | Path, **kwargs) -> AgentRunResult:
    """Resume a previously rate-limited run from its recorded session, in place."""
    from aet.core.run_manifest import RunManifest
    run_dir = Path(run_dir).resolve()
    meta = {}
    mpath = run_dir / "run_manifest.yaml"
    if mpath.is_file():
        meta = RunManifest.load(mpath).metadata or {}
    task_file = run_dir / "TASK.md"
    workspace = kwargs.pop("workspace", None) or run_dir
    return run_agent(task_file if task_file.is_file() else "resume", workspace,
                     into=run_dir, resume_session=str(meta.get("session_id", "")),
                     label=run_dir.name, **kwargs)
