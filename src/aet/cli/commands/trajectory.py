"""Trajectory + agentic commands: import, plot, plot-sessions, run, monitor."""
from __future__ import annotations

import sys
from pathlib import Path


from aet.cli._common import (
    _load_trajectory, _save_fig,
)

def _cmd_import(args) -> None:
    from aet.trajectory.classify import ActivityConfig
    from aet.trajectory.importers import get_importer

    # `--format` is the spec spelling; it aliases `--source` (the historical flag). Either works.
    source = getattr(args, "fmt", None) or args.source
    importer = get_importer(source)
    cfg = ActivityConfig.from_json_file(args.classifier_config) if args.classifier_config else None
    circt = None
    if getattr(args, "circt", None) is True:
        circt = True
    elif getattr(args, "no_circt", False):
        circt = False

    kwargs = dict(
        classifier_config=cfg,
        circt=circt,
        milestone_time=args.milestone_time,
        run_id=args.run_id or "",
    )
    # the Codex importer resolves its requested model against the OpenAI price snapshot
    if source == "codex":
        kwargs["model"] = getattr(args, "model", None) or "gpt-5-codex"
        if getattr(args, "price_snapshot", None):
            kwargs["price_snapshot"] = args.price_snapshot
        if getattr(args, "billing_mode", None):
            kwargs["billing_mode"] = args.billing_mode
        if getattr(args, "provider", None):
            kwargs["provider"] = args.provider
    # the generic transcript importer accepts an optional terminal pass/fail + label
    if source == "transcript":
        kwargs["label"] = getattr(args, "label", None)
        kwargs["n_total"] = getattr(args, "n_total", 1)
        pb = getattr(args, "pass_bool", None)
        if pb is not None:
            kwargs["pass_bool"] = pb
    # the full-fidelity OTel importer records a terminal verdict as n_passed / n_total
    elif source == "otel":
        n_total = getattr(args, "n_total", 1)
        kwargs["n_total"] = n_total
        pb = getattr(args, "pass_bool", None)
        if pb is not None:
            kwargs["n_passed"] = n_total if pb else 0

    traj = importer(args.raw, **kwargs)

    out = Path(args.out) if args.out else Path(args.raw) / "trajectory.json"
    traj.to_json(out)
    print(f"[aet] imported {source} run '{traj.run_id}': "
          f"{traj.num_rounds} rounds, {len(traj.points)} points, "
          f"{len(traj.milestones)} milestones, {len(traj.bands)} activity bands")
    _cost_str = "unpriced" if traj.final_cost_usd is None else f"${traj.final_cost_usd:.4f}"
    print(f"[aet]   final: {traj.final_input_tokens + traj.final_output_tokens + traj.final_cache_tokens:,} tokens, "
          f"{_cost_str}, {traj.duration_s / 60.0:.1f} min")
    if traj.milestones:
        prog = " → ".join(str(m.n_passed) for m in sorted(traj.milestones, key=lambda m: m.t_s))
        print(f"[aet]   test-pass milestones: {prog} / {traj.milestones[-1].n_total}")
    print(f"[aet] wrote {out}")

    if getattr(args, "into", None):
        from aet.trajectory.recording import materialize_run
        run_path = materialize_run(traj, Path(args.into))
        print(f"[aet] materialized aet run at {run_path}")


def _cmd_otel_sink(args) -> None:
    """Run the minimal OTLP/HTTP receiver that captures Claude Code telemetry to a JSONL file."""
    from aet.tracking.otel_sink import serve
    serve(args.port, args.out, host=getattr(args, "host", "127.0.0.1"))


def _cmd_plot(args) -> None:
    kind = getattr(args, "kind", "trajectory")
    try:
        if kind in ("trajectory", "comparison"):
            from aet.viz.trajectory_plot import plot_trajectory, plot_comparison
        else:
            from aet.viz.comparison import (
                plot_rate_panels, plot_cost_vs_time, plot_tests_facets,
            )
    except ImportError as e:
        print(f"[aet] {e}", file=sys.stderr)
        sys.exit(1)

    main_traj = _load_trajectory(Path(args.run))
    trajs = [main_traj] + [_load_trajectory(Path(p)) for p in (args.comparison or [])]
    labels = [t.run_id for t in trajs]

    # --split-cache is honoured by both the single-run figure and the rate panels, so the
    # "you asked for a split this run does not carry" warning is checked once for both.
    split_cache = getattr(args, "split_cache", False)
    if split_cache and kind in ("trajectory", "rate-panels"):
        if not any(p.cum_cache_read_tokens or p.cum_cache_creation_tokens
                   for t in trajs for p in t.points):
            # Two lines flat at zero look like a measurement of "no cache activity" rather than
            # like missing data, so say which it is instead of drawing it silently.
            print("[aet] warning: --split-cache requested but this trajectory carries no "
                  "cache read/creation split (both series are zero); the source recorded only "
                  "the cache total", file=sys.stderr)

    if kind == "rate-panels":
        fig = plot_rate_panels(trajs, labels, split_cache=split_cache)
    elif kind == "cost-vs-time":
        fig = plot_cost_vs_time(trajs, labels)
    elif kind == "tests-facets":
        fig = plot_tests_facets(trajs, labels)
    elif kind == "comparison" or (kind == "trajectory" and args.comparison):
        from aet.viz.trajectory_plot import plot_comparison
        fig = plot_comparison(trajs, log_tokens=not args.linear_tokens)
    else:
        fig = plot_trajectory(main_traj, log_tokens=not args.linear_tokens,
                              show_spend=not args.no_spend, split_cache=split_cache)

    out = Path(args.out) if args.out else Path(args.run).with_suffix(f".{kind}.png")
    for p in _save_fig(fig, out, args.dpi):
        print(f"[aet] wrote {p}")



_SESSION_KINDS = {
    "rate-panels": "plot_rate_panels",
    "cost-vs-time": "plot_cost_vs_time",
    "tests-facets": "plot_tests_facets",
}


def _cmd_plot_sessions(args) -> None:
    """Import one-or-many raw Claude Code sessions and render the comparison figures in one step.

    Each ``session`` is a transcript ``*.jsonl`` file or a directory holding session logs; it becomes
    one arm (labelled by its stem/dir name). No prior ``aet import`` needed — this is the
    'just point it at your sessions' path."""
    from aet.trajectory.importers.transcript import import_transcript
    try:
        from aet.viz import comparison as C
    except ImportError as e:
        print(f"[aet] {e}", file=sys.stderr)
        sys.exit(1)

    trajs, labels = [], []
    for s in args.sessions:
        p = Path(s)
        # label by the file stem, but fall back to the parent dir when the file has a generic
        # name (transcript.jsonl / session.jsonl) so per-run dirs stay distinguishable
        if p.is_file():
            label = p.parent.name if p.stem in ("transcript", "session", "stream") else p.stem
        else:
            label = p.name
        traj = import_transcript(p, run_id=label,
                                 pass_bool=(True if args.pass_all else None),
                                 n_total=args.n_total)
        if not traj.points:
            print(f"[aet] plot-sessions: no agent turns parsed from {p}; skipping", file=sys.stderr)
            continue
        trajs.append(traj)
        labels.append(label)
        tok = traj.final_input_tokens + traj.final_output_tokens + traj.final_cache_tokens
        cost = ("unpriced" if traj.final_cost_usd is None
                else ("~$" if traj.provisional else "$") + f"{traj.final_cost_usd:.2f}")
        print(f"[aet]   {label}: {tok / 1e6:.2f}M tok · {cost} · {traj.duration_s / 60.0:.1f} min")

    if not trajs:
        print("[aet] plot-sessions: no usable sessions found", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out) if args.out else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = args.kinds or list(_SESSION_KINDS)
    for kind in kinds:
        fn = getattr(C, _SESSION_KINDS[kind])
        fig = fn(trajs, labels)
        for pth in _save_fig(fig, out_dir / f"{kind}.png", args.dpi):
            print(f"[aet] wrote {pth}  ({len(trajs)} sessions)")


# ---------------------------------------------------------------------------
# run / resume — a sandboxed, recorded, rate-limit-resilient agent invocation
# ---------------------------------------------------------------------------

def _cmd_run(args) -> None:
    from aet.runner import run_agent, resume_run, STATUS_UNFINISHED

    common = dict(
        model=args.model,
        sandbox=args.sandbox,
        allow=args.allow or [],
        deny=args.deny or [],
        extra_binds=args.extra_binds or [],
        rw_binds=args.rw_binds or [],
        mask_files=args.mask_files or [],
        unsetenv=args.unsetenv or [],
        unshare_net=getattr(args, "unshare_net", False),
        env_prefix=args.env_prefix or "",
        allow_unsandboxed=args.allow_unsandboxed,
        agent_cmd=args.agent_cmd,
        poll_seconds=args.poll_seconds,
        max_rate_limit_waits=args.max_rate_limit_waits,
    )
    if getattr(args, "resume", None):
        res = resume_run(args.resume, **common)
    else:
        if not args.task or not args.workspace:
            print("[aet] Error: --task and --workspace are required (or use --resume <run>)",
                  file=sys.stderr)
            sys.exit(2)
        res = run_agent(args.task, args.workspace, into=args.into, label=args.label or "",
                        **common)

    print(f"[aet] run {res.status}: {res.run_dir}")
    print(f"[aet]   {res.attempts} attempt(s), {res.rate_limit_waits} rate-limit wait(s), "
          f"session={res.session_id or '—'}")
    if res.status == STATUS_UNFINISHED:
        print(f"[aet]   ⚠ rate-limited ({res.limit_type or 'usage'}); left UNFINISHED.md. "
              f"Resume with:  aet run --resume {res.run_dir}", file=sys.stderr)
        sys.exit(3)
    if res.trajectory_path:
        print(f"[aet]   plot it:  aet plot {res.run_dir} --kind trajectory")


# ---------------------------------------------------------------------------
# monitor — live activity view of an in-flight agent session
# ---------------------------------------------------------------------------

def _selfcheck_tests_passed(path) -> tuple[int, int] | None:
    """Best all-scope (n_passed, n_total) from a self-check log, for a live tests-passed readout."""
    p = Path(path)
    if not p.is_file():
        return None
    best = (0, 0)
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = __import__("json").loads(line)
        except Exception:
            continue
        if str(r.get("capsules")) == "all" and (r.get("n_capsules") or 0) >= 20:
            if (r.get("n_passed") or 0) >= best[0]:
                best = (r.get("n_passed") or 0, r.get("n_capsules") or 0)
    return best if best[1] else None


def _monitor_classifier(args):
    from aet.trajectory.classify import ActivityClassifier, ActivityConfig, capsule_bench_config
    if args.classifier_config:
        return ActivityClassifier(ActivityConfig.from_json_file(args.classifier_config))
    if args.preset == "capsule-bench":
        return ActivityClassifier(capsule_bench_config(circt=args.circt))
    return ActivityClassifier()


def _cmd_monitor(args) -> None:
    from aet.trajectory.stream import TrajectoryStream

    classifier = _monitor_classifier(args)
    selfcheck = args.selfcheck

    def _status(traj) -> str:
        cur = traj.bands[-1].category if traj.bands else "—"
        tok = traj.final_input_tokens + traj.final_output_tokens + traj.final_cache_tokens
        if traj.final_cost_usd is None:
            cost = "unpriced"
        else:
            cost = f"~${traj.final_cost_usd:.3f}(prov)" if traj.provisional else f"${traj.final_cost_usd:.3f}"
        tests = ""
        if selfcheck:
            tp = _selfcheck_tests_passed(selfcheck)
            if tp:
                tests = f" | tests {tp[0]}/{tp[1]}"
        return (f"[{traj.duration_s / 60.0:5.1f} min] {tok:>12,} tok | {cost:>16} "
                f"| now: {cur:<6}{tests}")

    last = {"traj": None}

    def _on_update(traj):
        last["traj"] = traj
        print("\r" + _status(traj).ljust(90), end="", flush=True)

    stream = TrajectoryStream(classifier=classifier, on_update=_on_update,
                              flush_every=args.flush_every, run_id=Path(args.attach).stem)
    traj = stream.attach_file(args.attach, poll_s=args.interval,
                              follow=not args.no_follow, max_seconds=args.max_seconds)
    print()  # end the rewriting status line
    print(f"[aet] {'streaming ended (result event)' if not traj.provisional else 'stopped (still in flight)'}: "
          f"{traj.num_rounds} round(s), {len(traj.points)} points, "
          f"{traj.final_input_tokens + traj.final_output_tokens + traj.final_cache_tokens:,} tok, "
          f"{'unpriced' if traj.final_cost_usd is None else ('~$' if traj.provisional else '$') + format(traj.final_cost_usd, '.3f')}, "
          f"{traj.duration_s / 60.0:.1f} min")
    if args.emit_json:
        traj.to_json(args.emit_json)
        print(f"[aet] wrote snapshot {args.emit_json}")
    if getattr(args, "plot", None):
        try:
            from aet.viz.trajectory_plot import plot_trajectory
            plot_trajectory(traj).savefig(args.plot, dpi=200)
            print(f"[aet] wrote plot {args.plot}")
        except ImportError as e:
            print(f"[aet] {e}", file=sys.stderr)

