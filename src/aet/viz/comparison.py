"""Presentation comparison figures — the polished N-arm views, consuming ONLY the data-model.

Three public figures (repo-agnostic generalisations of the oscar-merlin one-off plots):

  * :func:`plot_rate_panels` — one panel per arm, each on its **own** time scale, with input/output
    **token-rate** lines over an activity-share background, gold test-pass milestones, faint round
    dividers, a corner summary chip, and a below-axis fixed-duration ruler (the same "20 min" renders
    at a different length per panel → the differing time scales read at a glance).
  * :func:`plot_cost_vs_time` — a single axes with one cumulative-spend line per arm (``series_styles``
    identity), an endpoint marker + ``$X · YM tok`` label per arm.
  * :func:`plot_tests_facets` — small multiples, one tests-passing step-lane per arm, with fill,
    a marker at each intermediate rise, a big endpoint marker + ``final/total`` label.

Everything is derived from :class:`RunTrajectory` (``token_series``/``bands``/``milestones``/
``rounds``/``tests_steps``) — no raw transcript, no project specifics. matplotlib/numpy stay behind
the ``[viz]`` extra via ``aet.viz.style``'s guarded import.
"""
from __future__ import annotations

from aet.trajectory.model import RunTrajectory
from aet.viz import style as S
from aet.viz.style import plt, np
from aet.viz.trajectory_plot import ACTS, activity_share

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

RND_C = "#6f675c"   # round-divider colour


def _fmt_cost(x: float | None, *, provisional: bool = False, ref: float | None = None) -> str:
    """A $ label whose precision matches the magnitude — cents for sub-$10 sweeps (abc-testing-scale
    per-session costs), whole dollars for large runs — so small-dollar arms don't all read as ``$2``.
    ``ref`` (e.g. the max across arms) picks the precision for a whole figure so labels are uniform.
    ``None`` means unpriced (cost unavailable) — labelled as such, never drawn as a fabricated $0."""
    if x is None:
        return "unpriced"
    scale = ref if ref is not None else x
    digits = 2 if scale < 10 else (1 if scale < 100 else 0)
    return ("~$" if provisional else "$") + f"{x:.{digits}f}"


# --------------------------------------------------------------------- rate
def rate_series(traj: RunTrajectory, series: str = "input", win: int | None = None):
    """(t_min, tok_per_min) — a Hanning-smoothed derivative of a cumulative token curve.

    ``series`` selects the cumulative source (``"input"``/``"output"``/``"cache"``/``"total"``).
    Returns finite, non-negative arrays; degrades to zeros for a run with < 4 points."""
    s = traj.token_series()
    t = np.asarray(s["t"], float)
    arr = np.asarray(s[series], float)
    if len(t) < 4:
        return t, np.zeros_like(arr)
    # Collapse points that share a time (turns can be co-timed at a round start) to the last
    # cumulative value at that time, so np.gradient sees strictly-increasing x with REAL spacing —
    # never a near-zero step that would manufacture a spurious multi-billion tok/min spike.
    t = np.maximum.accumulate(t)                       # enforce non-decreasing
    keep = np.concatenate([np.diff(t) > 1e-9, [True]])  # last of each equal-time run
    tu, au = t[keep], arr[keep]
    if len(tu) < 4:
        return tu, np.zeros_like(au)
    # Resample the (monotonic) cumulative curve onto a uniform fine grid before differentiating, so
    # the rate line is smooth and evenly spaced instead of lurching across widely-spaced turns. This
    # is faithful: a flat cumulative stretch (e.g. a long tool-wait with no tokens) interpolates to a
    # flat segment → gradient 0, so genuine zero-rate waits are preserved, not smeared into a ramp.
    ng = max(len(tu), 240)
    gu = np.linspace(tu[0], tu[-1], ng)
    cu = np.interp(gu, tu, au)
    r = np.gradient(cu, gu)
    w = win or (max(9, ng // 30) | 1)
    k = np.hanning(w)
    k /= k.sum()
    r = np.convolve(np.pad(r, w // 2, mode="edge"), k, mode="valid")[:len(r)]
    return gu, np.clip(r, 0.0, None)


# --------------------------------------------------------------------- panel pieces
def _has_cache_split(traj) -> bool:
    """True when the source recorded read/creation separately rather than only their sum."""
    return any(p.cum_cache_read_tokens or p.cum_cache_creation_tokens for p in traj.points)


#: Legend entry per rate series: (colour, linestyle, label). Keyed by the series name
#: :func:`_rate_lines` reports drawing, so the key can never advertise an absent line.
_RATE_LEGEND = {
    "input":          (S.L_INPUT, "-", "rate input"),
    "output":         (S.L_OUTPUT, "-", "rate output"),
    "cache":          (S.L_CACHE, (0, (5, 2)), "rate cache"),
    "cache_read":     (S.L_CACHE_READ, (0, (5, 2)), "rate cache read"),
    "cache_creation": (S.L_CACHE_WRITE, (0, (1, 2)), "rate cache write"),
}


def _rate_line_handles(drawn):
    """Legend entries for exactly the series :func:`_rate_lines` drew.

    Built from what was drawn rather than from what was asked for. A short run has no
    derivable rate at all — ``rate_series`` needs four points — so a figure can legitimately
    contain no rate lines, and a key listing four of them would be describing another run.
    """
    return [Line2D([0], [0], color=c, lw=3.2, ls=ls, label=lab)
            for name, (c, ls, lab) in _RATE_LEGEND.items() if name in drawn]


def _share_stack(ax, traj, *, band_alpha=0.40):
    """Stack the activity lanes. A run with no bands gets no axis label.

    Labelling an empty 0-1 axis "activity share" told the reader a quantity had been measured and
    come out flat. A source that records no tool events (chia's aet sink, for one) has no activity
    to share out, and the figure must not imply otherwise."""
    if not traj.bands:
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        return
    g, sh = activity_share(traj)
    ax.stackplot(g, *[sh[a] for a in ACTS], colors=[S.ACT_COL[a] for a in ACTS],
                 alpha=band_alpha, zorder=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("activity share", fontsize=13)


def _rate_lines(axT, traj, *, fs=1.0, split_cache=False):
    """Draw the token-rate lines: input, output, and the cache class(es).

    Cache is here because leaving it out was a defect, not a simplification. ``rate_series`` has
    always accepted ``"cache"``; this caller drew input and output alone, so on a cache-heavy run
    the panel labelled "token rate" plotted the minority of the tokens — 696 of 152,088 on the
    reference chia run. Cache reads and writes bill 12.5x apart, so ``split_cache`` separates them
    when the source recorded the split."""
    lines = [("input", S.L_INPUT, "-"), ("output", S.L_OUTPUT, "-")]
    if split_cache and _has_cache_split(traj):
        lines += [("cache_read", S.L_CACHE_READ, (0, (5, 2))),
                  ("cache_creation", S.L_CACHE_WRITE, (0, (1, 2)))]
    else:
        lines += [("cache", S.L_CACHE, (0, (5, 2)))]
    drawn = set()
    for series, col, ls in lines:
        t, r = rate_series(traj, series)
        # An identically-zero series is omitted, not drawn at the log floor. clip(r, 1, ...) is
        # what keeps a log axis drawable, but it turns a true zero into an apparent 1 tok/min —
        # so the warm arm, which writes no cache at all, would show a cache-write rate it never had.
        # A run with fewer than 4 points has no derivable rate at all and lands here too.
        if not np.any(r > 0):
            continue
        axT.plot(t, np.clip(r, 1, None), color=col, lw=3.2 * fs, ls=ls, zorder=8,
                 path_effects=S.LHALO)
        drawn.add(series)
    axT.set_yscale("log")
    axT.set_ylabel("token rate (tok/min, log)", fontsize=13 * fs)
    axT.tick_params(labelsize=10 * fs)
    axT.spines["top"].set_visible(False)
    axT.spines["right"].set_color(S.INK)
    return drawn


def _spend_axis(ax, traj, *, outward=62, fs=1.0):
    s = traj.token_series()
    axS = ax.twinx()
    axS.spines["right"].set_position(("outward", outward))
    axS.spines["top"].set_visible(False)
    axS.spines["right"].set_color(S.L_SPEND)
    axS.plot(s["t"], s["spend"], color=S.L_SPEND, lw=3.2 * fs, ls=(0, (6, 2)),
             zorder=9, path_effects=S.LHALO)
    top = max((s["spend"][-1] if s["spend"] else 0.0) * 1.18, 1.0)
    axS.set_ylim(0, top)
    axS.set_ylabel("cumulative spend ($)", color=S.L_SPEND, fontsize=12 * fs)
    axS.tick_params(colors=S.L_SPEND, labelsize=10 * fs)
    return axS


def _round_dividers(ax, traj, *, topax=None, fs=1.0, labels=True):
    tx = topax or ax
    starts = [rb.t_start_s / 60.0 for rb in traj.rounds]
    for st in starts[1:]:
        ax.axvline(st, color=RND_C, ls=(0, (3, 2)), lw=1.3, alpha=0.55, zorder=2)
    if not labels or not traj.rounds:
        return
    total = max(traj.duration_s / 60.0, 1.0)
    ends = [rb.t_end_s / 60.0 for rb in traj.rounds]
    mids = [(starts[k] + ends[k]) / 2 for k in range(len(starts))]
    min_sep = 0.05 * total
    i = 0
    while i < len(mids):
        j = i
        while j + 1 < len(mids) and mids[j + 1] - mids[j] < min_sep:
            j += 1
        lab = f"r{i}" if i == j else f"r{i} → r{j}"
        xc = mids[i] if i == j else (mids[i] + mids[j]) / 2
        tx.text(xc, 0.022, lab, transform=tx.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=11 * fs, fontweight="bold", color="#4f483f", zorder=20,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#cbbfa8", lw=0.9, alpha=0.96))
        i = j + 1


def _milestones(ax, traj, *, topax=None, fs=1.0, labels=True):
    ms = traj.milestone_series()
    if not ms:
        return
    tx = topax or ax
    n_total = traj.tests_total()
    total = max(traj.duration_s / 60.0, 1.0)
    s = traj.token_series()
    for x, _ in ms:
        tx.axvline(x, color=S.GOLD, ls=(0, (5, 2)), lw=3.0, alpha=1.0, zorder=18)
        tx.plot([x], [1.0], marker="v", ms=12, color=S.GOLD, mec=S.INK, mew=1.2,
                transform=tx.get_xaxis_transform(), clip_on=False, zorder=21)
    if not labels:
        return
    levels = [0.96, 0.74, 0.52, 0.30]
    min_gap = 0.20 * total
    last_x = [-1e18] * len(levels)
    for x, c in sorted(ms):
        li = next((j for j in range(len(levels)) if x - last_x[j] >= min_gap), len(levels) - 1)
        last_x[li] = x
        tok = float(np.interp(x, s["t"], s["total"])) if s["t"] else 0.0
        cost = float(np.interp(x, s["t"], s["spend"])) if s["t"] else 0.0
        ha, dx = ("right", -9) if x > 0.18 * total else ("left", 9)
        frac = f"{c}/{n_total}" if n_total is not None else f"{c}"
        tx.annotate(f"{frac}\n${cost:.1f} · {S.fmt_tokens(tok)}", (x, levels[li]),
                    xycoords=("data", "axes fraction"), xytext=(dx, -4), textcoords="offset points",
                    ha=ha, va="top", fontsize=9.5 * fs, fontweight="bold", color=S.GOLDLAB, zorder=22,
                    bbox=dict(boxstyle="round,pad=0.30", fc="#fdf6e6", ec=S.GOLD, lw=1.2, alpha=1.0))


def _chip(ax, traj, *, y=1.045, fs=1.0):
    tok = traj.final_input_tokens + traj.final_output_tokens + traj.final_cache_tokens
    n_total = traj.tests_total()
    cost = _fmt_cost(traj.final_cost_usd, provisional=traj.provisional)
    n_rounds = traj.num_rounds
    parts = [f"{S.fmt_duration(traj.duration_s)} active", cost, f"{S.fmt_tokens(tok)} tok",
             f"{n_rounds} round{'' if n_rounds == 1 else 's'}"]
    # Only claim a score when one was recorded. A run with no test record used to render
    # "final 0/20" here, against a denominator nothing had measured.
    if n_total is not None:
        parts.append(f"final {traj.final_tests()}/{n_total}")
    txt = "   ·   ".join(parts)
    ax.text(1.0, y, txt, transform=ax.transAxes, fontsize=11.5 * fs, color=S.INK,
            va="bottom", ha="right", zorder=11,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#d9cfc0", lw=1.0))


def _scale_bar(ax, traj, *, minutes=20, y=-0.185, fs=1.0, color=S.BLUE):
    """A fixed-``minutes`` ruler drawn just below the time axis; left-aligned at t=0. Because each
    panel has its own time scale, the same duration renders at a different length per panel.

    The bar length is **clipped to the panel's own axis** so a ruler longer than a short run never
    extends past the axes — otherwise ``bbox_inches='tight'`` would blow the canvas up (a 20-min bar
    on a 0.01-min run is 2000× the axis width)."""
    total = traj.duration_s / 60.0
    if total <= 0 or minutes <= 0:
        return
    tr = ax.get_xaxis_transform()
    x0, x1 = 0.0, min(float(minutes), total)     # never draw past the panel's own time span
    ax.plot([x0, x1], [y, y], transform=tr, color=color, lw=4.2, zorder=27,
            solid_capstyle="butt", clip_on=False)
    for xx in (x0, x1):
        ax.plot([xx, xx], [y - 0.032, y + 0.032], transform=tr, color=color, lw=4.2,
                zorder=27, clip_on=False)
    ax.text(x1 + 0.012 * total, y, f"{minutes} min", transform=tr, ha="left", va="center",
            fontsize=15 * fs, fontweight="bold", color=color, zorder=28, clip_on=False)


def _panel_rate(ax, traj, label, last, *, show_spend=False, show_milestones=True,
                round_labels=True, scale_bar_minutes=20, fs=1.0, split_cache=False):
    S.style_ax(ax, grid=None)
    _share_stack(ax, traj)
    axT = ax.twinx()
    drawn = _rate_lines(axT, traj, fs=fs, split_cache=split_cache)
    axfront = _spend_axis(ax, traj, fs=fs) if show_spend else axT
    _round_dividers(ax, traj, topax=axfront, fs=fs, labels=round_labels)
    if show_milestones:
        _milestones(ax, traj, topax=axfront, fs=fs, labels=True)
    _chip(ax, traj, fs=fs)
    S.title(ax, label, fs=17 * fs, pad=10)
    ax.set_xlim(0, max(traj.duration_s / 60.0, 1e-6) * 1.01)
    ax.tick_params(labelsize=12 * fs)
    if scale_bar_minutes:
        _scale_bar(ax, traj, minutes=scale_bar_minutes, fs=fs)
    if last:
        pad = 54 if scale_bar_minutes else 16
        ax.set_xlabel("Time (min)   —   own scale per arm", fontsize=15 * fs, labelpad=pad)
    return drawn


def _labels_for(trajs, labels):
    if labels:
        return [labels[i] if i < len(labels) else trajs[i].run_id for i in range(len(trajs))]
    return [t.run_id for t in trajs]


# --------------------------------------------------------------------- public figures
def _nice_scale_bar(trajs) -> float:
    """A 'nice' ruler length (min) that fits every panel: the largest of 1/2/5/10/15/30/60… that is
    ≤ 90% of the shortest run, so the same absolute bar is drawable on every arm's own scale."""
    durs = [t.duration_s / 60.0 for t in trajs if t.duration_s > 0]
    if not durs:
        return 0.0
    shortest = min(durs)
    nice = [1, 2, 5, 10, 15, 20, 30, 60, 120, 240]
    fit = [b for b in nice if b <= shortest * 0.9]
    return float(fit[-1]) if fit else round(shortest * 0.5, 1)


def plot_rate_panels(trajs, labels=None, *, independent_scales=True, scale_bar_minutes=None,
                     show_spend=False, show_milestones=True, split_cache=False):
    """N per-arm token-rate panels, each on its own time scale with a fixed-duration ruler.

    ``scale_bar_minutes`` defaults to an auto-picked 'nice' length that fits the shortest run (so it
    works for minute-scale sweeps *and* hour-scale runs); pass a number to force it, or 0 to omit."""
    S.use_house_style()
    trajs = [t for t in trajs if t is not None]
    labs = _labels_for(trajs, labels)
    n = max(1, len(trajs))
    figh = 3.0 + 3.6 * n
    if scale_bar_minutes is None:
        scale_bar_minutes = _nice_scale_bar(trajs)
    bar_m = scale_bar_minutes if independent_scales else 0
    fig, axes = plt.subplots(n, 1, figsize=(19, figh), squeeze=False)
    axes = [a[0] for a in axes]
    drawn_series: set = set()
    fig.subplots_adjust(left=0.06, right=0.88, top=1 - 0.9 / figh, bottom=1.9 / figh, hspace=0.6)
    for i, (ax, traj) in enumerate(zip(axes, trajs)):
        drawn_series |= _panel_rate(ax, traj, labs[i], i == len(trajs) - 1,
                                    show_spend=show_spend, show_milestones=show_milestones,
                                    scale_bar_minutes=bar_m, fs=1.35, split_cache=split_cache)
    # Only advertise activity lanes that something actually drew. The legend used to list all five
    # unconditionally, so a run with no tool events produced a five-entry key over a blank axis.
    drawn = {b.category for t in trajs for b in t.bands}
    handles = [Patch(fc=S.ACT_COL[a], alpha=0.4, label=S.ACT_LAB[a]) for a in ACTS if a in drawn]
    handles += _rate_line_handles(drawn_series)
    if show_milestones and any(t.milestones for t in trajs):
        handles.append(Line2D([0], [0], color=S.GOLD, lw=2.6, ls=(0, (4, 3)),
                              label="test-pass milestone"))
    handles.append(Line2D([0], [0], color=RND_C, lw=1.3, ls=(0, (3, 2)), label="round"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=13, frameon=True,
               facecolor="white", edgecolor="#d9cfc0", bbox_to_anchor=(0.5, 0.008),
               columnspacing=1.3, handlelength=1.8, borderpad=0.7)
    return fig


def plot_cost_vs_time(trajs, labels=None):
    """Single axes: one cumulative-spend line per arm + endpoint ``$X · YM tok`` label."""
    S.use_house_style()
    trajs = [t for t in trajs if t is not None]
    labs = _labels_for(trajs, labels)
    styles = S.series_styles(len(trajs))
    fig, ax = plt.subplots(figsize=(15, 8.8))
    S.style_ax(ax)
    xmax = max((t.duration_s / 60.0 for t in trajs), default=1.0)
    ymax = max((t.final_cost_usd or 0.0 for t in trajs), default=1.0)
    for traj, lab, (col, mk, ls) in zip(trajs, labs, styles):
        s = traj.token_series()
        ax.plot(s["t"], s["spend"], color=col, lw=4.2, ls=ls, zorder=5, solid_capstyle="round")
        xe = s["t"][-1] if s["t"] else 0.0
        ye = s["spend"][-1] if s["spend"] else 0.0
        tok = traj.final_input_tokens + traj.final_output_tokens + traj.final_cache_tokens
        ax.scatter([xe], [ye], s=230, color=col, ec=S.INK, lw=1.8, zorder=7, marker=mk)
        ax.annotate(f"{_fmt_cost(traj.final_cost_usd, provisional=traj.provisional, ref=ymax)}"
                    f" · {S.fmt_tokens(tok)}", (xe, ye),
                    xytext=(0, 18), textcoords="offset points", color=col, fontsize=19,
                    fontweight="bold", va="bottom", ha="center", zorder=9,
                    path_effects=S._HALO_TXT(3.8))
    ax.set_xlim(0, xmax * 1.12)
    ax.set_ylim(0, ymax * 1.26)
    ax.set_xlabel("Time (min)", fontsize=22)
    ax.set_ylabel("cumulative cost (USD)", fontsize=22)
    ax.tick_params(labelsize=18)
    S.title(ax, "Spend over time   (label = total $ · tokens)", fs=25)
    ax.legend(handles=[Line2D([0], [0], color=c, lw=4.2, ls=ls, marker=m, mfc=c, mec=S.INK, ms=12,
                              label=lab) for lab, (c, m, ls) in zip(labs, styles)],
              loc="upper left", fontsize=17, framealpha=0.96)
    return fig


def plot_tests_facets(trajs, labels=None):
    """Small multiples: one tests-passing step-lane per arm, shared time axis, zero overlap."""
    S.use_house_style()
    trajs = [t for t in trajs if t is not None]
    labs = _labels_for(trajs, labels)
    styles = S.series_styles(len(trajs))
    n = max(1, len(trajs))
    figh = 3.4 * n + 1.4
    fig, axes = plt.subplots(n, 1, figsize=(15.5, figh), sharex=True, squeeze=False)
    axes = [a[0] for a in axes]
    fig.subplots_adjust(left=0.085, right=0.965, top=1 - 1.5 / figh,
                        bottom=0.9 / figh, hspace=0.6)
    xmax = max((t.duration_s / 60.0 for t in trajs), default=1.0)
    for i, (ax, traj, lab, (col, mk, ls)) in enumerate(zip(axes, trajs, labs, styles)):
        S.style_ax(ax)
        xs, ys = traj.tests_steps()
        # each lane scales to its OWN suite size (arms may have different N: 8 vs 32 vs 182), so a
        # small suite isn't dwarfed by a large one on a shared y-axis
        ntot = max(traj.tests_total() or 0, 1)
        ax.fill_between(xs, 0, ys, step="post", color=col, alpha=0.18, zorder=2)
        ax.step(xs, ys, where="post", color=col, lw=5.0, zorder=5, solid_capstyle="round")
        rises = [k for k in range(1, len(ys)) if ys[k] != ys[k - 1]]
        ax.scatter([xs[k] for k in rises], [ys[k] for k in rises], s=140, color=col, ec=S.INK,
                   lw=1.6, zorder=6, marker=mk)
        ax.scatter([xs[-1]], [ys[-1]], s=320, color=col, ec=S.INK, lw=2.2, zorder=7, marker=mk)
        ax.set_xlim(0, xmax * 1.12)
        ax.set_ylim(0, ntot * 1.09)
        ax.set_yticks(sorted({0, ntot // 2, ntot}) if ntot > 1 else [0, 1])
        ax.tick_params(labelsize=20)
        ax.set_ylabel("tests", fontsize=24)
        ax.set_title(lab, loc="left", color=col, fontsize=26, fontweight="bold", pad=10)
        n_total = traj.tests_total()
        endpoint = f"{ys[-1]}/{n_total}" if n_total is not None else f"{ys[-1]}"
        ax.text(xs[-1] + 0.012 * xmax, ys[-1], endpoint, color=col,
                fontsize=23, fontweight="bold", va="center", ha="left",
                path_effects=S._HALO_TXT(4.0))
        if i == n - 1:
            ax.set_xlabel("Time (min)", fontsize=25)
    S.suptitle(fig, "Tests passing over time — one lane per run", y=1 - 0.45 / figh, fs=30)
    return fig
