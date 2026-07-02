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
    r = np.gradient(au, tu)
    w = win or (max(7, len(r) // 22) | 1)
    k = np.hanning(w)
    k /= k.sum()
    r = np.convolve(np.pad(r, w // 2, mode="edge"), k, mode="valid")[:len(r)]
    return tu, np.clip(r, 0.0, None)


# --------------------------------------------------------------------- panel pieces
def _share_stack(ax, traj, *, band_alpha=0.40):
    g, sh = activity_share(traj)
    ax.stackplot(g, *[sh[a] for a in ACTS], colors=[S.ACT_COL[a] for a in ACTS],
                 alpha=band_alpha, zorder=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("activity share", fontsize=13)


def _rate_lines(axT, traj, *, fs=1.0):
    for series, col in (("input", S.L_INPUT), ("output", S.L_OUTPUT)):
        t, r = rate_series(traj, series)
        axT.plot(t, np.clip(r, 1, None), color=col, lw=3.2 * fs, zorder=8, path_effects=S.LHALO)
    axT.set_yscale("log")
    axT.set_ylabel("token rate (tok/min, log)", fontsize=13 * fs)
    axT.tick_params(labelsize=10 * fs)
    axT.spines["top"].set_visible(False)
    axT.spines["right"].set_color(S.INK)


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
        tx.annotate(f"{c}/{n_total}\n${cost:.1f} · {tok / 1e6:.1f}M", (x, levels[li]),
                    xycoords=("data", "axes fraction"), xytext=(dx, -4), textcoords="offset points",
                    ha=ha, va="top", fontsize=9.5 * fs, fontweight="bold", color=S.GOLDLAB, zorder=22,
                    bbox=dict(boxstyle="round,pad=0.30", fc="#fdf6e6", ec=S.GOLD, lw=1.2, alpha=1.0))


def _chip(ax, traj, *, y=1.045, fs=1.0):
    tok = (traj.final_input_tokens + traj.final_output_tokens + traj.final_cache_tokens) / 1e6
    fin = traj.final_tests()
    n_total = traj.tests_total()
    cost = ("~$" if traj.provisional else "$") + f"{traj.final_cost_usd:.0f}"
    txt = (f"{traj.duration_s / 60.0:.0f} min active   ·   {cost}   ·   {tok:.0f}M tok   ·   "
           f"{traj.num_rounds} rounds   ·   final {fin}/{n_total}")
    ax.text(1.0, y, txt, transform=ax.transAxes, fontsize=11.5 * fs, color=S.INK,
            va="bottom", ha="right", zorder=11,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#d9cfc0", lw=1.0))


def _scale_bar(ax, traj, *, minutes=20, y=-0.185, fs=1.0, color=S.BLUE):
    """A fixed-``minutes`` ruler drawn just below the time axis; left-aligned at t=0. Because each
    panel has its own time scale, the same duration renders at a different length per panel."""
    tr = ax.get_xaxis_transform()
    total = max(traj.duration_s / 60.0, 1.0)
    x0, x1 = 0.0, float(minutes)
    ax.plot([x0, x1], [y, y], transform=tr, color=color, lw=4.2, zorder=27,
            solid_capstyle="butt", clip_on=False)
    for xx in (x0, x1):
        ax.plot([xx, xx], [y - 0.032, y + 0.032], transform=tr, color=color, lw=4.2,
                zorder=27, clip_on=False)
    ax.text(x1 + 0.012 * total, y, f"{minutes} min", transform=tr, ha="left", va="center",
            fontsize=15 * fs, fontweight="bold", color=color, zorder=28, clip_on=False)


def _panel_rate(ax, traj, label, last, *, show_spend=False, show_milestones=True,
                round_labels=True, scale_bar_minutes=20, fs=1.0):
    S.style_ax(ax, grid=None)
    _share_stack(ax, traj)
    axT = ax.twinx()
    _rate_lines(axT, traj, fs=fs)
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


def _labels_for(trajs, labels):
    if labels:
        return [labels[i] if i < len(labels) else trajs[i].run_id for i in range(len(trajs))]
    return [t.run_id for t in trajs]


# --------------------------------------------------------------------- public figures
def plot_rate_panels(trajs, labels=None, *, independent_scales=True, scale_bar_minutes=20,
                     show_spend=False, show_milestones=True):
    """N per-arm token-rate panels, each on its own time scale with a fixed-duration ruler."""
    S.use_merlin_style()
    trajs = [t for t in trajs if t is not None]
    labs = _labels_for(trajs, labels)
    n = max(1, len(trajs))
    figh = 3.0 + 3.6 * n
    bar_m = scale_bar_minutes if independent_scales else 0
    fig, axes = plt.subplots(n, 1, figsize=(19, figh), squeeze=False)
    axes = [a[0] for a in axes]
    fig.subplots_adjust(left=0.06, right=0.88, top=1 - 0.9 / figh, bottom=1.9 / figh, hspace=0.6)
    for i, (ax, traj) in enumerate(zip(axes, trajs)):
        _panel_rate(ax, traj, labs[i], i == len(trajs) - 1, show_spend=show_spend,
                    show_milestones=show_milestones, scale_bar_minutes=bar_m, fs=1.35)
    handles = [Patch(fc=S.ACT_COL[a], alpha=0.4, label=S.ACT_LAB[a]) for a in ACTS] + [
        Line2D([0], [0], color=S.L_INPUT, lw=3.2, label="rate input"),
        Line2D([0], [0], color=S.L_OUTPUT, lw=3.2, label="rate output")]
    if show_milestones:
        handles.append(Line2D([0], [0], color=S.GOLD, lw=2.6, ls=(0, (4, 3)),
                              label="test-pass milestone"))
    handles.append(Line2D([0], [0], color=RND_C, lw=1.3, ls=(0, (3, 2)), label="round"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=13, frameon=True,
               facecolor="white", edgecolor="#d9cfc0", bbox_to_anchor=(0.5, 0.008),
               columnspacing=1.3, handlelength=1.8, borderpad=0.7)
    return fig


def plot_cost_vs_time(trajs, labels=None):
    """Single axes: one cumulative-spend line per arm + endpoint ``$X · YM tok`` label."""
    S.use_merlin_style()
    trajs = [t for t in trajs if t is not None]
    labs = _labels_for(trajs, labels)
    styles = S.series_styles(len(trajs))
    fig, ax = plt.subplots(figsize=(15, 8.8))
    S.style_ax(ax)
    xmax = max((t.duration_s / 60.0 for t in trajs), default=1.0)
    ymax = max((t.final_cost_usd for t in trajs), default=1.0)
    for traj, lab, (col, mk, ls) in zip(trajs, labs, styles):
        s = traj.token_series()
        ax.plot(s["t"], s["spend"], color=col, lw=4.2, ls=ls, zorder=5, solid_capstyle="round")
        xe = s["t"][-1] if s["t"] else 0.0
        ye = s["spend"][-1] if s["spend"] else 0.0
        tok = (traj.final_input_tokens + traj.final_output_tokens + traj.final_cache_tokens) / 1e6
        ax.scatter([xe], [ye], s=230, color=col, ec=S.INK, lw=1.8, zorder=7, marker=mk)
        prefix = "~$" if traj.provisional else "$"
        ax.annotate(f"{prefix}{traj.final_cost_usd:.0f} · {tok:.0f}M", (xe, ye),
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
    S.use_merlin_style()
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
    ymax = max((t.tests_total() for t in trajs), default=20)
    for i, (ax, traj, lab, (col, mk, ls)) in enumerate(zip(axes, trajs, labs, styles)):
        S.style_ax(ax)
        xs, ys = traj.tests_steps()
        ax.fill_between(xs, 0, ys, step="post", color=col, alpha=0.18, zorder=2)
        ax.step(xs, ys, where="post", color=col, lw=5.0, zorder=5, solid_capstyle="round")
        rises = [k for k in range(1, len(ys)) if ys[k] != ys[k - 1]]
        ax.scatter([xs[k] for k in rises], [ys[k] for k in rises], s=140, color=col, ec=S.INK,
                   lw=1.6, zorder=6, marker=mk)
        ax.scatter([xs[-1]], [ys[-1]], s=320, color=col, ec=S.INK, lw=2.2, zorder=7, marker=mk)
        ax.set_xlim(0, xmax * 1.12)
        ax.set_ylim(0, ymax * 1.09)
        ax.set_yticks([0, ymax // 2, ymax])
        ax.tick_params(labelsize=20)
        ax.set_ylabel("tests", fontsize=24)
        ax.set_title(lab, loc="left", color=col, fontsize=26, fontweight="bold", pad=10)
        ax.text(xs[-1] + 0.012 * xmax, ys[-1], f"{ys[-1]}/{traj.tests_total()}", color=col,
                fontsize=23, fontweight="bold", va="center", ha="left",
                path_effects=S._HALO_TXT(4.0))
        if i == n - 1:
            ax.set_xlabel("Time (min)", fontsize=25)
    S.suptitle(fig, "Tests passing over time — one lane per run", y=1 - 0.45 / figh, fs=30)
    return fig
