"""Plot a :class:`RunTrajectory` in the house style — a thin consumer of the data-model.

Reads only ``token_series()`` / ``bands`` / ``milestones`` / ``rounds`` — never a raw transcript.
``plot_trajectory`` renders one run (cumulative tokens on a log axis, cumulative spend on a twin
axis, activity-share background bands, and gold test-pass milestones); ``plot_comparison`` stacks
several runs on aligned panels.
"""
from __future__ import annotations

from aet.trajectory.model import RunTrajectory
from aet.viz import style as S
from aet.viz.style import plt, np, pe   # matplotlib guard lives in style


ACTS = ["think", "read", "write", "bash", "tool"]
_HALO = [pe.withStroke(linewidth=3.6, foreground=S.BG)]


def activity_share(traj: RunTrajectory, ngrid: int = 420, win: int = 23):
    """(grid_minutes, {category: share}) — Hanning-smoothed occupancy of each activity lane."""
    total_min = traj.duration_s / 60.0
    g = np.linspace(0, total_min, ngrid) if total_min else np.array([0.0, 1.0])
    raw = {a: np.zeros(len(g)) for a in ACTS}
    for b in traj.bands:
        if b.category not in raw:
            continue
        x0, x1 = b.t0_s / 60.0, b.t1_s / 60.0
        raw[b.category][(g >= x0) & (g < x1)] += 1.0
    if win > 1:
        k = np.hanning(win)
        k /= k.sum()
        for a in ACTS:
            raw[a] = np.convolve(np.pad(raw[a], win // 2, mode="edge"), k, mode="valid")[:len(g)]
    tot = sum(raw[a] for a in ACTS)
    tot[tot == 0] = 1.0
    return g, {a: raw[a] / tot for a in ACTS}


def _coarse_bands(ax, traj, alpha=0.28, ngrid=240, win=35):
    """Colour the background by the dominant activity in each stretch of time."""
    g, sh = activity_share(traj, ngrid=ngrid, win=win)
    if not len(g):
        return
    dom = [max(ACTS, key=lambda a: sh[a][i]) for i in range(len(g))]
    i = 0
    while i < len(g):
        j = i
        while j + 1 < len(g) and dom[j + 1] == dom[i]:
            j += 1
        ax.axvspan(g[i], g[min(j + 1, len(g) - 1)], color=S.ACT_COL[dom[i]],
                   alpha=alpha, lw=0, zorder=0)
        i = j + 1


def plot_trajectory(traj: RunTrajectory, *, ax=None, log_tokens: bool = True,
                    show_spend: bool = True, show_milestones: bool = True,
                    show_activity: bool = True, fs: float = 1.0):
    """Render one run's trajectory; returns the Figure."""
    S.use_merlin_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 4.2))
    else:
        fig = ax.figure
    S.style_ax(ax, grid="y")

    s = traj.token_series()
    t = s["t"]

    if show_activity and traj.bands:
        _coarse_bands(ax, traj)

    # cumulative token lines
    if t:
        ax.plot(t, s["output"], color=S.L_OUTPUT, lw=2.4, label="output tokens", zorder=5)
        ax.plot(t, s["input"], color=S.L_INPUT, lw=2.0, label="input tokens", zorder=5)
        ax.plot(t, s["cache"], color=S.L_CACHE, lw=1.6, ls=(0, (5, 2)),
                label="cache tokens", zorder=4)
        ax.plot(t, s["total"], color=S.L_TOTAL, lw=1.4, ls=":", label="total tokens", zorder=4)
    if log_tokens:
        ax.set_yscale("log")
    ax.set_xlabel("Time (min)", fontsize=13 * fs)
    ax.set_ylabel("cumulative tokens", fontsize=13 * fs)
    ax.tick_params(labelsize=11 * fs)
    ax.set_xlim(0, max(t[-1] if t else 1.0, 1e-6))

    # cumulative spend on a twin axis
    if show_spend and t:
        axs = ax.twinx()
        axs.plot(t, s["spend"], color=S.L_SPEND, lw=2.2, label="spend ($)", zorder=6)
        axs.set_ylabel("cumulative spend ($)", fontsize=12 * fs, color=S.L_SPEND)
        axs.tick_params(axis="y", labelsize=10 * fs, colors=S.L_SPEND)
        for sp in ("top",):
            axs.spines[sp].set_visible(False)

    # gold test-pass milestones
    if show_milestones and traj.milestones:
        ymin, ymax = ax.get_ylim()
        for (mt, mc) in traj.milestone_series():
            ax.axvline(mt, color=S.GOLD, ls=(0, (2, 2)), lw=1.4, zorder=3)
            n_total = traj.milestones[0].n_total or 20
            ax.text(mt, ymax, f" {mc}/{n_total}", color=S.GOLDLAB, fontsize=11 * fs,
                    fontweight="bold", va="top", ha="left", rotation=90,
                    path_effects=_HALO, zorder=7)

    # faint round dividers
    for rb in traj.rounds[1:]:
        ax.axvline(rb.t_start_s / 60.0, color=S.INK, ls=(0, (1, 3)), lw=0.8, alpha=0.35, zorder=2)

    cost = f"~${traj.final_cost_usd:.2f}" if traj.provisional else f"${traj.final_cost_usd:.2f}"
    S.title(ax, f"{traj.run_id}    ·    {traj.num_rounds} rounds · {cost} · "
                f"{traj.duration_s / 60.0:.0f} min", fs=15 * fs)
    fig.tight_layout()
    return fig


def plot_comparison(trajs: list[RunTrajectory], *, labels=None, **kw):
    """Stack several runs on aligned panels; returns the Figure."""
    S.use_merlin_style()
    trajs = [t for t in trajs if t is not None]
    n = max(1, len(trajs))
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.4 * n), squeeze=False)
    for i, traj in enumerate(trajs):
        if labels and i < len(labels):
            traj.run_id = labels[i]
        plot_trajectory(traj, ax=axes[i][0], **kw)
    fig.tight_layout()
    return fig
