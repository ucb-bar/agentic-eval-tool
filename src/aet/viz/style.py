"""aet house plotting style — ported verbatim from the merlin house style.

A single source of truth for every aet figure: a warm cream canvas, ash-black ink, one colour per
series, and 3-D block-shadow bars. Standalone (matplotlib/numpy only). Importing this in a base
install (no ``[viz]`` extra) raises a friendly, actionable ImportError.
"""
from __future__ import annotations

import glob
from pathlib import Path

try:  # the one place the optional dependency is surfaced
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib import font_manager as fm
    from matplotlib.transforms import offset_copy
    from matplotlib.patches import Rectangle
    import numpy as np
except ImportError as e:  # pragma: no cover - exercised via the [viz]-absent path
    raise ImportError(
        "aet visualization requires matplotlib and numpy. "
        "Install with:  pip install 'aet[viz]'"
    ) from e

# ----------------------------------------------------------------- house palette
BG    = "#FDF7EF"   # cream background (all plots)
INK   = "#2E2D2C"   # ash black — text, edges, the 3-D block shadow
GOLD  = "#AB9A89"   # california gold — emphasis text (bold)
BLUE  = "#333351"   # indigo — emphasis text / bars
NAVY  = "#0F3759"   # deep navy — hero bars (ours)
SLATE = "#8B93A6"   # gray-blue — bars
MAUVE = "#815E5E"   # mauve — bars (baseline / killed)
SAGE  = "#7D886C"   # sage — bars (OpenBLAS)

SERIF = "DM Serif Display"
SANS = "Inter"

SHADOW = pe.withSimplePatchShadow(offset=(3.0, -3.0),
                                  shadow_rgbFace=(0.18, 0.178, 0.173), alpha=0.26, rho=1.0)


def use_house_style() -> None:
    """Register the house fonts and apply the rcParams. Safe to call repeatedly."""
    for fp in (glob.glob("/usr/share/fonts/opentype/inter/Inter-*.otf")
               + glob.glob("/usr/share/fonts/opentype/inter/InterDisplay-*.otf")
               + glob.glob(str(Path.home() / ".local/share/fonts/DMSerifDisplay-*.ttf"))):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    plt.rcParams.update({
        "font.family": SANS,
        "font.size": 11.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": INK, "axes.linewidth": 1.0,
        "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
        "legend.frameon": True, "legend.framealpha": 0.95,
        "legend.facecolor": "white", "legend.edgecolor": "#d9cfc0",
        "svg.fonttype": "none",
    })


# Deprecated alias (the style used to be named after the originating project). Kept so external
# callers don't break; prefer ``use_house_style``.
use_merlin_style = use_house_style


def style_ax(ax, *, grid="y") -> None:
    """Cream background, ink left/bottom spines (top/right off), dotted value-axis grid."""
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.0)
    if grid:
        ax.grid(True, axis=grid, ls=":", lw=0.8, color=INK, alpha=0.22, zorder=0)
    ax.tick_params(length=0)


def title(ax, text, fs=15, pad=12) -> None:
    ax.set_title(text, loc="left", color=INK, pad=pad, fontfamily=SERIF, fontsize=fs)


def suptitle(fig, text, y=0.99, fs=18) -> None:
    fig.suptitle(text, color=INK, fontfamily=SERIF, fontsize=fs, y=y)


def emph(ax, x, y, text, color=GOLD, fs=11, **kw):
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, text, color=color, fontsize=fs, **kw)


def block_shadow(ax, x, y, w, h, dx=5.5, dy=-5.5, z=2.4):
    trans = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
    r = Rectangle((x, y), w, h, facecolor=INK, edgecolor="none", zorder=z, transform=trans)
    ax.add_patch(r)
    return r


def vbars(ax, x, heights, color, hatch="", width=0.6, base=0.0, z=3, shadow=True):
    cont = ax.bar(x, np.asarray(heights) - base, width, bottom=base,
                  color=color, edgecolor=INK, linewidth=1.3, zorder=z, hatch=(hatch or None))
    if shadow:
        for p in cont.patches:
            block_shadow(ax, p.get_x(), p.get_y(), p.get_width(), p.get_height(), z=z - 0.6)
    return cont


def hbars(ax, y, widths, color, hatch="", height=0.6, left=0.0, z=3, shadow=True):
    cont = ax.barh(y, np.asarray(widths) - left, height, left=left,
                   color=color, edgecolor=INK, linewidth=1.3, zorder=z, hatch=(hatch or None))
    if shadow:
        for p in cont.patches:
            block_shadow(ax, p.get_x(), p.get_y(), p.get_width(), p.get_height(), z=z - 0.6)
    return cont


# activity-lane colours (match the reference trajectory figures)
ACT_COL = {"think": "#4C4C73", "read": "#6E93B0", "write": "#C2974A",
           "bash": "#9DB682", "tool": "#B06A6A"}
ACT_LAB = {"think": "thinking", "read": "reading", "write": "writing code",
           "bash": "bash / shell", "tool": "tool wait"}
# token-line tones
L_INPUT, L_OUTPUT, L_CACHE, L_TOTAL, L_SPEND = SAGE, NAVY, SLATE, INK, "#4B3F6E"
GOLDLAB = "#7a6a40"

# cream text-halo so labels read over lines (port of the reference _HALO_TXT / LHALO)
def _HALO_TXT(lw: float = 3.4):
    return [pe.withStroke(linewidth=lw, foreground=BG)]


LHALO = [pe.withStroke(linewidth=5.2, foreground=BG)]


# ---- per-series identity for N-arm comparisons (repo-agnostic generalization of RUN_STYLE) ----
_SERIES_COLORS = [MAUVE, NAVY, SLATE, GOLD, SAGE, BLUE]
_SERIES_MARKERS = ["o", "s", "D", "^", "v", "P"]
_SERIES_DASHES = ["-", (0, (7, 2.5)), (0, (1.5, 2.0)), (0, (8, 2.5, 1.5, 2.5)),
                  (0, (4, 2)), (0, (2, 1.5))]


def series_styles(n: int) -> list[tuple]:
    """`n` distinct (color, marker, linestyle) triples cycling the house palette — one per arm.
    Colors assign by order, so a caller keeps identity stable by passing arms in a fixed order."""
    return [(_SERIES_COLORS[i % len(_SERIES_COLORS)],
             _SERIES_MARKERS[i % len(_SERIES_MARKERS)],
             _SERIES_DASHES[i % len(_SERIES_DASHES)]) for i in range(n)]


# ---- label formatters ---------------------------------------------------------------
# Both exist because the fixed-unit versions they replace rounded real measurements away:
# `{duration_s/60:.0f} min` titled a 26-second run "0 min", and `{tokens/1e6:.0f}M`
# labelled a 151,000-token arm "0M". A figure that reports a measurement as zero is worse
# than one that omits it, because zero reads as a finding.


def fmt_duration(seconds: float) -> str:
    """A wall time in the unit that does not round it away — seconds under 90 s, else minutes."""
    if seconds < 90.0:
        return f"{seconds:.0f} s"
    return f"{seconds / 60.0:.0f} min"


def fmt_tokens(count: float) -> str:
    """A token count in the largest unit that still shows a significant digit."""
    if count < 1_000:
        return f"{count:.0f}"
    if count < 1_000_000:
        return f"{count / 1_000:.0f}k"
    return f"{count / 1_000_000:.1f}M"
