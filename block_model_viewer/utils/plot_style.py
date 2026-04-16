"""
GeoX Unified Plotting Standard — Phase 2.

This module defines the single authoritative standard for all analytical
plots in GeoX.  Every Matplotlib figure created anywhere in the application
must use these constants and helper functions to guarantee visual consistency,
scientific correctness, export quality, and theme awareness.

Design philosophy
-----------------
* Scientific-publication quality by default (300 DPI, clear labels, legible
  fonts on both screen and print).
* Dark/light theme awareness — all colors resolve through ``design_tokens``.
* Mining & geostatistics domain conventions (units in metres, grades
  dimensionless unless labelled, semivariance on y-axis, etc.).
* One call to ``style_axes()`` replaces 15+ duplicated lines of boilerplate
  that currently appear in every panel.

Usage
-----
    from block_model_viewer.utils.plot_style import (
        PlotDefaults, create_figure, style_axes, add_statistics_box,
        add_1to1_line, SERIES_COLORS, COMPARISON_COLORS,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    FigureCanvasQTAgg = None

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS — design tokens (soft dependency so unit-tests work stand-alone)
# ═══════════════════════════════════════════════════════════════════════════

def _get_colors():
    """Return the active ColorPalette from design_tokens, or a safe fallback."""
    try:
        from block_model_viewer.ui.design_tokens import tokens
        return tokens.colors()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS — The GeoX Plotting Standard
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _PlotDefaults:
    """Immutable record of every default value used by GeoX plots."""

    # ── ISS-009: Schema version for tracking changes across releases ──
    VERSION: str = "1.0.0"

    # ── Figure ────────────────────────────────────────────────────────
    FIGURE_WIDTH: float = 8.0          # inches
    FIGURE_HEIGHT: float = 6.0         # inches
    FIGURE_DPI: int = 100              # screen DPI (export overrides)

    # ── Export ────────────────────────────────────────────────────────
    EXPORT_DPI: int = 300              # publication / report quality
    EXPORT_FORMAT: str = "png"         # default export format
    EXPORT_BBOX: str = "tight"         # tight bounding box

    # ── Typography ────────────────────────────────────────────────────
    FONT_FAMILY: str = "sans-serif"    # matches Qt UI font stack
    TITLE_SIZE: float = 13.0           # axis title
    TITLE_WEIGHT: str = "semibold"     # title weight
    TITLE_PAD: float = 12.0            # padding below title
    LABEL_SIZE: float = 11.0           # axis labels
    LABEL_WEIGHT: str = "medium"       # axis label weight
    TICK_SIZE: float = 9.5             # tick labels
    LEGEND_SIZE: float = 9.0           # legend text
    ANNOTATION_SIZE: float = 8.0       # annotations & insets
    STATS_BOX_SIZE: float = 8.5        # statistics text boxes
    EQUATION_SIZE: float = 8.5         # model equation boxes

    # ── Line widths ───────────────────────────────────────────────────
    LINE_PRIMARY: float = 2.2          # primary data / model lines
    LINE_SECONDARY: float = 1.5        # secondary / reference lines
    LINE_REFERENCE: float = 1.2        # 1:1 lines, grid references
    LINE_THIN: float = 0.8             # subtle helper lines
    LINE_SPINE: float = 0.8            # axis spine width

    # ── Markers ───────────────────────────────────────────────────────
    MARKER_SIZE: float = 30.0          # scatter marker area (s= param)
    MARKER_SIZE_SMALL: float = 18.0    # dense scatter
    MARKER_SIZE_LARGE: float = 50.0    # highlighted / variogram points
    MARKER_EDGE_WIDTH: float = 0.4     # marker edge line width
    MARKER_ALPHA: float = 0.70         # default marker transparency

    # ── Grid ──────────────────────────────────────────────────────────
    GRID_ALPHA: float = 0.18           # grid line transparency
    GRID_STYLE: str = "-"              # solid (subtle at low alpha)
    GRID_WIDTH: float = 0.5            # grid line width

    # ── Confidence / fill ─────────────────────────────────────────────
    FILL_ALPHA: float = 0.12           # fill_between transparency
    CI_ALPHA: float = 0.10             # confidence interval bands

    # ── Statistics box ────────────────────────────────────────────────
    STATS_BOX_ALPHA: float = 0.88      # background alpha
    STATS_BOX_PAD: float = 0.4         # padding inside box
    STATS_BOX_STYLE: str = "round,pad=0.4"

    # ── Legend ─────────────────────────────────────────────────────────
    LEGEND_ALPHA: float = 0.90         # legend background alpha
    LEGEND_EDGE_WIDTH: float = 0.6     # legend border
    LEGEND_LOC: str = "best"           # default location

    # ── Spacing ───────────────────────────────────────────────────────
    TIGHT_PAD: float = 1.08            # tight_layout pad
    SUBPLOT_HSPACE: float = 0.35       # vertical spacing between subplots
    SUBPLOT_WSPACE: float = 0.30       # horizontal spacing between subplots


PlotDefaults = _PlotDefaults()


# ── Series color palettes ─────────────────────────────────────────────────
# These are sequential colors for overlaying multiple data series on a
# single plot.  Chosen for distinguishability on both dark and light
# backgrounds and for colour-vision-deficiency accessibility.

SERIES_COLORS: List[str] = [
    "#3b82f6",  # blue        — primary
    "#ef4444",  # red         — secondary
    "#22c55e",  # green       — tertiary
    "#f59e0b",  # amber       — quaternary
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
    "#ec4899",  # pink
    "#78350f",  # brown
]

# Specific semantic colors for common plot elements
COLOR_TONNAGE: str = "#3b82f6"      # blue — tonnage lines
COLOR_GRADE: str = "#ef4444"        # red — grade lines
COLOR_MODEL: str = "#ff7043"        # deep orange — fitted model curves
COLOR_EXPERIMENTAL: str = "#4fc3f7" # light blue — experimental variogram
COLOR_EXCLUDED: str = "#F44336"     # red — excluded data points
COLOR_REFERENCE: str = "#9e9e9e"    # grey — reference / helper lines
COLOR_1TO1: str = "#9e9e9e"         # grey — 1:1 parity line
COLOR_REGRESSION: str = "#ff7043"   # orange — regression / trend line
COLOR_NUGGET: str = "#9e9e9e"       # grey — nugget reference
COLOR_SILL: str = "#ef5350"         # red — sill reference
COLOR_RANGE: str = "#66bb6a"        # green — range reference
COLOR_CI: str = "#2980B9"           # confidence interval band

# Comparison palette for multi-source overlays (colour-blind safe)
COMPARISON_COLORS: List[str] = [
    "#3b82f6",  # blue
    "#ef4444",  # red
    "#22c55e",  # green
    "#f59e0b",  # amber
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
    "#ec4899",  # pink
    "#64748b",  # slate
]


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_figure(
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    nrows: int = 1,
    ncols: int = 1,
    constrained_layout: bool = False,
) -> Tuple[Figure, "FigureCanvasQTAgg"]:
    """
    Create a themed Matplotlib figure with Qt canvas.

    Parameters
    ----------
    figsize : tuple, optional
        (width, height) in inches.  Defaults to PlotDefaults values.
    dpi : int, optional
        Screen DPI.  Defaults to PlotDefaults.FIGURE_DPI.
    nrows, ncols : int
        Subplot grid dimensions.
    constrained_layout : bool
        Use constrained_layout instead of tight_layout.

    Returns
    -------
    fig : Figure
    canvas : FigureCanvasQTAgg (or None if Qt backend unavailable)
    """
    w = figsize[0] if figsize else PlotDefaults.FIGURE_WIDTH
    h = figsize[1] if figsize else PlotDefaults.FIGURE_HEIGHT
    d = dpi or PlotDefaults.FIGURE_DPI

    colors = _get_colors()
    bg = colors.BG_ELEVATED if colors else "#333337"

    fig = Figure(figsize=(w, h), dpi=d, facecolor=bg,
                 constrained_layout=constrained_layout)

    canvas = None
    if FigureCanvasQTAgg is not None:
        canvas = FigureCanvasQTAgg(fig)

    return fig, canvas


def style_axes(
    ax: Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid: bool = True,
    legend: bool = False,
    legend_loc: Optional[str] = None,
    minor_ticks: bool = False,
) -> None:
    """
    Apply the GeoX plotting standard to a single Axes.

    This replaces 15+ lines of boilerplate per panel.  It sets:
    background color, text colors, spine colors & width, grid style,
    tick parameters, title, labels, and optional legend styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    title : str
    xlabel, ylabel : str
    grid : bool
    legend : bool
        If True, call ``ax.legend()`` with standard styling.
    legend_loc : str, optional
        Override default legend location.
    minor_ticks : bool
        Show minor ticks if True.
    """
    colors = _get_colors()

    # ── Background ────────────────────────────────────────────────────
    bg = colors.BG_ELEVATED if colors else "#333337"
    text = colors.TEXT_PRIMARY if colors else "#d4d4d4"
    text2 = colors.TEXT_SECONDARY if colors else "#a0a0a0"
    border = colors.BORDER_DEFAULT if colors else "#474747"
    grid_color = colors.TEXT_SECONDARY if colors else "#a0a0a0"

    ax.set_facecolor(bg)

    # ── Spines ────────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_edgecolor(border)
        spine.set_linewidth(PlotDefaults.LINE_SPINE)

    # ── Ticks ─────────────────────────────────────────────────────────
    ax.tick_params(
        colors=text,
        labelsize=PlotDefaults.TICK_SIZE,
        direction="out",
        width=PlotDefaults.LINE_SPINE,
    )
    if minor_ticks:
        ax.minorticks_on()
        ax.tick_params(which="minor", length=3, width=0.5, colors=text)

    # ── Grid ──────────────────────────────────────────────────────────
    if grid:
        ax.grid(
            True,
            alpha=PlotDefaults.GRID_ALPHA,
            color=grid_color,
            linestyle=PlotDefaults.GRID_STYLE,
            linewidth=PlotDefaults.GRID_WIDTH,
        )
    else:
        ax.grid(False)

    # ── Labels ────────────────────────────────────────────────────────
    if title:
        ax.set_title(
            title,
            fontsize=PlotDefaults.TITLE_SIZE,
            fontweight=PlotDefaults.TITLE_WEIGHT,
            color=text,
            pad=PlotDefaults.TITLE_PAD,
        )
    if xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=PlotDefaults.LABEL_SIZE,
            fontweight=PlotDefaults.LABEL_WEIGHT,
            color=text,
        )
    if ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=PlotDefaults.LABEL_SIZE,
            fontweight=PlotDefaults.LABEL_WEIGHT,
            color=text,
        )

    # ── Legend ─────────────────────────────────────────────────────────
    if legend:
        loc = legend_loc or PlotDefaults.LEGEND_LOC
        leg = ax.legend(
            loc=loc,
            fontsize=PlotDefaults.LEGEND_SIZE,
            framealpha=PlotDefaults.LEGEND_ALPHA,
            edgecolor=border,
            facecolor=bg,
        )
        if leg is not None:
            for t in leg.get_texts():
                t.set_color(text)


def style_twin_axis(
    ax: Axes,
    ylabel: str = "",
    color: str = "",
) -> None:
    """
    Style a twinx() axis to match the GeoX standard.

    Parameters
    ----------
    ax : Axes
        The twin axis (result of ``ax_primary.twinx()``).
    ylabel : str
    color : str
        Color for the y-label and tick labels.
    """
    colors = _get_colors()
    border = colors.BORDER_DEFAULT if colors else "#474747"

    for spine in ax.spines.values():
        spine.set_edgecolor(border)
        spine.set_linewidth(PlotDefaults.LINE_SPINE)

    if ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=PlotDefaults.LABEL_SIZE,
            fontweight=PlotDefaults.LABEL_WEIGHT,
            color=color or (colors.TEXT_PRIMARY if colors else "#d4d4d4"),
        )
    if color:
        ax.tick_params(axis="y", labelcolor=color, labelsize=PlotDefaults.TICK_SIZE)
    else:
        ax.tick_params(
            axis="y",
            colors=colors.TEXT_PRIMARY if colors else "#d4d4d4",
            labelsize=PlotDefaults.TICK_SIZE,
        )


def add_statistics_box(
    ax: Axes,
    data: np.ndarray,
    position: str = "upper right",
    include_cv: bool = False,
    include_percentiles: bool = False,
    decimal_places: int = 2,
) -> None:
    """
    Add a formatted statistics text box to an axes.

    Parameters
    ----------
    ax : Axes
    data : np.ndarray
        Raw data (NaN values are removed internally).
    position : str
        'upper right', 'upper left', 'lower right', 'lower left'.
    include_cv : bool
        Include coefficient of variation.
    include_percentiles : bool
        Include P10, P50, P90.
    decimal_places : int
        Decimal precision for values.
    """
    colors = _get_colors()
    bg = colors.BG_ELEVATED if colors else "#333337"
    border = colors.BORDER_DEFAULT if colors else "#474747"
    text_color = colors.TEXT_PRIMARY if colors else "#d4d4d4"

    clean = data[np.isfinite(data)]
    if len(clean) == 0:
        return

    fmt = f".{decimal_places}f"
    lines = [
        f"n = {len(clean):,}",
        f"Mean = {np.mean(clean):{fmt}}",
        f"Std = {np.std(clean):{fmt}}",
        f"Min = {np.min(clean):{fmt}}",
        f"Max = {np.max(clean):{fmt}}",
    ]
    if include_cv and np.mean(clean) != 0:
        cv = np.std(clean) / abs(np.mean(clean))
        lines.append(f"CV = {cv:.3f}")
    if include_percentiles:
        p10, p50, p90 = np.percentile(clean, [10, 50, 90])
        lines.extend([
            f"P10 = {p10:{fmt}}",
            f"P50 = {p50:{fmt}}",
            f"P90 = {p90:{fmt}}",
        ])

    stats_text = "\n".join(lines)

    # Position mapping
    ha_map = {"upper right": "right", "upper left": "left",
              "lower right": "right", "lower left": "left"}
    va_map = {"upper right": "top", "upper left": "top",
              "lower right": "bottom", "lower left": "bottom"}
    x_map = {"upper right": 0.97, "upper left": 0.03,
             "lower right": 0.97, "lower left": 0.03}
    y_map = {"upper right": 0.97, "upper left": 0.97,
             "lower right": 0.03, "lower left": 0.03}

    pos = position if position in ha_map else "upper right"

    ax.text(
        x_map[pos], y_map[pos], stats_text,
        transform=ax.transAxes,
        fontsize=PlotDefaults.STATS_BOX_SIZE,
        fontfamily="monospace",
        verticalalignment=va_map[pos],
        horizontalalignment=ha_map[pos],
        color=text_color,
        bbox=dict(
            boxstyle=PlotDefaults.STATS_BOX_STYLE,
            facecolor=bg,
            edgecolor=border,
            alpha=PlotDefaults.STATS_BOX_ALPHA,
            linewidth=0.6,
        ),
        zorder=8,
    )


def add_1to1_line(
    ax: Axes,
    label: str = "1:1",
    color: Optional[str] = None,
    zorder: int = 2,
) -> None:
    """
    Add a 1:1 reference line spanning the current axis limits.

    The line automatically adjusts to the data extent.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    c = color or COLOR_1TO1
    ax.plot(
        [lo, hi], [lo, hi],
        linestyle="--",
        linewidth=PlotDefaults.LINE_REFERENCE,
        color=c,
        alpha=0.7,
        label=label,
        zorder=zorder,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def add_regression_line(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: Optional[str] = None,
    show_equation: bool = True,
    show_r2: bool = True,
) -> Optional[float]:
    """
    Add a linear regression line with optional equation and R².

    Returns
    -------
    r_squared : float or None
    """
    mask = np.isfinite(x) & np.isfinite(y)
    xc, yc = x[mask], y[mask]
    if len(xc) < 2:
        return None

    z = np.polyfit(xc, yc, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(xc)

    c = color or COLOR_REGRESSION
    label_parts = []
    if show_equation:
        label_parts.append(f"y = {z[0]:.3f}x + {z[1]:.3f}")

    # R² calculation
    yhat = p(xc)
    ss_res = np.sum((yc - yhat) ** 2)
    ss_tot = np.sum((yc - np.mean(yc)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12) if ss_tot > 0 else 0.0

    if show_r2:
        label_parts.append(f"R² = {r2:.3f}")

    ax.plot(
        x_sorted, p(x_sorted),
        linestyle="--",
        linewidth=PlotDefaults.LINE_SECONDARY,
        color=c,
        alpha=0.8,
        label=" | ".join(label_parts) if label_parts else "Regression",
        zorder=3,
    )
    return r2


def save_figure(
    fig: Figure,
    filepath: str,
    dpi: Optional[int] = None,
    transparent: bool = False,
) -> bool:
    """
    Save figure to file using the GeoX export standard.

    Parameters
    ----------
    fig : Figure
    filepath : str
    dpi : int, optional
        Defaults to PlotDefaults.EXPORT_DPI (300).
    transparent : bool
        Transparent background.

    Returns
    -------
    success : bool
    """
    try:
        fig.savefig(
            filepath,
            dpi=dpi or PlotDefaults.EXPORT_DPI,
            bbox_inches=PlotDefaults.EXPORT_BBOX,
            facecolor=fig.get_facecolor() if not transparent else "none",
            edgecolor="none",
            transparent=transparent,
        )
        logger.info("Saved figure to %s", filepath)
        return True
    except Exception as e:
        logger.error("Error saving figure to %s: %s", filepath, e, exc_info=True)
        return False


def no_data_placeholder(ax: Axes, message: str = "No data available") -> None:
    """Show a centred 'no data' message on an empty axes."""
    colors = _get_colors()
    text_color = colors.TEXT_SECONDARY if colors else "#a0a0a0"
    bg = colors.BG_ELEVATED if colors else "#333337"
    ax.set_facecolor(bg)
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        fontsize=PlotDefaults.LABEL_SIZE,
        color=text_color,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def format_number(value: float, decimal_places: int = 2) -> str:
    """Format a number with thousands separators and fixed decimals."""
    if abs(value) >= 1e6:
        return f"{value:,.{decimal_places}f}"
    return f"{value:.{decimal_places}f}"


def format_tonnage_tick(x: float, pos=None) -> str:
    """Format tonnage axis ticks with K/M/B suffixes."""
    if x >= 1e9:
        return f"{x / 1e9:.1f}B"
    elif x >= 1e6:
        return f"{x / 1e6:.1f}M"
    elif x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def render_gt_comparison(
    ax_tonnage: Axes,
    ax_grade: Axes,
    sources: list,
    title: str = "Grade-Tonnage Comparison",
    tonnage_label: str = "Tonnage",
    grade_label: str = "Average Grade",
    normalize_tonnage: bool = False,
) -> None:
    """
    Shared rendering logic for grade-tonnage comparison plots.

    PL-06 fix: single implementation used by both grade_tonnage_panel.py
    and grade_tonnage_basic_panel.py, eliminating ~120 lines of duplication.

    Parameters
    ----------
    ax_tonnage : Axes
        Primary axis (left y-axis) for tonnage.
    ax_grade : Axes
        Secondary axis (right y-axis) for grade.  May be ``ax_tonnage.twinx()``.
    sources : list of dict
        Each dict must have:
        - 'name' : str — display name for legend
        - 'cutoffs' : array-like — cutoff grade values
        - 'tonnage' : array-like — tonnage above cutoff
        - 'grade' : array-like — average grade above cutoff
        Optional:
        - 'total_tonnage' : float — for normalisation
    title : str
    tonnage_label, grade_label : str
    normalize_tonnage : bool
        If True and 'total_tonnage' is present, plot tonnage as % of total.
    """
    colors = _get_colors()
    text = colors.TEXT_PRIMARY if colors else "#d4d4d4"
    bg = colors.BG_ELEVATED if colors else "#333337"
    border = colors.BORDER_DEFAULT if colors else "#474747"

    ax_tonnage.set_facecolor(bg)

    for i, src in enumerate(sources):
        c = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
        cutoffs = np.asarray(src["cutoffs"])
        tonnage = np.asarray(src["tonnage"])
        grade = np.asarray(src["grade"])
        name = src.get("name", f"Source {i + 1}")

        if normalize_tonnage:
            total = src.get("total_tonnage", tonnage[0] if len(tonnage) else 1.0)
            if total > 0:
                tonnage = tonnage / total * 100

        # Tonnage — solid on primary axis
        ax_tonnage.plot(
            cutoffs, tonnage, color=c, linestyle="-",
            linewidth=PlotDefaults.LINE_PRIMARY,
            label=f"{name} (Tonnage)", alpha=0.9, zorder=3,
        )
        # Markers at sparse intervals
        n_markers = min(8, len(cutoffs))
        if n_markers > 1:
            idx = np.linspace(0, len(cutoffs) - 1, n_markers, dtype=int)
            ax_tonnage.scatter(
                cutoffs[idx], tonnage[idx], color=c,
                s=PlotDefaults.MARKER_SIZE, zorder=5,
                edgecolors="white", linewidths=PlotDefaults.MARKER_EDGE_WIDTH,
            )

        # Grade — dashed on secondary axis
        ax_grade.plot(
            cutoffs, grade, color=c, linestyle="--",
            linewidth=PlotDefaults.LINE_SECONDARY,
            label=f"{name} (Grade)", alpha=0.8, zorder=3,
        )

    # ── Styling ──────────────────────────────────────────────────────
    ylabel_ton = "Tonnage (% of Total)" if normalize_tonnage else tonnage_label
    style_axes(ax_tonnage, title=title, xlabel="Cutoff Grade", ylabel=ylabel_ton)
    style_twin_axis(ax_grade, ylabel=grade_label, color=COLOR_GRADE)

    # Format tonnage axis with K/M/B suffixes
    try:
        import matplotlib.ticker as mticker
        ax_tonnage.yaxis.set_major_formatter(mticker.FuncFormatter(format_tonnage_tick))
    except Exception:
        pass

    # Combined legend
    lines1, labels1 = ax_tonnage.get_legend_handles_labels()
    lines2, labels2 = ax_grade.get_legend_handles_labels()
    leg = ax_tonnage.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper right", frameon=True,
        facecolor=bg, edgecolor=border,
        fontsize=PlotDefaults.LEGEND_SIZE,
    )
    if leg:
        for t in leg.get_texts():
            t.set_color(text)
        leg.get_frame().set_alpha(PlotDefaults.LEGEND_ALPHA)
