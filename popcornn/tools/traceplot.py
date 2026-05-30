"""Reusable vertically stacked per-iter trace plotter.

Designed to render the locked "4-stack" trajectory layout used by the
fp32 3-seed sweep + comparison plots, but stays domain-agnostic — accepts
arbitrary series and panel definitions.

Typical use:

    from popcornn.tools.traceplot import (
        plot_stacked_traces, Series, Panel, read_jsonl,
    )
    series = [
        Series(rows=read_jsonl('a.jsonl'), label='seed 0', color='C0'),
        Series(rows=read_jsonl('b.jsonl'), label='seed 1', color='C1'),
    ]
    panels = [
        Panel(field='barrier', ylabel='barrier'),
        Panel(field='fmax',    ylabel='fmax', log_y=True, hlines=[(0.05, 'target')]),
        Panel(field='loss',    ylabel='loss'),
        Panel(field='gnorm_inf', ylabel='|g|_inf', log_y=True, hlines=[(1e-3, 'thr')]),
    ]
    plot_stacked_traces(series, panels, title='my run', output_path='out.png')

The locked layout (used across the fp32 3-seed plots) is:
  - vertical stack, sharex=True
  - title is set on the top panel only (not figure suptitle)
  - uniform font size (default 10)
  - per-series convergence trigger marked as dashed vertical line
  - optional white-bg annotation textbox in the top panel
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


__all__ = [
    'Series', 'Panel',
    'plot_stacked_traces',
    'read_jsonl', 'extract_xy',
    'last_iter', 'last_value',
]


@dataclass
class Series:
    """One trajectory to plot. `rows` is a list of {field: value} dicts."""
    rows: list[dict]
    label: str | None = None
    color: str = 'C0'
    linestyle: str = '-'
    alpha: float = 0.95
    lw: float = 1.1
    iter_field: str = 'iter'
    wall_field: str | None = 't_s'
    show_trigger: bool = True


@dataclass
class Panel:
    """One panel in the stacked figure."""
    field: str
    ylabel: str
    log_y: bool = False
    # Horizontal reference lines: list of (y, label). label=None → no legend.
    hlines: list[tuple[float, str | None]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts (one per line)."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_xy(rows: list[dict], field: str, iter_field: str = 'iter') -> tuple[np.ndarray, np.ndarray]:
    """Pull (x=iter, y=field) arrays, skipping rows where the value is None/NaN."""
    xs, ys = [], []
    for r in rows:
        v = r.get(field)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):
            continue
        xs.append(r[iter_field])
        ys.append(fv)
    return np.asarray(xs), np.asarray(ys)


def last_iter(rows: list[dict], iter_field: str = 'iter') -> int | None:
    return int(rows[-1][iter_field]) if rows else None


def last_value(rows: list[dict], field: str) -> float | None:
    for r in reversed(rows):
        v = r.get(field)
        if v is not None:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(f):
                return f
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _set_uniform_font(font_size: int) -> None:
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size,
    })


def _draw_panel(ax: Axes, panel: Panel, series: Iterable[Series], xlabel: str | None) -> None:
    for s in series:
        xs, ys = extract_xy(s.rows, panel.field, s.iter_field)
        if len(xs) == 0:
            continue
        ax.plot(xs, ys,
                color=s.color, linestyle=s.linestyle,
                lw=s.lw, alpha=s.alpha)
    for y, lbl in panel.hlines:
        ax.axhline(y, color='k', ls=':', lw=0.9, alpha=0.7, zorder=0,
                   label=lbl)
    if panel.log_y:
        ax.set_yscale('log')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(panel.ylabel)
    ax.grid(True, alpha=0.3)


def _draw_triggers(ax: Axes, series: Iterable[Series]) -> None:
    for s in series:
        if not s.show_trigger:
            continue
        it = last_iter(s.rows, s.iter_field)
        if it is None:
            continue
        ax.axvline(it, color=s.color, linestyle=s.linestyle,
                   lw=0.8, alpha=0.55)


def plot_stacked_traces(
        series: list[Series],
        panels: list[Panel],
        title: str,
        output_path: str | Path,
        *,
        figsize: tuple[float, float] | None = None,
        font_size: int = 10,
        legend_handles: list[Line2D] | None = None,
        legend_loc: str = 'best',
        annotation: str | None = None,
        annotation_loc: tuple[float, float] = (0.985, 0.04),
        sharex: bool = True,
        xlabel: str | None = None,
        dpi: int = 140,
) -> Path:
    """Render a vertically stacked sharex figure to ``output_path``.

    Parameters
    ----------
    series : list[Series]
        One entry per trajectory. The color / linestyle / label is encoded
        per series (e.g. one Series per seed, or per (seed, recipe) pair).
    panels : list[Panel]
        One entry per metric. Drawn top-to-bottom.
    title : str
        Set on the topmost panel via ``ax.set_title``. May contain ``\\n``.
    output_path : str | Path
        PNG output location.
    figsize : (w, h), optional
        Auto-derived from panel count as ``(8.5, 2.75 * n_panels)`` if omitted.
    font_size : int
        Applied via rcParams for the duration of the call (NOT restored).
    legend_handles : list[Line2D], optional
        Custom legend entries on the top panel (overrides auto from series).
        Pass an empty list to suppress the legend entirely.
    annotation : str, optional
        Multi-line text rendered as a monospace whitebox in the top panel.
    annotation_loc : (x, y), optional
        Axes-fraction location for the annotation (default upper-right).
    sharex : bool
        Whether all panels share the x-axis. Default True.

    Returns
    -------
    Path to the saved PNG.
    """
    if figsize is None:
        figsize = (8.5, 2.75 * len(panels))
    _set_uniform_font(font_size)

    # Auto xlabel: prefer explicit arg; else 'iter' if the iter_field of
    # the first series is 'iter', else use that field as the label.
    if xlabel is None and series:
        f = series[0].iter_field
        xlabel = 'iter' if f == 'iter' else f
    fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=sharex)
    if len(panels) == 1:
        axes = [axes]
    for i, (ax, panel) in enumerate(zip(axes, panels)):
        is_last = (i == len(panels) - 1)
        # Only the last panel gets an xlabel when sharex (default), all panels otherwise.
        _draw_panel(ax, panel, series,
                    xlabel=xlabel if (is_last or not sharex) else None)
        _draw_triggers(ax, series)

    # Legend (auto from series unless caller overrides).
    ax_top = axes[0]
    if legend_handles is None:
        auto = []
        for s in series:
            if s.label is None:
                continue
            auto.append(Line2D([0], [0], color=s.color, linestyle=s.linestyle,
                              lw=1.4, label=s.label))
        legend_handles = auto
    if legend_handles:
        ax_top.legend(handles=legend_handles, loc=legend_loc, framealpha=0.85)

    if annotation:
        ax_top.text(annotation_loc[0], annotation_loc[1], annotation,
                    transform=ax_top.transAxes,
                    ha='right' if annotation_loc[0] > 0.5 else 'left',
                    va='bottom' if annotation_loc[1] < 0.5 else 'top',
                    family='monospace',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.6'))

    ax_top.set_title(title)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
