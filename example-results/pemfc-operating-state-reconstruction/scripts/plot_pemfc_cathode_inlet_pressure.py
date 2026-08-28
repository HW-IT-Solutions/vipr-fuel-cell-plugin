#!/usr/bin/env python3
"""Standalone plot — run with: python plot_pemfc_cathode_inlet_pressure.py [-o output.svg]

No VIPR installation required.
Dependencies: matplotlib>=3.5.0, numpy>=1.21.0
"""

from __future__ import annotations

__dependencies__ = ['matplotlib>=3.5.0', 'numpy>=1.21.0']

# ---------------------------------------------------------------------------
# Exported VIPR style snapshot (editable)
# ---------------------------------------------------------------------------
VIPR_STYLE = {'plot_id': 'pemfc_cathode_inlet_pressure', 'format': 'svg', 'dpi': 150, 'figsize': None, 'rc': {}}

def _vipr_default_figsize(default=(10, 6)):
    figsize = VIPR_STYLE.get('figsize')
    if isinstance(figsize, (list, tuple)) and len(figsize) == 2:
        return float(figsize[0]), float(figsize[1])
    return default

def apply_vipr_style(fig) -> None:
    """Apply exported VIPR style (figsize + text/tick sizes) to a figure."""
    figsize = VIPR_STYLE.get('figsize')
    if isinstance(figsize, (list, tuple)) and len(figsize) == 2:
        fig.set_size_inches(float(figsize[0]), float(figsize[1]))

    rc = VIPR_STYLE.get('rc') or {}
    axes_labelsize = rc.get('axes.labelsize', rc.get('font.size'))
    axes_titlesize = rc.get('axes.titlesize')
    xtick_labelsize = rc.get('xtick.labelsize', rc.get('font.size'))
    ytick_labelsize = rc.get('ytick.labelsize', rc.get('font.size'))
    legend_fontsize = rc.get('legend.fontsize')

    for ax in fig.get_axes():
        if xtick_labelsize is not None:
            ax.tick_params(axis='x', labelsize=xtick_labelsize)
        if ytick_labelsize is not None:
            ax.tick_params(axis='y', labelsize=ytick_labelsize)
        if axes_labelsize is not None:
            ax.xaxis.label.set_size(axes_labelsize)
            ax.yaxis.label.set_size(axes_labelsize)
        if axes_titlesize is not None and ax.get_title():
            ax.title.set_size(axes_titlesize)
        legend = ax.get_legend()
        if legend is not None and legend_fontsize is not None:
            for text in legend.get_texts():
                text.set_fontsize(legend_fontsize)

def save_vipr_styled_figure(fig, output_path: str) -> None:
    """Save figure using VIPR-exported DPI while keeping user output format."""
    dpi = VIPR_STYLE.get('dpi', 150)
    fig.savefig(output_path, bbox_inches='tight', dpi=dpi)


import csv
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Series definitions — each series is stored in its own CSV file.
# The CSV files live in ../diagrams/ relative to this script.
# Customise labels, colours, etc. freely.
# ---------------------------------------------------------------------------
SERIES = [{'csv': 'pemfc_cathode_inlet_pressure_Posterior_mean.csv', 'x': 'time', 'y': 'posterior_mean', 'err': '', 'x_err': 'time_error', 'label': 'Posterior mean'}]


def load_series(csv_path: Path, x_col: str, y_col: str, err_col: str, x_err_col: str) -> tuple:
    """Load one series from a CSV file.

    Returns (x, y, xerr_or_None, yerr_or_None).
    """
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    x = [float(r[x_col]) for r in rows]
    y = [float(r[y_col]) for r in rows]
    xerr = None
    yerr = None
    if x_err_col and rows and x_err_col in rows[0]:
        xerr = [float(r[x_err_col]) for r in rows]
    if err_col and rows and err_col in rows[0]:
        yerr = [float(r[err_col]) for r in rows]
    return x, y, xerr, yerr


def make_plot(title: str = 'Cathode inlet pressure') -> plt.Figure:
    """Create the combined plot from all series CSVs.  Customise freely."""
    data_dir = Path(__file__).parent.parent / 'diagrams'
    fig, ax = plt.subplots(figsize=_vipr_default_figsize((10, 6)))

    for s in SERIES:
        csv_path = data_dir / s['csv']
        if not csv_path.exists():
            print(f'Warning: {csv_path} not found, skipping series "{s["label"]}".')
            continue
        x_err_col = s.get('x_err', '')
        x, y, xerr, yerr = load_series(csv_path, s['x'], s['y'], s['err'], x_err_col)
        if xerr is not None or yerr is not None:
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='-o', label=s['label'], capsize=3)
        else:
            ax.plot(x, y, '-o', label=s['label'])

    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Cathode inlet pressure [atm]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    apply_vipr_style(fig)
    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot VIPR diagram standalone.')
    parser.add_argument('-o', '--output', default=None,
                        help='Save figure to file instead of showing')
    args = parser.parse_args()

    fig = make_plot()
    if args.output:
        save_vipr_styled_figure(fig, args.output)
        print(f'Saved: {args.output}')
    else:
        plt.show()
