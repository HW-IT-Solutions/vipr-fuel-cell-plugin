#!/usr/bin/env python3
"""Recreate the empirical PEMFC posterior snapshot exported by VIPR.

The adjacent ``*_data.npz`` file contains the histogram bins, densities,
posterior means, parameter labels and snapshot metadata. No VIPR installation
is required to run the exported copy of this script.
"""

from __future__ import annotations

__dependencies__ = ["matplotlib>=3.10.0", "numpy>=1.26"]

import argparse
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


def find_data_file() -> Path:
    """Return the NPZ file exported next to this standalone script."""
    script_path = Path(__file__)
    base = script_path.stem.removeprefix("plot_")
    candidate = script_path.parent / f"{base}_data.npz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"No data file found for {script_path.name!r}. Expected: {candidate}"
    )


def load_data(npz_path: Path | None = None) -> dict[str, np.ndarray]:
    """Load the exported posterior histogram data."""
    path = npz_path or find_data_file()
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _scalar(value: np.ndarray):
    return np.asarray(value).reshape(-1)[0].item()


def make_plot(
    *,
    bin_edges=None,
    densities=None,
    posterior_means=None,
    parameter_labels=None,
    parameter_units=None,
    valid_time_step_index=None,
    time_value=None,
    time_label=None,
    time_unit=None,
):
    """Create the 3 x 3 empirical marginal-posterior figure."""
    if bin_edges is None:
        exported = load_data()
        bin_edges = exported["bin_edges"]
        densities = exported["densities"]
        posterior_means = exported["posterior_means"]
        parameter_labels = exported["parameter_labels"]
        parameter_units = exported["parameter_units"]
        valid_time_step_index = _scalar(exported["valid_time_step_index"])
        time_value = _scalar(exported["time_value"])
        time_label = _scalar(exported["time_label"])
        time_unit = _scalar(exported["time_unit"])

    valid_time_step_index = _scalar(np.asarray(valid_time_step_index))
    time_value = _scalar(np.asarray(time_value))
    time_label = str(_scalar(np.asarray(time_label)))
    time_unit = str(_scalar(np.asarray(time_unit)))
    axis_value = f"{time_label} {float(time_value):g}"
    if time_unit:
        axis_value += f" {time_unit}"

    edges_array = np.asarray(bin_edges, dtype=float)
    density_array = np.asarray(densities, dtype=float)
    means_array = np.asarray(posterior_means, dtype=float)
    labels = [str(value) for value in np.asarray(parameter_labels)]
    units = [str(value) for value in np.asarray(parameter_units)]
    if edges_array.shape[0] != len(labels) or density_array.shape[0] != len(labels):
        raise ValueError("Histogram arrays must contain one row per parameter")
    if edges_array.shape[1] != density_array.shape[1] + 1:
        raise ValueError("Each histogram needs one more bin edge than density values")

    column_count = 3
    row_count = ceil(len(labels) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(12, 3.2 * row_count),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for parameter_index, label in enumerate(labels):
        axis = flat_axes[parameter_index]
        edges = edges_array[parameter_index]
        density = density_array[parameter_index]
        axis.bar(
            edges[:-1],
            density,
            width=np.diff(edges),
            align="edge",
            color="#4C92C3",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.35,
        )
        axis.axvline(
            means_array[parameter_index],
            color="#C53B3B",
            linestyle="--",
            linewidth=1.5,
        )
        axis.set_title(label, fontsize=10)
        axis.set_xlabel(f"Value [{units[parameter_index]}]")
        if parameter_index % column_count == 0:
            axis.set_ylabel("Probability density")
        axis.grid(alpha=0.2, linewidth=0.5)

    for axis in flat_axes[len(labels) :]:
        axis.set_visible(False)

    fig.suptitle(
        f"Empirical PEMFC posterior distributions\n{axis_value}",
        fontsize=13,
    )
    fig.legend(
        handles=[
            Patch(facecolor="#4C92C3", alpha=0.75, label="Posterior samples"),
            Line2D(
                [0],
                [0],
                color="#C53B3B",
                linestyle="--",
                linewidth=1.5,
                label="Posterior mean",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a VIPR PEMFC posterior snapshot."
    )
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    fig = make_plot()
    if args.output:
        fig.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Saved: {args.output}")
    else:
        plt.show()
