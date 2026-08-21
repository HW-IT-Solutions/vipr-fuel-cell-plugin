"""Convert a MATLAB 7.3 PEMFC result to the public sensor-profile CSV schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


COLUMNS = {
    "tout": "time_s",
    "U_cell_V": "cell_voltage_V",
    "E_Con": "concentration_loss_V",
    "E_Ohm": "ohmic_loss_V",
    "E_Cross": "crossover_loss_V",
    "E_act": "activation_loss_V",
    "Imax_An_fit": "anode_limiting_current_density_A_cm2",
    "Imax_Kath_fit": "cathode_limiting_current_density_A_cm2",
    "N_flux_H2O_th_An": "anode_water_flux_mol_m2_s",
    "N_flux_H2O_th_Kath": "cathode_water_flux_mol_m2_s",
    "N_flux_H2_th_An": "anode_hydrogen_flux_mol_m2_s",
    "N_flux_O2_th_Kath": "cathode_oxygen_flux_mol_m2_s",
}


def convert(source: Path, destination: Path) -> None:
    """Convert model signals and add their zero-based simulation steps."""
    with h5py.File(source, "r") as handle:
        group = handle["out/Data"]
        values = {
            public_name: np.asarray(group[stored_name][()], dtype=np.float64).squeeze()
            for stored_name, public_name in COLUMNS.items()
        }

    lengths = {len(column) for column in values.values()}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent sensor column lengths: {sorted(lengths)}")

    sample_count = lengths.pop()
    values = {
        "simulation_step": np.arange(sample_count, dtype=np.int64),
        **values,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(values)
        for row in zip(*values.values()):
            writer.writerow([format(float(value), ".17g") for value in row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
