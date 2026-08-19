"""MATLAB 7.3 loader for the HZwo-DigiTwin acceptance scenarios."""

from __future__ import annotations

from typing import Literal

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vipr.plugins.discovery.decorators import discover_data_loader
from vipr.plugins.inference.dataset import DataSet
from vipr.plugins.inference.handlers.data_loader import DataLoaderHandler
from vipr_fuel_cell.constants import FULL_CONDITION_NAMES
from vipr_fuel_cell.load_data.reference_parser import parse_reference_file
from vipr_fuel_cell.paths import resolve_required_file


class PEMFCMatDataLoaderParams(BaseModel):
    """Configuration for the PEMFC acceptance MAT loader."""

    model_config = ConfigDict(extra="forbid")

    data_path: str = Field(description="MATLAB 7.3 acceptance file")
    condition_names: list[str] = Field(
        default_factory=lambda: list(FULL_CONDITION_NAMES),
        description="Performance signals to read, in their stored order",
    )
    data_group: str = Field(default="out/Data")
    time_name: str = Field(default="tout")
    reference_path: str | None = Field(
        default=None,
        description="Optional legacy INI file containing operating references",
    )
    missing_conditions: Literal["error", "ignore"] = Field(default="error")


def _read_vector(dataset: h5py.Dataset, name: str) -> np.ndarray:
    values = np.asarray(dataset[()], dtype=np.float64).squeeze()
    if values.ndim != 1:
        raise ValueError(
            f"MAT signal {name!r} must become one-dimensional after squeezing; "
            f"got shape {values.shape}"
        )
    return values


def _resolve_group(handle: h5py.File, group_path: str) -> h5py.Group:
    current: h5py.Group | h5py.File = handle
    for part in group_path.strip("/").split("/"):
        if not part:
            continue
        if part not in current:
            raise KeyError(f"MAT group {group_path!r} does not exist")
        current = current[part]
    if not isinstance(current, h5py.Group):
        raise TypeError(f"MAT path {group_path!r} is not a group")
    return current


@discover_data_loader("pemfc_mat", PEMFCMatDataLoaderParams)
class PEMFCMatDataLoader(DataLoaderHandler):
    """Load time-aligned PEMFC performance signals from an acceptance MAT file."""

    class Meta:
        label = "pemfc_mat"

    def _load_data(self, **kwargs) -> DataSet:
        params = PEMFCMatDataLoaderParams.model_validate(kwargs)
        data_path = resolve_required_file(self.app, params.data_path, "PEMFC MAT file")

        with h5py.File(data_path, "r") as handle:
            group = _resolve_group(handle, params.data_group)
            available = list(group.keys())
            missing = [name for name in params.condition_names if name not in group]
            if missing and params.missing_conditions == "error":
                raise ValueError(
                    f"MAT file is missing configured conditions {missing}; "
                    f"available signals are {available}"
                )
            selected_names = [name for name in params.condition_names if name in group]
            if not selected_names:
                raise ValueError(
                    "No configured PEMFC conditions are present in the MAT file"
                )

            vectors = [_read_vector(group[name], name) for name in selected_names]
            lengths = {len(values) for values in vectors}
            if len(lengths) != 1:
                raise ValueError(
                    f"PEMFC conditions have inconsistent lengths: {sorted(lengths)}"
                )

            if params.time_name in group:
                time = _read_vector(group[params.time_name], params.time_name)
            else:
                time = np.arange(len(vectors[0]), dtype=np.float64)
                self.app.log.warning(
                    f"MAT signal {params.time_name!r} not found; using sample indices"
                )
            if len(time) != len(vectors[0]):
                raise ValueError(
                    f"Time axis has {len(time)} values, conditions have {len(vectors[0])}"
                )

        reference_values: dict[str, float] = {}
        reference_source = None
        if params.reference_path:
            reference_file = resolve_required_file(
                self.app, params.reference_path, "PEMFC reference file"
            )
            reference_values = parse_reference_file(reference_file)
            reference_source = str(reference_file)

        conditions = np.column_stack(vectors)
        metadata = {
            "domain": "pemfc",
            "source": str(data_path),
            "data_group": params.data_group,
            "condition_names": selected_names,
            "available_signals": available,
            "time_name": params.time_name,
            "time_unit": "s" if params.time_name == "tout" else "index",
            "reference_values": reference_values,
            "reference_source": reference_source,
            "original_time_steps": int(len(time)),
            "conditions_scaled": False,
        }
        self.app.log.info(
            f"Loaded {len(time)} PEMFC time steps with {len(selected_names)} conditions "
            f"from {data_path.name}"
        )
        # x is the condition matrix. y deliberately carries the aligned time
        # axis so it remains an immutable first-class array rather than a large
        # list in metadata.
        return DataSet(x=conditions, y=time[:, None], metadata=metadata)
