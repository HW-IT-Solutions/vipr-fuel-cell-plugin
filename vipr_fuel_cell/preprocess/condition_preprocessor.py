"""Condition selection, validation and scaling for PEMFC inference."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vipr.plugins.discovery.decorators import discover_filter
from vipr.plugins.inference.dataset import DataSet
from vipr_fuel_cell.contracts import PEMFCDatasetContext
from vipr_fuel_cell.load_model.bundle import as_pemfc_bundle


class PEMFCConditionPreprocessorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invalid_rows: Literal["drop", "error"] = Field(default="drop")
    out_of_range: Literal["allow", "warn", "error"] = Field(default="warn")
    range_tolerance: float = Field(default=0.0, ge=0.0)


class PEMFCConditionPreprocessor:
    """Prepare a sensor-condition matrix for the loaded model."""

    def __init__(self, app):
        self.app = app

    @discover_filter(
        "INFERENCE_PREPROCESS_PRE_FILTER",
        enabled_in_config=True,
        parameters=PEMFCConditionPreprocessorParams,
    )
    def preprocess_conditions(self, data: DataSet, **kwargs) -> DataSet:
        params = PEMFCConditionPreprocessorParams.model_validate(kwargs)
        bundle = as_pemfc_bundle(
            getattr(getattr(self.app, "inference", None), "model", None)
        )
        context = PEMFCDatasetContext.model_validate(data.metadata)

        supplied_ids = context.condition_ids
        if len(supplied_ids) != data.x.shape[1]:
            raise ValueError(
                "DataSet condition_ids metadata does not match the condition matrix"
            )
        ordered_descriptors = [
            bundle.conditions[name] for name in bundle.condition_names
        ]
        missing = [
            f"{descriptor.id} ({descriptor.name})"
            for descriptor in ordered_descriptors
            if descriptor.id not in supplied_ids
        ]
        if missing:
            raise ValueError(
                "Loaded model requires condition IDs missing from the DataSet: "
                f"{missing}"
            )
        selected_ids = [descriptor.id for descriptor in ordered_descriptors]
        indices = [supplied_ids.index(identifier) for identifier in selected_ids]
        raw_conditions = np.asarray(data.x[:, indices], dtype=np.float32)
        time = np.asarray(data.y, dtype=np.float64) if data.y is not None else None

        valid = np.all(np.isfinite(raw_conditions), axis=1)
        if time is not None:
            valid &= np.all(np.isfinite(time), axis=1)
        invalid_indices = np.flatnonzero(~valid).tolist()
        if invalid_indices and params.invalid_rows == "error":
            raise ValueError(
                f"PEMFC input contains non-finite values at rows {invalid_indices}"
            )
        if invalid_indices:
            raw_conditions = raw_conditions[valid]
            time = time[valid] if time is not None else None
            self.app.log.warning(
                f"Dropped {len(invalid_indices)} PEMFC time step(s) with non-finite conditions"
            )
        if not len(raw_conditions):
            raise ValueError("No valid PEMFC time steps remain after preprocessing")

        minmax = bundle.condition_scaler.minmax
        tolerance = params.range_tolerance
        below = raw_conditions < (minmax[:, 0] - tolerance)
        above = raw_conditions > (minmax[:, 1] + tolerance)
        out_of_range_mask = below | above
        out_of_range_count = int(np.count_nonzero(out_of_range_mask))
        if out_of_range_count:
            affected = [
                descriptor.id
                for index, descriptor in enumerate(ordered_descriptors)
                if np.any(out_of_range_mask[:, index])
            ]
            message = (
                f"Found {out_of_range_count} PEMFC condition value(s) outside the "
                f"training ranges; affected conditions: {affected}"
            )
            if params.out_of_range == "error":
                raise ValueError(message)
            if params.out_of_range == "warn":
                self.app.log.warning(message)

        scaled = bundle.condition_scaler.transform_numpy(raw_conditions)
        metadata_payload = context.model_dump(mode="python")
        metadata_payload.update(
            {
                "condition_ids": selected_ids,
                "conditions_scaled": True,
                "valid_time_steps": int(len(scaled)),
                "dropped_time_step_indices": invalid_indices,
                "out_of_range_value_count": out_of_range_count,
            }
        )
        metadata = PEMFCDatasetContext.model_validate(metadata_payload).model_dump(
            mode="python"
        )
        return data.copy_with_updates(x=scaled, y=time, metadata=metadata)
