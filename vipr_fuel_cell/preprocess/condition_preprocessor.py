"""Condition selection, validation and scaling for PEMFC inference."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vipr.plugins.discovery.decorators import discover_filter
from vipr.plugins.inference.dataset import DataSet
from vipr_fuel_cell.load_data.dataset_context import PEMFCDatasetContext
from vipr_fuel_cell.load_model.bundle import as_pemfc_bundle


class PEMFCConditionPreprocessorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invalid_rows: Literal["drop", "error"] = Field(default="drop")
    out_of_range: Literal["allow", "warn", "error"] = Field(default="warn")
    range_tolerance: float = Field(default=0.0, ge=0.0)


def _select_and_order_conditions(data, context, bundle):
    supplied_ids = context.condition_ids
    if len(supplied_ids) != data.x.shape[1]:
        raise ValueError(
            "DataSet condition_ids metadata does not match the condition matrix"
        )

    descriptors = [
        bundle.conditions[name] for name in bundle.condition_names
    ]
    missing = [
        f"{descriptor.id} ({descriptor.name})"
        for descriptor in descriptors
        if descriptor.id not in supplied_ids
    ]
    if missing:
        raise ValueError(
            "Loaded model requires condition IDs missing from the DataSet: "
            f"{missing}"
        )

    selected_ids = [descriptor.id for descriptor in descriptors]
    indices = [supplied_ids.index(identifier) for identifier in selected_ids]
    conditions = np.asarray(data.x[:, indices], dtype=np.float32)
    return conditions, selected_ids


def _handle_invalid_rows(conditions, time, policy, log):
    valid = np.all(np.isfinite(conditions), axis=1)
    if time is not None:
        valid &= np.all(np.isfinite(time), axis=1)
    invalid_indices = np.flatnonzero(~valid).tolist()

    if invalid_indices and policy == "error":
        raise ValueError(
            f"PEMFC input contains non-finite values at rows {invalid_indices}"
        )
    if invalid_indices:
        conditions = conditions[valid]
        time = time[valid] if time is not None else None
        log.warning(
            f"Dropped {len(invalid_indices)} PEMFC time step(s) "
            "with non-finite conditions"
        )
    if not len(conditions):
        raise ValueError("No valid PEMFC time steps remain after preprocessing")

    return conditions, time, invalid_indices


def _check_training_ranges(
    conditions, condition_ids, minmax, policy, tolerance, log
):
    below = conditions < (minmax[:, 0] - tolerance)
    above = conditions > (minmax[:, 1] + tolerance)
    out_of_range = below | above
    out_of_range_value_count = int(np.count_nonzero(out_of_range))
    if not out_of_range_value_count:
        return 0

    affected = [
        condition_id
        for index, condition_id in enumerate(condition_ids)
        if np.any(out_of_range[:, index])
    ]
    message = (
        f"Found {out_of_range_value_count} PEMFC condition value(s) outside the "
        f"training ranges; affected conditions: {affected}"
    )
    if policy == "error":
        raise ValueError(message)
    if policy == "warn":
        log.warning(message)
    return out_of_range_value_count


def _build_output_metadata(
    context,
    selected_ids,
    valid_time_steps,
    invalid_indices,
    out_of_range_value_count,
):
    payload = context.model_dump(mode="python")
    payload.update(
        {
            "condition_ids": selected_ids,
            "conditions_scaled": True,
            "valid_time_steps": valid_time_steps,
            "dropped_time_step_indices": invalid_indices,
            "out_of_range_value_count": out_of_range_value_count,
        }
    )
    return PEMFCDatasetContext.model_validate(payload).model_dump(mode="python")


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
        # Validate external inputs at the VIPR filter boundary.
        params = PEMFCConditionPreprocessorParams.model_validate(kwargs)
        bundle = as_pemfc_bundle(
            getattr(getattr(self.app, "inference", None), "model", None)
        )
        context = PEMFCDatasetContext.model_validate(data.metadata)

        # Map profile IDs to the exact condition order stored in the checkpoint.
        conditions, selected_ids = _select_and_order_conditions(
            data, context, bundle
        )
        time = np.asarray(data.y, dtype=np.float64) if data.y is not None else None

        # Remove rows containing NaN or infinity from conditions and time together.
        conditions, time, invalid_indices = _handle_invalid_rows(
            conditions, time, params.invalid_rows, self.app.log
        )

        # Scaler bounds use raw input units, so check values before scaling.
        out_of_range_value_count = _check_training_ranges(
            conditions,
            selected_ids,
            bundle.condition_scaler.minmax,
            params.out_of_range,
            params.range_tolerance,
            self.app.log,
        )

        scaled = bundle.condition_scaler.transform_numpy(conditions)

        # Record preprocessing decisions for prediction and result collection.
        metadata = _build_output_metadata(
            context,
            selected_ids,
            len(scaled),
            invalid_indices,
            out_of_range_value_count,
        )
        return data.copy_with_updates(x=scaled, y=time, metadata=metadata)
