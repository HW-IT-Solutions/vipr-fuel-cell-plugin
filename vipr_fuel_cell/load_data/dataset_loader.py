"""Load PEMFC sensor profiles through an explicit column mapping."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vipr.plugins.discovery.decorators import discover_data_loader
from vipr.plugins.inference.dataset import DataSet
from vipr.plugins.inference.handlers.data_loader import DataLoaderHandler
from vipr_fuel_cell.contracts import PEMFCDatasetContext
from vipr_fuel_cell.paths import resolve_required_file


class ConditionMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    column: str


class TimeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = "time_s"
    label: str = "Time"
    unit: str = "s"


class PEMFCDatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    title: str
    description: str
    source: dict[str, str] = Field(default_factory=dict)
    time: TimeMetadata = Field(default_factory=TimeMetadata)
    conditions: list[ConditionMapping] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_conditions(self):
        ids = [item.id for item in self.conditions]
        columns = [item.column for item in self.conditions]
        if len(set(ids)) != len(ids):
            raise ValueError("profile condition ids must be unique")
        if len(set(columns)) != len(columns):
            raise ValueError("profile CSV columns must be unique")
        return self


class PEMFCDatasetLoaderParams(BaseModel):
    """Paths for a PEMFC sensor profile and its column mapping."""

    model_config = ConfigDict(extra="forbid")

    data_path: str = Field(
        description="Sensor CSV path, resolved relative to the VIPR config",
    )
    profile_path: str = Field(
        description="Profile YAML path, resolved relative to the VIPR config",
    )


def _read_profile(path: Path) -> PEMFCDatasetProfile:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PEMFCDatasetProfile.model_validate(parsed)


def _read_numeric_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"PEMFC sensor CSV has no header: {path}")
        values = {name: [] for name in reader.fieldnames}
        for row_index, row in enumerate(reader, start=2):
            for name in reader.fieldnames:
                try:
                    values[name].append(float(row[name]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric value in {path.name}, row {row_index}, "
                        f"column {name!r}: {row[name]!r}"
                    ) from exc
    return {
        name: np.asarray(column, dtype=np.float64) for name, column in values.items()
    }


def _build_condition_matrix(
    profile: PEMFCDatasetProfile,
    csv_columns: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Map profile condition IDs to CSV columns in profile-defined order."""
    condition_ids: list[str] = []
    condition_columns: list[np.ndarray] = []
    missing: list[str] = []
    for item in profile.conditions:
        condition_ids.append(item.id)
        if item.column not in csv_columns:
            missing.append(item.column)
        else:
            condition_columns.append(csv_columns[item.column])

    if missing:
        raise ValueError(
            f"PEMFC sensor CSV is missing condition columns {missing}; "
            f"available: {sorted(csv_columns)}"
        )

    matrix = np.column_stack(condition_columns)
    return condition_ids, matrix


@discover_data_loader("pemfc_dataset", PEMFCDatasetLoaderParams)
class PEMFCDatasetLoader(DataLoaderHandler):
    """Load a PEMFC sensor profile and its stable condition identifiers."""

    class Meta:
        label = "pemfc_dataset"

    def _load_data(self, **kwargs) -> DataSet:
        params = PEMFCDatasetLoaderParams.model_validate(kwargs)
        data_path = resolve_required_file(
            self.app, params.data_path, "PEMFC sensor CSV"
        )
        profile_path = resolve_required_file(
            self.app, params.profile_path, "PEMFC dataset profile"
        )
        profile = _read_profile(profile_path)
        columns = _read_numeric_columns(data_path)
        if profile.time.column not in columns:
            raise ValueError(
                f"PEMFC sensor CSV is missing time column {profile.time.column!r}; "
                f"available: {sorted(columns)}"
            )

        time = columns[profile.time.column]
        condition_ids, conditions = _build_condition_matrix(profile, columns)
        metadata = PEMFCDatasetContext(
            dataset_id=profile.id,
            dataset_title=profile.title,
            dataset_description=profile.description,
            dataset_source=profile.source,
            source=str(data_path),
            profile_source=str(profile_path),
            condition_ids=condition_ids,
            time_label=profile.time.label,
            time_unit=profile.time.unit,
            original_time_steps=int(len(time)),
        ).model_dump(mode="python")
        self.app.log.info(
            f"Loaded PEMFC dataset {profile.id!r} from {data_path} with "
            f"{len(time)} time steps and {len(condition_ids)} conditions"
        )
        return DataSet(x=conditions, y=time[:, None], metadata=metadata)
