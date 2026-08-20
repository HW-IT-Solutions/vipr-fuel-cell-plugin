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


class ProfileColumn(BaseModel):
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
    columns: list[ProfileColumn] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_columns(self):
        ids = [item.id for item in self.columns]
        columns = [item.column for item in self.columns]
        if len(set(ids)) != len(ids):
            raise ValueError("profile column ids must be unique")
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


@discover_data_loader("pemfc_dataset", PEMFCDatasetLoaderParams)
class PEMFCDatasetLoader(DataLoaderHandler):
    """Load a PEMFC sensor profile and its stable signal identifiers."""

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
        required_columns = [profile.time.column] + [
            item.column for item in profile.columns
        ]
        missing = [name for name in required_columns if name not in columns]
        if missing:
            raise ValueError(
                f"PEMFC sensor CSV is missing profile columns {missing}; "
                f"available: {sorted(columns)}"
            )

        signal_ids = [item.id for item in profile.columns]
        conditions = np.column_stack(
            [columns[item.column] for item in profile.columns]
        )
        time = columns[profile.time.column]
        metadata = PEMFCDatasetContext(
            dataset_id=profile.id,
            dataset_title=profile.title,
            dataset_description=profile.description,
            dataset_source=profile.source,
            source=str(data_path),
            profile_source=str(profile_path),
            signal_ids=signal_ids,
            time_label=profile.time.label,
            time_unit=profile.time.unit,
            original_time_steps=int(len(time)),
        ).model_dump(mode="python")
        self.app.log.info(
            f"Loaded PEMFC dataset {profile.id!r} from {data_path} with "
            f"{len(time)} time steps and {len(signal_ids)} signals"
        )
        return DataSet(x=conditions, y=time[:, None], metadata=metadata)
