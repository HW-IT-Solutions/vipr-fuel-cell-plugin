"""Load curated PEMFC sensor profiles with explicit English metadata."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vipr.plugins.discovery.decorators import discover_data_loader
from vipr.plugins.inference.dataset import DataSet
from vipr.plugins.inference.handlers.data_loader import DataLoaderHandler
from vipr_fuel_cell.paths import resolve_required_file
from vipr_fuel_cell.resource_catalog import dataset_directory


class SignalMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    column: str
    label: str
    unit: str


class ReferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    label: str
    value: float
    unit: str


class TimeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = "time_s"
    label: str = "Time"
    unit: str = "s"


class PEMFCDatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    title: str
    description: str
    source: dict[str, str] = Field(default_factory=dict)
    time: TimeMetadata = Field(default_factory=TimeMetadata)
    conditions: list[SignalMetadata]
    references: list[ReferenceMetadata] = Field(default_factory=list)


class PEMFCDatasetLoaderParams(BaseModel):
    """Configuration for a curated or user-provided PEMFC CSV dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset: str | None = Field(
        default=None,
        description="Built-in dataset identifier; defaults to operating_profile",
    )
    data_path: str | None = Field(
        default=None,
        description="Optional custom sensor CSV, resolved relative to the VIPR config",
    )
    metadata_path: str | None = Field(
        default=None,
        description="Optional metadata YAML for a custom sensor CSV",
    )


def _load_metadata(path: Path) -> PEMFCDatasetMetadata:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PEMFCDatasetMetadata.model_validate(parsed)


def _load_columns(path: Path) -> dict[str, np.ndarray]:
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


def _resolve_dataset_files(app, params: PEMFCDatasetLoaderParams) -> tuple[Path, Path]:
    if params.data_path:
        data_path = resolve_required_file(app, params.data_path, "PEMFC sensor CSV")
        if params.metadata_path:
            metadata_path = resolve_required_file(
                app, params.metadata_path, "PEMFC dataset metadata"
            )
        else:
            metadata_path = data_path.with_name("metadata.yaml")
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    "Custom PEMFC datasets require metadata_path or a metadata.yaml "
                    f"next to the CSV: {metadata_path}"
                )
        return data_path, metadata_path

    resource_dir = dataset_directory(params.dataset or "operating_profile")
    return resource_dir / "sensor_data.csv", resource_dir / "metadata.yaml"


@discover_data_loader("pemfc_dataset", PEMFCDatasetLoaderParams)
class PEMFCDatasetLoader(DataLoaderHandler):
    """Load a PEMFC sensor profile and its domain metadata."""

    class Meta:
        label = "pemfc_dataset"

    def _load_data(self, **kwargs) -> DataSet:
        params = PEMFCDatasetLoaderParams.model_validate(kwargs)
        data_path, metadata_path = _resolve_dataset_files(self.app, params)
        for label, path in (
            ("sensor CSV", data_path),
            ("metadata YAML", metadata_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"PEMFC {label} not found: {path}")

        metadata_definition = _load_metadata(metadata_path)
        columns = _load_columns(data_path)
        required_columns = [metadata_definition.time.column] + [
            condition.column for condition in metadata_definition.conditions
        ]
        missing = [name for name in required_columns if name not in columns]
        if missing:
            raise ValueError(
                f"PEMFC sensor CSV is missing metadata columns {missing}; "
                f"available: {sorted(columns)}"
            )

        condition_names = [
            condition.name for condition in metadata_definition.conditions
        ]
        conditions = np.column_stack(
            [columns[condition.column] for condition in metadata_definition.conditions]
        )
        time = columns[metadata_definition.time.column]
        references = {
            reference.name: reference.value
            for reference in metadata_definition.references
        }
        parameter_labels = {
            reference.name: reference.label
            for reference in metadata_definition.references
        }
        parameter_units = {
            reference.name: reference.unit
            for reference in metadata_definition.references
        }

        metadata = {
            "domain": "pemfc",
            "dataset_id": metadata_definition.id,
            "dataset_title": metadata_definition.title,
            "dataset_description": metadata_definition.description,
            "dataset_source": metadata_definition.source,
            "source": str(data_path),
            "metadata_source": str(metadata_path),
            "condition_names": condition_names,
            "condition_labels": {
                condition.name: condition.label
                for condition in metadata_definition.conditions
            },
            "condition_units": {
                condition.name: condition.unit
                for condition in metadata_definition.conditions
            },
            "time_name": metadata_definition.time.label,
            "time_unit": metadata_definition.time.unit,
            "reference_values": references,
            "parameter_labels": parameter_labels,
            "parameter_units": parameter_units,
            "original_time_steps": int(len(time)),
            "conditions_scaled": False,
        }
        self.app.log.info(
            f"Loaded built-in PEMFC dataset {metadata_definition.id!r} with "
            f"{len(time)} time steps and {len(condition_names)} conditions"
        )
        return DataSet(x=conditions, y=time[:, None], metadata=metadata)
