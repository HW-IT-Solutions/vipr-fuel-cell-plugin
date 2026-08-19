"""Typed contracts at the VIPR boundaries of the PEMFC workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def quantile_key(value: float) -> str:
    """Return the stable dictionary key used for a posterior quantile."""
    return format(value, ".10g")


class ParameterDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    id: str
    label: str
    unit: str


class PEMFCDatasetContext(BaseModel):
    """Lightweight metadata passed alongside the condition matrix."""

    # DataSet metadata is an extension boundary shared with other filters.
    # Preserve unknown keys instead of making this plugin own the entire map.
    model_config = ConfigDict(extra="allow")

    domain: Literal["pemfc"] = "pemfc"
    dataset_id: str
    dataset_title: str
    dataset_description: str
    dataset_source: dict[str, str] = Field(default_factory=dict)
    source: str
    metadata_source: str
    condition_names: list[str]
    condition_labels: dict[str, str]
    condition_units: dict[str, str]
    time_label: str
    time_unit: str
    reference_values: dict[str, float] = Field(default_factory=dict)
    original_time_steps: int = Field(ge=1)
    conditions_scaled: bool = False
    valid_time_steps: int | None = None
    dropped_time_step_indices: list[int] = Field(default_factory=list)
    out_of_range_value_count: int = 0

    @model_validator(mode="after")
    def validate_condition_metadata(self):
        names = set(self.condition_names)
        if len(names) != len(self.condition_names):
            raise ValueError("condition_names must be unique")
        if set(self.condition_labels) != names or set(self.condition_units) != names:
            raise ValueError(
                "condition_labels and condition_units must cover condition_names exactly"
            )
        return self


class PosteriorParameterStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: list[float]
    std: list[float]
    min: list[float]
    max: list[float]
    quantiles: dict[str, list[float]]


class PEMFCPosteriorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_title: str
    dataset_source: dict[str, str] = Field(default_factory=dict)
    model_id: str | None = None
    num_samples: int
    seed: int
    quantiles: list[float]
    common_latent_samples: bool
    inference_seconds: float
    valid_time_steps: int = Field(ge=1)
    dropped_time_step_indices: list[int] = Field(default_factory=list)
    out_of_range_value_count: int = 0

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, values: list[float]) -> list[float]:
        normalized = [float(value) for value in values]
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("quantiles must contain unique values")
        if any(value < 0.0 or value > 1.0 for value in normalized):
            raise ValueError("quantiles must be in [0, 1]")
        return normalized


class PEMFCPosteriorResult(BaseModel):
    """Validated result returned to VIPR and consumed by the collector."""

    model_config = ConfigDict(extra="forbid")

    prediction_type: Literal["pemfc_cinn_posterior"] = "pemfc_cinn_posterior"
    time: list[float]
    time_label: str
    time_unit: str
    parameter_names: list[str]
    parameters: dict[str, ParameterDescriptor]
    statistics: dict[str, PosteriorParameterStatistics]
    reference_values: dict[str, float] = Field(default_factory=dict)
    metadata: PEMFCPosteriorMetadata

    @model_validator(mode="after")
    def validate_parameter_payload(self):
        names = set(self.parameter_names)
        if len(names) != len(self.parameter_names):
            raise ValueError("parameter_names must be unique")
        if set(self.parameters) != names or set(self.statistics) != names:
            raise ValueError(
                "parameters and statistics must cover parameter_names exactly"
            )
        expected_length = len(self.time)
        expected_quantiles = {
            quantile_key(value) for value in self.metadata.quantiles
        }
        for name, statistics in self.statistics.items():
            actual_quantiles = set(statistics.quantiles)
            if actual_quantiles != expected_quantiles:
                raise ValueError(
                    f"Posterior quantiles for {name!r} do not match metadata; "
                    f"expected={sorted(expected_quantiles)}, "
                    f"actual={sorted(actual_quantiles)}"
                )
            series = {
                "mean": statistics.mean,
                "std": statistics.std,
                "min": statistics.min,
                "max": statistics.max,
                **{
                    f"quantile {key}": values
                    for key, values in statistics.quantiles.items()
                },
            }
            invalid = [
                label for label, values in series.items() if len(values) != expected_length
            ]
            if invalid:
                raise ValueError(
                    f"Posterior series for {name!r} do not match the time axis: {invalid}"
                )
        return self

    def as_vipr_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python")
