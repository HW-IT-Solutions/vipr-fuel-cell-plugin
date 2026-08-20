"""Metadata carried with PEMFC condition data through the VIPR workflow."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PEMFCDatasetContext(BaseModel):
    """Validated metadata accompanying the condition matrix."""

    # DataSet metadata is an extension boundary shared with other filters.
    # Preserve unknown keys instead of making this plugin own the entire map.
    model_config = ConfigDict(extra="allow")

    domain: Literal["pemfc"] = "pemfc"
    dataset_id: str
    dataset_title: str
    dataset_description: str
    dataset_source: dict[str, str] = Field(default_factory=dict)
    source: str
    profile_source: str
    condition_ids: list[str]
    time_label: str
    time_unit: str
    original_time_steps: int = Field(ge=1)
    conditions_scaled: bool = False
    valid_time_steps: int | None = None
    dropped_time_step_indices: list[int] = Field(default_factory=list)
    out_of_range_value_count: int = 0

    @model_validator(mode="after")
    def validate_condition_ids(self):
        if len(set(self.condition_ids)) != len(self.condition_ids):
            raise ValueError("condition_ids must be unique")
        return self
