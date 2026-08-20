"""Schema and parser for PEMFC model manifests."""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class VariableDescriptor(BaseModel):
    """Semantic description of a model condition or reconstructed target."""

    model_config = ConfigDict(extra="forbid")

    name: str
    id: str
    label: str
    unit: str


class PublicationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    doi: str
    test_case: str


class ArtifactDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    sha256: str

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError("artifact filename must be a basename")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters")
        return value.lower()


class ModelArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: ArtifactDescriptor
    parameter_scaler: ArtifactDescriptor
    condition_scaler: ArtifactDescriptor

    @model_validator(mode="after")
    def validate_unique_filenames(self):
        filenames = [
            self.checkpoint.filename,
            self.parameter_scaler.filename,
            self.condition_scaler.filename,
        ]
        if len(set(filenames)) != len(filenames):
            raise ValueError("model artifact filenames must be unique")
        return self


class PEMFCModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    title: str
    description: str
    publication: PublicationMetadata
    conditions: list[VariableDescriptor] = Field(min_length=1)
    targets: list[VariableDescriptor] = Field(min_length=1)
    artifacts: ModelArtifacts

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError(
                "model id must contain lowercase letters, digits, underscores, "
                "or hyphens and must start with a letter or digit"
            )
        return value

    @model_validator(mode="after")
    def validate_descriptors(self):
        for label, descriptors in (
            ("condition", self.conditions),
            ("target", self.targets),
        ):
            names = [descriptor.name for descriptor in descriptors]
            ids = [descriptor.id for descriptor in descriptors]
            if len(set(names)) != len(names):
                raise ValueError(f"model manifest has duplicate {label} names")
            if len(set(ids)) != len(ids):
                raise ValueError(f"model manifest has duplicate {label} ids")
            invalid_ids = sorted(
                value for value in ids if not _SAFE_ID.fullmatch(value)
            )
            if invalid_ids:
                raise ValueError(
                    f"model manifest has invalid {label} ids: {invalid_ids}"
                )
        return self


def load_model_manifest(path: Path) -> PEMFCModelManifest:
    """Load and validate a semantic model and artifact description."""
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PEMFCModelManifest.model_validate(parsed)
