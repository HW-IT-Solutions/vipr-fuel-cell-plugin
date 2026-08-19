"""Resolve, verify, and load complete PEMFC cINN model bundles."""

from __future__ import annotations

import os
from hashlib import sha256
from importlib import resources
from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vipr_fuel_cell.contracts import ParameterDescriptor
from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.load_model.model import PEMFCCINN
from vipr_fuel_cell.load_model.scaler import MinMaxScaler

FUEL_CELL_ROOT_ENV_VAR = "VIPR_FUEL_CELL_ROOT_DIR"
DEFAULT_MODEL_ID = "test_case_1"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_FILENAMES = ("checkpoint.ckpt", "scaler_x.json", "scaler_y.json")


class PublicationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    doi: str
    test_case: str


class PEMFCModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    title: str
    description: str
    publication: PublicationMetadata
    targets: list[ParameterDescriptor] = Field(min_length=1)


def _model_metadata_resource(model_id: str):
    root = resources.files("vipr_fuel_cell").joinpath("resources", "models")
    available = sorted(
        child.name for child in root.iterdir() if child.is_dir()
    )
    if model_id not in available:
        raise ValueError(
            f"Unknown PEMFC model {model_id!r}; available: {', '.join(available)}"
        )
    return root.joinpath(model_id)


def _load_model_metadata(model_id: str) -> PEMFCModelMetadata:
    resource = _model_metadata_resource(model_id).joinpath("metadata.yaml")
    if not resource.is_file():
        raise FileNotFoundError(
            f"PEMFC model {model_id!r} has no packaged metadata.yaml"
        )
    metadata = PEMFCModelMetadata.model_validate(
        yaml.safe_load(resource.read_text(encoding="utf-8"))
    )
    if metadata.id != model_id:
        raise ValueError(
            f"PEMFC model metadata id {metadata.id!r} does not match {model_id!r}"
        )
    return metadata


def _artifact_directory(model_id: str) -> Path:
    configured_root = os.getenv(FUEL_CELL_ROOT_ENV_VAR)
    if configured_root:
        return Path(configured_root).expanduser() / "models" / model_id

    repository_directory = _PROJECT_ROOT / "models" / model_id
    if repository_directory.is_dir():
        return repository_directory

    raise FileNotFoundError(
        f"PEMFC model {model_id!r} is not provisioned. Set "
        f"{FUEL_CELL_ROOT_ENV_VAR} to a directory containing "
        f"models/{model_id}/, or provide checkpoint_path explicitly."
    )


def _read_checksums(model_id: str) -> dict[str, str]:
    resource = _model_metadata_resource(model_id).joinpath("checksums.sha256")
    if not resource.is_file():
        raise FileNotFoundError(
            f"PEMFC model {model_id!r} has no packaged checksums.sha256"
        )
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        resource.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(
                f"Invalid checksum entry for {model_id!r} at line {line_number}"
            )
        filename = parts[1].lstrip("*")
        if Path(filename).name != filename:
            raise ValueError(
                f"Checksum entry must contain a filename only: {filename!r}"
            )
        checksums[filename] = parts[0].lower()
    missing = sorted(set(_ARTIFACT_FILENAMES) - checksums.keys())
    extra = sorted(checksums.keys() - set(_ARTIFACT_FILENAMES))
    if missing or extra:
        raise ValueError(
            f"Checksum manifest for {model_id!r} does not match its artifacts; "
            f"missing={missing}, extra={extra}"
        )
    return checksums


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifacts(
    model_id: str, artifact_directory: Path
) -> tuple[Path, Path, Path]:
    paths = tuple(artifact_directory / name for name in _ARTIFACT_FILENAMES)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        root_hint = os.getenv(FUEL_CELL_ROOT_ENV_VAR)
        location_hint = (
            f" under {FUEL_CELL_ROOT_ENV_VAR}={root_hint!r}"
            if root_hint
            else " in the repository-local models directory"
        )
        raise FileNotFoundError(
            f"PEMFC model {model_id!r} is incomplete{location_hint}; missing: {missing}. "
            "See models/README.md for provisioning instructions."
        )

    checksums = _read_checksums(model_id)
    invalid = [
        path.name
        for path in paths
        if _file_sha256(path) != checksums[path.name]
    ]
    if invalid:
        raise ValueError(
            f"PEMFC model {model_id!r} failed SHA-256 verification: {invalid}"
        )
    return paths


def _load_bundle_from_paths(
    *,
    checkpoint_path: Path,
    parameter_scaler_path: Path,
    condition_scaler_path: Path,
    device: torch.device,
    strict: bool,
    model_id: str | None,
    metadata: PEMFCModelMetadata | None,
) -> PEMFCCINNBundle:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hyperparams = checkpoint.get("hyper_parameters", checkpoint.get("hparams", {}))
    parameter_names = list(hyperparams.get("input_names", []))
    condition_names = list(hyperparams.get("output_names", []))
    if not parameter_names or not condition_names:
        raise ValueError("Checkpoint does not declare input_names and output_names")

    if metadata is None:
        descriptors = {
            name: ParameterDescriptor(name=name, id=name, label=name, unit="")
            for name in parameter_names
        }
    else:
        descriptors = {target.name: target for target in metadata.targets}
        if len(descriptors) != len(metadata.targets):
            raise ValueError(f"PEMFC model {metadata.id!r} has duplicate target names")
        missing = sorted(set(parameter_names) - descriptors.keys())
        extra = sorted(descriptors.keys() - set(parameter_names))
        if missing or extra:
            raise ValueError(
                f"PEMFC model metadata does not cover checkpoint input_names; "
                f"missing={missing}, extra={extra}"
            )

    model = PEMFCCINN(
        parameter_names=parameter_names,
        condition_names=condition_names,
        n_blocks=int(hyperparams.get("n_blocks", 10)),
        subnet_hidden_size=int(hyperparams.get("subnet_hidden_size", 512)),
    ).to(device)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint contains no model state_dict")
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    parameter_scaler = MinMaxScaler.from_json(parameter_scaler_path)
    condition_scaler = MinMaxScaler.from_json(condition_scaler_path)
    if parameter_scaler.n_features != len(parameter_names):
        raise ValueError(
            "Parameter scaler dimension does not match checkpoint input_names"
        )
    if condition_scaler.n_features != len(condition_names):
        raise ValueError(
            "Condition scaler dimension does not match checkpoint output_names"
        )

    return PEMFCCINNBundle(
        model=model,
        condition_scaler=condition_scaler,
        parameter_scaler=parameter_scaler,
        condition_names=condition_names,
        parameter_names=parameter_names,
        parameters=descriptors,
        device=device,
        checkpoint_path=str(checkpoint_path),
        model_id=model_id,
    )


def load_bundled_model(
    model_id: str, device: torch.device, strict: bool
) -> PEMFCCINNBundle:
    """Load one named model after validating its package metadata and artifacts."""
    metadata = _load_model_metadata(model_id)
    checkpoint, parameter_scaler, condition_scaler = _verify_artifacts(
        model_id, _artifact_directory(model_id)
    )
    return _load_bundle_from_paths(
        checkpoint_path=checkpoint,
        parameter_scaler_path=parameter_scaler,
        condition_scaler_path=condition_scaler,
        device=device,
        strict=strict,
        model_id=model_id,
        metadata=metadata,
    )


def load_custom_model(
    *,
    checkpoint_path: Path,
    parameter_scaler_path: Path,
    condition_scaler_path: Path,
    device: torch.device,
    strict: bool,
) -> PEMFCCINNBundle:
    """Load explicitly supplied artifacts without imposing a packaged model ID."""
    for label, path in (
        ("checkpoint", checkpoint_path),
        ("parameter scaler", parameter_scaler_path),
        ("condition scaler", condition_scaler_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"PEMFC {label} not found: {path}")
    return _load_bundle_from_paths(
        checkpoint_path=checkpoint_path,
        parameter_scaler_path=parameter_scaler_path,
        condition_scaler_path=condition_scaler_path,
        device=device,
        strict=strict,
        model_id=None,
        metadata=None,
    )
