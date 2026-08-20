"""Resolve, verify, and load complete PEMFC cINN model bundles."""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import torch

from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.load_model.cinn_network import PEMFCCINN
from vipr_fuel_cell.load_model.manifest import (
    PEMFCModelManifest,
    VariableDescriptor,
)
from vipr_fuel_cell.load_model.scaler import MinMaxScaler

FUEL_CELL_ROOT_ENV_VAR = "VIPR_FUEL_CELL_ROOT_DIR"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_artifact_directory(model_id: str) -> Path:
    """Resolve a model bundle from the deployment root or source checkout."""
    configured_root = os.getenv(FUEL_CELL_ROOT_ENV_VAR)
    if configured_root:
        return Path(configured_root).expanduser() / "models" / model_id

    repository_directory = _PROJECT_ROOT / "models" / model_id
    if repository_directory.is_dir():
        return repository_directory

    raise FileNotFoundError(
        f"PEMFC model {model_id!r} is not provisioned. Set "
        f"{FUEL_CELL_ROOT_ENV_VAR} to a directory containing "
        f"models/{model_id}/, or configure artifact_dir explicitly."
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(
    manifest: PEMFCModelManifest, artifact_directory: Path
) -> tuple[Path, Path, Path]:
    """Return verified checkpoint, parameter-scaler, and condition-scaler paths."""
    descriptors = (
        manifest.artifacts.checkpoint,
        manifest.artifacts.parameter_scaler,
        manifest.artifacts.condition_scaler,
    )
    paths = tuple(artifact_directory / item.filename for item in descriptors)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"PEMFC model {manifest.id!r} is incomplete in {artifact_directory}; "
            f"missing: {missing}. See models/README.md for provisioning instructions."
        )

    invalid = [
        path.name
        for path, descriptor in zip(paths, descriptors)
        if _file_sha256(path) != descriptor.sha256
    ]
    if invalid:
        raise ValueError(
            f"PEMFC model {manifest.id!r} failed SHA-256 verification: {invalid}"
        )
    return paths


def _descriptor_map(
    *,
    checkpoint_names: list[str],
    descriptors: list[VariableDescriptor],
    label: str,
) -> dict[str, VariableDescriptor]:
    by_name = {descriptor.name: descriptor for descriptor in descriptors}
    missing = sorted(set(checkpoint_names) - by_name.keys())
    extra = sorted(by_name.keys() - set(checkpoint_names))
    if missing or extra:
        raise ValueError(
            f"PEMFC model manifest {label} do not match checkpoint names; "
            f"missing={missing}, extra={extra}"
        )
    return by_name


def _load_bundle_from_paths(
    *,
    checkpoint_path: Path,
    parameter_scaler_path: Path,
    condition_scaler_path: Path,
    device: torch.device,
    manifest: PEMFCModelManifest,
) -> PEMFCCINNBundle:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hyperparams = checkpoint.get("hyper_parameters", checkpoint.get("hparams", {}))
    parameter_names = list(hyperparams.get("input_names", []))
    condition_names = list(hyperparams.get("output_names", []))
    if not parameter_names or not condition_names:
        raise ValueError("Checkpoint does not declare input_names and output_names")

    conditions = _descriptor_map(
        checkpoint_names=condition_names,
        descriptors=manifest.conditions,
        label="conditions",
    )
    parameters = _descriptor_map(
        checkpoint_names=parameter_names,
        descriptors=manifest.targets,
        label="targets",
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
    model.load_state_dict(state_dict, strict=True)
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
        conditions=conditions,
        parameters=parameters,
        device=device,
        model_id=manifest.id,
    )


def load_model_bundle(
    *,
    manifest: PEMFCModelManifest,
    artifact_directory: Path,
    device: torch.device,
) -> PEMFCCINNBundle:
    """Load a complete, verified cINN bundle described by one manifest."""
    checkpoint, parameter_scaler, condition_scaler = verify_artifacts(
        manifest, artifact_directory
    )
    return _load_bundle_from_paths(
        checkpoint_path=checkpoint,
        parameter_scaler_path=parameter_scaler,
        condition_scaler_path=condition_scaler,
        device=device,
        manifest=manifest,
    )
