"""Built-in datasets and locally provisioned models used by the PEMFC plugin."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


DATASETS = {
    "operating_profile": "resources/datasets/operating_profile",
}

MODELS = {
    "test_case_1": "test_case_1",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resource_directory(catalog: dict[str, str], resource_id: str, kind: str) -> Path:
    try:
        relative_path = catalog[resource_id]
    except KeyError as exc:
        available = ", ".join(sorted(catalog))
        raise ValueError(
            f"Unknown built-in PEMFC {kind} {resource_id!r}; available: {available}"
        ) from exc

    directory = Path(str(resources.files("vipr_fuel_cell").joinpath(relative_path)))
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Built-in PEMFC {kind} {resource_id!r} is not installed: {directory}"
        )
    return directory


def dataset_directory(dataset_id: str) -> Path:
    """Resolve a built-in dataset identifier to its package directory."""
    return _resource_directory(DATASETS, dataset_id, "dataset")


def model_directory(model_id: str) -> Path:
    """Resolve a model identifier to the ignored repository-local model folder."""
    try:
        relative_path = MODELS[model_id]
    except KeyError as exc:
        available = ", ".join(sorted(MODELS))
        raise ValueError(
            f"Unknown local PEMFC model {model_id!r}; available: {available}"
        ) from exc
    return PROJECT_ROOT / "models" / relative_path
