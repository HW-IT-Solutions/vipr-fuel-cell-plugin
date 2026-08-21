"""Direct VIPR model loader for manifest-defined PEMFC cINN bundles."""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field

from vipr.plugins.discovery.decorators import discover_model_loader
from vipr.plugins.inference.handlers.model_loader import ModelLoaderHandler
from vipr_fuel_cell.load_model.artifacts import load_model_bundle
from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.load_model.manifest import load_model_manifest
from vipr_fuel_cell.paths import resolve_model_directory


class PEMFCCINNModelLoaderParams(BaseModel):
    """Paths and runtime options for a complete cINN model bundle."""

    model_config = ConfigDict(extra="forbid")

    model_dir: str = Field(
        description=(
            "Directory containing model.yaml, the checkpoint, and both scalers"
        ),
    )
    device: str = Field(default="cpu")


def _select_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for the PEMFC cINN but is not available")
    return device


@discover_model_loader("pemfc_cinn", PEMFCCINNModelLoaderParams)
class PEMFCCINNModelLoader(ModelLoaderHandler):
    """Load a verified cINN bundle without a registry."""

    class Meta:
        label = "pemfc_cinn"

    def _load_model(self, **kwargs) -> PEMFCCINNBundle:
        params = PEMFCCINNModelLoaderParams.model_validate(kwargs)
        device = _select_device(params.device)
        model_directory = resolve_model_directory(self.app, params.model_dir)
        manifest_path = model_directory / "model.yaml"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"PEMFC model manifest not found: {manifest_path}")
        manifest = load_model_manifest(manifest_path)

        bundle = load_model_bundle(
            manifest=manifest,
            model_directory=model_directory,
            device=device,
        )

        self.app.log.info(
            f"Loaded PEMFC cINN {manifest.id!r} with "
            f"{len(bundle.condition_names)} conditions and "
            f"{len(bundle.parameter_names)} reconstructed parameters from "
            f"{model_directory} on {device}"
        )
        return bundle
