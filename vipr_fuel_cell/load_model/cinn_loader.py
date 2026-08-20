"""Direct VIPR model loader for manifest-defined PEMFC cINN bundles."""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field

from vipr.plugins.discovery.decorators import discover_model_loader
from vipr.plugins.inference.handlers.model_loader import ModelLoaderHandler
from vipr_fuel_cell.load_model.artifacts import (
    load_model_bundle,
    load_model_manifest,
    resolve_artifact_directory,
)
from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.paths import resolve_required_directory, resolve_required_file


class PEMFCCINNModelLoaderParams(BaseModel):
    """Paths and runtime options for a complete cINN model bundle."""

    model_config = ConfigDict(extra="forbid")

    manifest_path: str = Field(
        description="Model manifest path, resolved relative to the VIPR config",
    )
    artifact_dir: str | None = Field(
        default=None,
        description=(
            "Optional artifact directory containing the checkpoint and scalers; "
            "otherwise the deployment root or source checkout is used"
        ),
    )
    device: str = Field(default="cpu")
    strict: bool = Field(default=True)


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
        manifest_path = resolve_required_file(
            self.app, params.manifest_path, "PEMFC model manifest"
        )
        manifest = load_model_manifest(manifest_path)
        explicit_directory = (
            resolve_required_directory(
                self.app, params.artifact_dir, "PEMFC model artifact directory"
            )
            if params.artifact_dir
            else None
        )
        artifact_directory = resolve_artifact_directory(
            manifest.id, explicit_directory
        )
        bundle = load_model_bundle(
            manifest=manifest,
            artifact_directory=artifact_directory,
            device=device,
            strict=params.strict,
        )

        self.app.log.info(
            f"Loaded PEMFC cINN {manifest.id!r} with "
            f"{len(bundle.condition_names)} conditions and "
            f"{len(bundle.parameter_names)} reconstructed parameters on {device}"
        )
        return bundle
