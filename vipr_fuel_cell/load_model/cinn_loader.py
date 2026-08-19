"""Direct VIPR model loader for PEMFC cINN checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from vipr.plugins.discovery.decorators import discover_model_loader
from vipr.plugins.inference.handlers.model_loader import ModelLoaderHandler
from vipr_fuel_cell.load_model.artifacts import (
    DEFAULT_MODEL_ID,
    load_bundled_model,
    load_custom_model,
)
from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.paths import resolve_required_file


class PEMFCCINNModelLoaderParams(BaseModel):
    """Configuration for direct cINN checkpoint loading."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(
        default=None,
        description="Provisioned model identifier; defaults to test_case_1",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description="Optional custom Lightning-style cINN checkpoint",
    )
    scaler_x_path: str | None = Field(
        default=None,
        description=(
            "Operating-parameter scaler; inferred next to the checkpoint or from "
            "the legacy run/scalers layout when omitted"
        ),
    )
    scaler_y_path: str | None = Field(
        default=None,
        description=(
            "Condition scaler; inferred next to the checkpoint or from the legacy "
            "run/scalers layout when omitted"
        ),
    )
    device: str = Field(default="cpu")
    strict: bool = Field(default=True)


def _default_scaler_path(checkpoint_path: Path, filename: str) -> Path:
    sibling = checkpoint_path.with_name(filename)
    if sibling.is_file():
        return sibling

    # Compatibility with the original training-run layout:
    # <run>/checkpoints/<checkpoint> and <run>/scalers/<filename>.
    legacy = checkpoint_path.parent.parent / "scalers" / filename
    return legacy if legacy.is_file() else sibling


def _select_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for the PEMFC cINN but is not available")
    return device


@discover_model_loader("pemfc_cinn", PEMFCCINNModelLoaderParams)
class PEMFCCINNModelLoader(ModelLoaderHandler):
    """Load the cINN, condition scaler and parameter scaler without a registry."""

    class Meta:
        label = "pemfc_cinn"

    def _load_model(self, **kwargs) -> PEMFCCINNBundle:
        params = PEMFCCINNModelLoaderParams.model_validate(kwargs)
        device = _select_device(params.device)
        if params.model and params.checkpoint_path:
            raise ValueError("Configure either model or checkpoint_path, not both")
        if not params.checkpoint_path and (params.scaler_x_path or params.scaler_y_path):
            raise ValueError(
                "scaler_x_path and scaler_y_path are only valid with checkpoint_path"
            )

        if params.checkpoint_path:
            checkpoint_path = resolve_required_file(
                self.app, params.checkpoint_path, "PEMFC cINN checkpoint"
            )
            scaler_x_path = (
                resolve_required_file(
                    self.app, params.scaler_x_path, "PEMFC parameter scaler"
                )
                if params.scaler_x_path
                else _default_scaler_path(checkpoint_path, "scaler_x.json")
            )
            scaler_y_path = (
                resolve_required_file(
                    self.app, params.scaler_y_path, "PEMFC condition scaler"
                )
                if params.scaler_y_path
                else _default_scaler_path(checkpoint_path, "scaler_y.json")
            )
            bundle = load_custom_model(
                checkpoint_path=checkpoint_path,
                parameter_scaler_path=scaler_x_path,
                condition_scaler_path=scaler_y_path,
                device=device,
                strict=params.strict,
            )
        else:
            bundle = load_bundled_model(
                params.model or DEFAULT_MODEL_ID,
                device=device,
                strict=params.strict,
            )

        self.app.log.info(
            f"Loaded PEMFC cINN with {len(bundle.condition_names)} conditions and "
            f"{len(bundle.parameter_names)} reconstructed parameters on {device}"
        )
        return bundle
