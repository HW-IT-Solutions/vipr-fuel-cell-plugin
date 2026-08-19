"""Direct VIPR model loader for PEMFC cINN checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from vipr.plugins.discovery.decorators import discover_model_loader
from vipr.plugins.inference.handlers.model_loader import ModelLoaderHandler
from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.load_model.model import PEMFCCINN
from vipr_fuel_cell.load_model.scaler import MinMaxScaler
from vipr_fuel_cell.paths import resolve_required_file


class PEMFCCINNModelLoaderParams(BaseModel):
    """Configuration for direct cINN checkpoint loading."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str = Field(description="Lightning-style cINN checkpoint")
    scaler_x_path: str | None = Field(
        default=None,
        description="Operating-parameter scaler; inferred from run directory when omitted",
    )
    scaler_y_path: str | None = Field(
        default=None,
        description="Condition scaler; inferred from run directory when omitted",
    )
    device: str = Field(default="cpu")
    strict: bool = Field(default=True)


def _default_scaler_path(checkpoint_path: Path, filename: str) -> Path:
    run_dir = checkpoint_path.parent.parent
    return run_dir / "scalers" / filename


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
        checkpoint_path = resolve_required_file(
            self.app, params.checkpoint_path, "PEMFC cINN checkpoint"
        )
        if params.scaler_x_path:
            scaler_x_path = resolve_required_file(
                self.app, params.scaler_x_path, "PEMFC parameter scaler"
            )
        else:
            scaler_x_path = _default_scaler_path(checkpoint_path, "scaler_x.json")
        if params.scaler_y_path:
            scaler_y_path = resolve_required_file(
                self.app, params.scaler_y_path, "PEMFC condition scaler"
            )
        else:
            scaler_y_path = _default_scaler_path(checkpoint_path, "scaler_y.json")
        for label, path in (
            ("parameter scaler", scaler_x_path),
            ("condition scaler", scaler_y_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"PEMFC {label} not found: {path}")

        device = _select_device(params.device)
        # weights_only avoids arbitrary object deserialization. The Lightning
        # checkpoint used by the acceptance tests consists of tensors and basic
        # containers and is compatible with this safe mode.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        hyperparams = checkpoint.get("hyper_parameters", checkpoint.get("hparams", {}))
        parameter_names = list(hyperparams.get("input_names", []))
        condition_names = list(hyperparams.get("output_names", []))
        if not parameter_names or not condition_names:
            raise ValueError("Checkpoint does not declare input_names and output_names")

        model = PEMFCCINN(
            parameter_names=parameter_names,
            condition_names=condition_names,
            n_blocks=int(hyperparams.get("n_blocks", 10)),
            subnet_hidden_size=int(hyperparams.get("subnet_hidden_size", 512)),
        ).to(device)
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("Checkpoint contains no model state_dict")
        model.load_state_dict(state_dict, strict=params.strict)
        model.eval()

        parameter_scaler = MinMaxScaler.from_json(scaler_x_path)
        condition_scaler = MinMaxScaler.from_json(scaler_y_path)
        if parameter_scaler.n_features != len(parameter_names):
            raise ValueError(
                "Parameter scaler dimension does not match checkpoint input_names"
            )
        if condition_scaler.n_features != len(condition_names):
            raise ValueError(
                "Condition scaler dimension does not match checkpoint output_names"
            )

        self.app.log.info(
            f"Loaded PEMFC cINN with {len(condition_names)} conditions and "
            f"{len(parameter_names)} reconstructed parameters on {device}"
        )
        return PEMFCCINNBundle(
            model=model,
            condition_scaler=condition_scaler,
            parameter_scaler=parameter_scaler,
            condition_names=condition_names,
            parameter_names=parameter_names,
            device=device,
            checkpoint_path=str(checkpoint_path),
        )
