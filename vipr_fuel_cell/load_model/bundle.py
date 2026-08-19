"""Runtime model bundle shared by preprocessing and prediction."""

from dataclasses import dataclass

import torch

from vipr_fuel_cell.contracts import ParameterDescriptor

from .model import PEMFCCINN
from .scaler import MinMaxScaler


@dataclass
class PEMFCCINNBundle:
    model: PEMFCCINN
    condition_scaler: MinMaxScaler
    parameter_scaler: MinMaxScaler
    condition_names: list[str]
    parameter_names: list[str]
    parameters: dict[str, ParameterDescriptor]
    device: torch.device
    checkpoint_path: str
    model_id: str | None = None


def as_pemfc_bundle(candidate) -> PEMFCCINNBundle:
    """Validate the workflow model while preserving explicit dependencies."""
    if not isinstance(candidate, PEMFCCINNBundle):
        raise TypeError("PEMFC inference requires a model loaded by pemfc_cinn")
    return candidate
