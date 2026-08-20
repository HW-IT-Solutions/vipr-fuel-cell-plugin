"""Runtime model bundle shared by preprocessing and prediction."""

from dataclasses import dataclass

import torch

from .cinn_network import PEMFCCINN
from .manifest import VariableDescriptor
from .scaler import MinMaxScaler


@dataclass
class PEMFCCINNBundle:
    model: PEMFCCINN
    condition_scaler: MinMaxScaler
    parameter_scaler: MinMaxScaler
    conditions: dict[str, VariableDescriptor]
    parameters: dict[str, VariableDescriptor]
    device: torch.device
    model_id: str

    @property
    def condition_names(self) -> list[str]:
        return self.model.condition_names

    @property
    def parameter_names(self) -> list[str]:
        return self.model.parameter_names


def as_pemfc_bundle(candidate) -> PEMFCCINNBundle:
    """Validate the workflow model while preserving explicit dependencies."""
    if not isinstance(candidate, PEMFCCINNBundle):
        raise TypeError("PEMFC inference requires a model loaded by pemfc_cinn")
    return candidate
