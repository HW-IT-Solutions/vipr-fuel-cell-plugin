"""Runtime model bundle shared by preprocessing and prediction."""

from dataclasses import dataclass

import torch

from .model import PEMFCCINN
from .scaler import MinMaxScaler


@dataclass
class PEMFCCINNBundle:
    model: PEMFCCINN
    condition_scaler: MinMaxScaler
    parameter_scaler: MinMaxScaler
    condition_names: list[str]
    parameter_names: list[str]
    device: torch.device
    checkpoint_path: str
