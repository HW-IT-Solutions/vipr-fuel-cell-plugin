"""Small JSON-backed min/max scaler used by the PEMFC cINN."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MinMaxScaler:
    """Inference-only min/max scaling without a scikit-learn dependency."""

    minmax: np.ndarray
    feature_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        values = np.asarray(self.minmax, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(
                f"Scaler minmax must have shape (features, 2), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Scaler minmax contains non-finite values")
        if np.any(values[:, 1] <= values[:, 0]):
            raise ValueError("Every scaler maximum must be greater than its minimum")
        object.__setattr__(self, "minmax", values)
        object.__setattr__(
            self,
            "feature_range",
            (float(self.feature_range[0]), float(self.feature_range[1])),
        )

    @property
    def n_features(self) -> int:
        return int(self.minmax.shape[0])

    @classmethod
    def from_json(cls, path: Path) -> "MinMaxScaler":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("minmax") is None:
            raise ValueError(f"Scaler has no minmax values: {path}")
        return cls(
            minmax=np.asarray(payload["minmax"], dtype=np.float32),
            feature_range=tuple(payload.get("feature_range", (0.0, 1.0))),
        )

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        self._check_last_dimension(values)
        lower = self.minmax[:, 0]
        span = self.minmax[:, 1] - lower
        low, high = self.feature_range
        return ((values - lower) / span) * (high - low) + low

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        self._check_last_dimension(values)
        minmax = torch.as_tensor(self.minmax, dtype=values.dtype, device=values.device)
        low, high = self.feature_range
        standardized = (values - low) / (high - low)
        return standardized * (minmax[:, 1] - minmax[:, 0]) + minmax[:, 0]

    def _check_last_dimension(self, values) -> None:
        if values.shape[-1] != self.n_features:
            raise ValueError(
                f"Scaler expects {self.n_features} features, got {values.shape[-1]}"
            )
