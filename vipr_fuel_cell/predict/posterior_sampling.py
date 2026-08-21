"""Posterior sampling and aggregation primitives for PEMFC inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.predict.posterior_result import quantile_key


class PosteriorSampler:
    """Draw physical operating-parameter samples for condition batches."""

    def __init__(
        self,
        bundle: PEMFCCINNBundle,
        *,
        num_samples: int,
        seed: int,
        common_latent_samples: bool,
    ):
        self._bundle = bundle
        self._num_samples = num_samples
        self._n_parameters = len(bundle.parameter_names)
        self._generator = torch.Generator(device=bundle.device)
        self._generator.manual_seed(seed)
        self._shared_latent = (
            torch.randn(
                num_samples,
                self._n_parameters,
                generator=self._generator,
                device=bundle.device,
            )
            if common_latent_samples
            else None
        )

    # The trained cINN is evaluated only; disable autograd and version tracking.
    @torch.inference_mode()
    def samples_for(self, conditions: np.ndarray) -> torch.Tensor:
        """Return finite physical samples as PyTorch inference tensors."""
        # VIPR DataSet arrays are read-only, so copy before creating a tensor.
        condition_tensor = torch.tensor(
            np.array(conditions, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=self._bundle.device,
        )
        batch_size = condition_tensor.shape[0]
        repeated_conditions = (
            condition_tensor[:, None, :]
            .expand(batch_size, self._num_samples, -1)
            .reshape(-1, condition_tensor.shape[-1])
        )
        latent = self._latent_for(batch_size)
        scaled_samples = self._bundle.model.inverse(latent, repeated_conditions)
        samples = self._bundle.parameter_scaler.inverse_transform_tensor(
            scaled_samples
        ).reshape(batch_size, self._num_samples, self._n_parameters)
        if not torch.all(torch.isfinite(samples)):
            raise ValueError("PEMFC cINN produced non-finite posterior samples")
        return samples

    def _latent_for(self, batch_size: int) -> torch.Tensor:
        if self._shared_latent is not None:
            return (
                self._shared_latent[None, :, :]
                .expand(batch_size, -1, -1)
                .reshape(-1, self._n_parameters)
            )
        return torch.randn(
            batch_size * self._num_samples,
            self._n_parameters,
            generator=self._generator,
            device=self._bundle.device,
        )


class PosteriorAccumulator:
    """Collect posterior summary series across condition batches."""

    def __init__(
        self,
        parameter_names: Sequence[str],
        quantiles: Sequence[float],
    ):
        self._parameter_names = list(parameter_names)
        self._quantiles = list(quantiles)
        self._quantile_keys = [quantile_key(value) for value in quantiles]
        self._statistics = {
            name: {
                "mean": [],
                "std": [],
                "min": [],
                "max": [],
                "quantiles": {key: [] for key in self._quantile_keys},
            }
            for name in self._parameter_names
        }

    # These statistics are exported as final values and never differentiated.
    # Disable autograd even if the input tensor tracks gradients.
    @torch.inference_mode()
    def add(self, samples: torch.Tensor) -> None:
        """Append one posterior sample batch to the summary series."""
        means = samples.mean(dim=1)
        stds = samples.std(dim=1, unbiased=False)
        minima = samples.amin(dim=1)
        maxima = samples.amax(dim=1)
        quantiles = torch.quantile(
            samples,
            torch.as_tensor(self._quantiles, device=samples.device),
            dim=1,
        )

        for parameter_index, name in enumerate(self._parameter_names):
            entry = self._statistics[name]
            entry["mean"].extend(means[:, parameter_index].cpu().tolist())
            entry["std"].extend(stds[:, parameter_index].cpu().tolist())
            entry["min"].extend(minima[:, parameter_index].cpu().tolist())
            entry["max"].extend(maxima[:, parameter_index].cpu().tolist())
            for quantile_index, key in enumerate(self._quantile_keys):
                entry["quantiles"][key].extend(
                    quantiles[quantile_index, :, parameter_index].cpu().tolist()
                )

    def result(self) -> dict[str, dict]:
        return self._statistics


def build_posterior_snapshot(
    samples: torch.Tensor,
    *,
    valid_time_step_index: int,
    time: float,
    parameter_names: Sequence[str],
    bins: int,
) -> dict:
    """Build empirical marginal histograms for one valid time step."""
    snapshot_samples = samples.detach().cpu().numpy()
    histograms = {}
    for parameter_index, name in enumerate(parameter_names):
        counts, bin_edges = np.histogram(
            snapshot_samples[:, parameter_index],
            bins=bins,
        )
        bin_widths = np.diff(bin_edges)
        density = counts.astype(np.float64) / (counts.sum() * bin_widths)
        histograms[name] = {
            "bin_edges": bin_edges.tolist(),
            "density": density.tolist(),
            "counts": counts.tolist(),
        }
    return {
        "valid_time_step_index": valid_time_step_index,
        "time": float(time),
        "histograms": histograms,
    }
