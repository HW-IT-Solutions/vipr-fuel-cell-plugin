"""Posterior sampling for PEMFC operating parameters."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vipr.plugins.discovery.decorators import discover_predictor
from vipr.plugins.inference.handlers.predictor import PredictorHandler
from vipr_fuel_cell.contracts import (
    PEMFCDatasetContext,
    PEMFCPosteriorMetadata,
    PEMFCPosteriorResult,
    quantile_key,
)
from vipr_fuel_cell.load_model.bundle import as_pemfc_bundle


class PEMFCPosteriorPredictorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_samples: int = Field(default=1000, ge=2)
    seed: int = Field(default=42, ge=0)
    quantiles: list[float] = Field(default_factory=lambda: [0.025, 0.5, 0.975])
    time_batch_size: int = Field(default=8, ge=1)
    common_latent_samples: bool = Field(
        default=True,
        description="Reuse latent draws across time steps for stable trajectory comparisons",
    )
    posterior_snapshot_indices: list[int] = Field(
        default_factory=list,
        description=(
            "Zero-based indices on the valid, preprocessed time axis for which "
            "empirical marginal posterior histograms are retained"
        ),
    )
    histogram_bins: int = Field(default=30, ge=5, le=200)

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, values: list[float]) -> list[float]:
        values = sorted(set(float(value) for value in values))
        if not values or any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("quantiles must contain unique values in [0, 1]")
        return values

    @field_validator("posterior_snapshot_indices")
    @classmethod
    def validate_snapshot_indices(cls, values: list[int]) -> list[int]:
        normalized = sorted(set(int(value) for value in values))
        if any(value < 0 for value in normalized):
            raise ValueError("posterior_snapshot_indices must be non-negative")
        return normalized


@discover_predictor("pemfc_posterior", PEMFCPosteriorPredictorParams)
class PEMFCPosteriorPredictor(PredictorHandler):
    """Reconstruct posterior summaries for all valid sensor time steps."""

    class Meta:
        label = "pemfc_posterior"

    def _predict(self, dataset, model, params):
        params = PEMFCPosteriorPredictorParams.model_validate(params)
        model = as_pemfc_bundle(model)
        context = PEMFCDatasetContext.model_validate(dataset.metadata)
        if not context.conditions_scaled:
            raise ValueError(
                "PEMFC conditions are not scaled; enable PEMFCConditionPreprocessor"
            )
        if dataset.x.shape[1] != len(model.condition_names):
            raise ValueError(
                f"Model expects {len(model.condition_names)} conditions, "
                f"DataSet provides {dataset.x.shape[1]}"
            )

        time = (
            np.asarray(dataset.y[:, 0], dtype=np.float64)
            if dataset.y is not None
            else np.arange(dataset.batch_size, dtype=np.float64)
        )
        n_steps = dataset.batch_size
        n_parameters = len(model.parameter_names)
        invalid_snapshot_indices = [
            index
            for index in params.posterior_snapshot_indices
            if index >= n_steps
        ]
        if invalid_snapshot_indices:
            raise ValueError(
                "posterior_snapshot_indices are outside the valid, preprocessed "
                f"time axis of length {n_steps}: {invalid_snapshot_indices}"
            )
        quantile_keys = [quantile_key(value) for value in params.quantiles]
        collected = {
            name: {
                "mean": [],
                "std": [],
                "min": [],
                "max": [],
                "quantiles": {key: [] for key in quantile_keys},
            }
            for name in model.parameter_names
        }
        snapshots: dict[int, dict] = {}

        generator = torch.Generator(device=model.device)
        generator.manual_seed(params.seed)
        shared_latent = None
        if params.common_latent_samples:
            shared_latent = torch.randn(
                params.num_samples,
                n_parameters,
                generator=generator,
                device=model.device,
            )

        started = perf_counter()
        with torch.inference_mode():
            for start in range(0, n_steps, params.time_batch_size):
                stop = min(start + params.time_batch_size, n_steps)
                # DataSet arrays are intentionally read-only. Copy the small
                # chunk before creating a tensor to avoid undefined behaviour
                # warnings from torch.as_tensor on non-writable NumPy memory.
                conditions = torch.tensor(
                    np.array(dataset.x[start:stop], dtype=np.float32, copy=True),
                    dtype=torch.float32,
                    device=model.device,
                )
                batch_size = conditions.shape[0]
                repeated_conditions = (
                    conditions[:, None, :]
                    .expand(batch_size, params.num_samples, -1)
                    .reshape(-1, conditions.shape[-1])
                )
                if shared_latent is not None:
                    latent = (
                        shared_latent[None, :, :]
                        .expand(batch_size, -1, -1)
                        .reshape(-1, n_parameters)
                    )
                else:
                    latent = torch.randn(
                        batch_size * params.num_samples,
                        n_parameters,
                        generator=generator,
                        device=model.device,
                    )

                scaled_samples = model.model.inverse(latent, repeated_conditions)
                samples = model.parameter_scaler.inverse_transform_tensor(
                    scaled_samples
                )
                samples = samples.reshape(batch_size, params.num_samples, n_parameters)
                if not torch.all(torch.isfinite(samples)):
                    raise ValueError("PEMFC cINN produced non-finite posterior samples")

                means = samples.mean(dim=1)
                stds = samples.std(dim=1, unbiased=False)
                minima = samples.amin(dim=1)
                maxima = samples.amax(dim=1)
                quantiles = torch.quantile(
                    samples,
                    torch.as_tensor(params.quantiles, device=model.device),
                    dim=1,
                )

                requested_in_chunk = [
                    index
                    for index in params.posterior_snapshot_indices
                    if start <= index < stop
                ]
                if requested_in_chunk:
                    snapshot_samples = samples.detach().cpu().numpy()
                    for index in requested_in_chunk:
                        local_index = index - start
                        histograms = {}
                        for parameter_index, name in enumerate(model.parameter_names):
                            counts, bin_edges = np.histogram(
                                snapshot_samples[local_index, :, parameter_index],
                                bins=params.histogram_bins,
                            )
                            bin_widths = np.diff(bin_edges)
                            density = counts.astype(np.float64) / (
                                counts.sum() * bin_widths
                            )
                            histograms[name] = {
                                "bin_edges": bin_edges.tolist(),
                                "density": density.tolist(),
                                "counts": counts.tolist(),
                            }
                        snapshots[index] = {
                            "valid_time_step_index": index,
                            "time": float(time[index]),
                            "histograms": histograms,
                        }

                for parameter_index, name in enumerate(model.parameter_names):
                    entry = collected[name]
                    entry["mean"].extend(means[:, parameter_index].cpu().tolist())
                    entry["std"].extend(stds[:, parameter_index].cpu().tolist())
                    entry["min"].extend(minima[:, parameter_index].cpu().tolist())
                    entry["max"].extend(maxima[:, parameter_index].cpu().tolist())
                    for quantile_index, key in enumerate(quantile_keys):
                        entry["quantiles"][key].extend(
                            quantiles[quantile_index, :, parameter_index].cpu().tolist()
                        )

        elapsed = perf_counter() - started
        self.app.log.info(
            f"Reconstructed {n_steps} PEMFC time steps with {params.num_samples} "
            f"posterior samples each in {elapsed:.3f} s"
        )
        return PEMFCPosteriorResult(
            time=time.tolist(),
            time_label=context.time_label,
            time_unit=context.time_unit,
            parameter_names=list(model.parameter_names),
            parameters=model.parameters,
            statistics=collected,
            snapshots=[
                snapshots[index] for index in params.posterior_snapshot_indices
            ],
            metadata=PEMFCPosteriorMetadata(
                dataset_id=context.dataset_id,
                dataset_title=context.dataset_title,
                dataset_source=context.dataset_source,
                model_id=model.model_id,
                num_samples=params.num_samples,
                seed=params.seed,
                quantiles=params.quantiles,
                common_latent_samples=params.common_latent_samples,
                histogram_bins=params.histogram_bins,
                posterior_snapshot_indices=params.posterior_snapshot_indices,
                inference_seconds=elapsed,
                valid_time_steps=n_steps,
                dropped_time_step_indices=context.dropped_time_step_indices,
                out_of_range_value_count=context.out_of_range_value_count,
            ),
        ).as_vipr_payload()
