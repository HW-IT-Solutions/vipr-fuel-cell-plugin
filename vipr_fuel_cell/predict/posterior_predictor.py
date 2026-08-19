"""Posterior sampling for PEMFC operating parameters."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vipr.plugins.discovery.decorators import discover_predictor
from vipr.plugins.inference.handlers.predictor import PredictorHandler
from vipr_fuel_cell.constants import PARAMETER_LABELS, PARAMETER_UNITS
from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle


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

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, values: list[float]) -> list[float]:
        values = sorted(set(float(value) for value in values))
        if not values or any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("quantiles must contain unique values in [0, 1]")
        return values


def _quantile_key(value: float) -> str:
    return format(value, ".10g")


@discover_predictor("pemfc_posterior", PEMFCPosteriorPredictorParams)
class PEMFCPosteriorPredictor(PredictorHandler):
    """Reconstruct posterior summaries for all valid sensor time steps."""

    class Meta:
        label = "pemfc_posterior"

    def _predict(self, dataset, model, params):
        params = PEMFCPosteriorPredictorParams.model_validate(params)
        if not isinstance(model, PEMFCCINNBundle):
            raise TypeError("pemfc_posterior requires a model loaded by pemfc_cinn")
        if not dataset.metadata.get("conditions_scaled"):
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
        quantile_keys = [_quantile_key(value) for value in params.quantiles]
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
        return {
            "prediction_type": "pemfc_cinn_posterior",
            "time": time.tolist(),
            "time_name": dataset.metadata.get("time_name", "time"),
            "time_unit": dataset.metadata.get("time_unit", "index"),
            "condition_names": list(model.condition_names),
            "parameter_names": list(model.parameter_names),
            "parameter_units": {
                name: dataset.metadata.get("parameter_units", {}).get(
                    name, PARAMETER_UNITS.get(name, "")
                )
                for name in model.parameter_names
            },
            "parameter_labels": {
                name: dataset.metadata.get("parameter_labels", {}).get(
                    name, PARAMETER_LABELS.get(name, name)
                )
                for name in model.parameter_names
            },
            "statistics": collected,
            "reference_values": dict(dataset.metadata.get("reference_values", {})),
            "metadata": {
                "dataset_id": dataset.metadata.get("dataset_id"),
                "dataset_title": dataset.metadata.get("dataset_title"),
                "dataset_source": dataset.metadata.get("dataset_source", {}),
                "model_id": model.model_id,
                "num_samples": params.num_samples,
                "seed": params.seed,
                "quantiles": params.quantiles,
                "common_latent_samples": params.common_latent_samples,
                "inference_seconds": elapsed,
                "valid_time_steps": n_steps,
                "dropped_time_step_indices": dataset.metadata.get(
                    "dropped_time_step_indices", []
                ),
                "out_of_range_value_count": dataset.metadata.get(
                    "out_of_range_value_count", 0
                ),
            },
        }
