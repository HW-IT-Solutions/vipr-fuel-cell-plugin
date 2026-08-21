"""Posterior sampling for PEMFC operating parameters."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vipr.plugins.discovery.decorators import discover_predictor
from vipr.plugins.inference.handlers.predictor import PredictorHandler
from vipr_fuel_cell.load_data.dataset_context import PEMFCDatasetContext
from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle, as_pemfc_bundle
from vipr_fuel_cell.predict.posterior_result import (
    PEMFCPosteriorMetadata,
    PEMFCPosteriorResult,
)
from vipr_fuel_cell.predict.posterior_sampling import (
    PosteriorAccumulator,
    PosteriorSampler,
    build_posterior_snapshot,
)


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


def _validate_prediction_inputs(dataset, bundle, params):
    context = PEMFCDatasetContext.model_validate(dataset.metadata)
    if not context.conditions_scaled:
        raise ValueError(
            "PEMFC conditions are not scaled; enable PEMFCConditionPreprocessor"
        )
    if dataset.x.shape[1] != len(bundle.condition_names):
        raise ValueError(
            f"Model expects {len(bundle.condition_names)} conditions, "
            f"DataSet provides {dataset.x.shape[1]}"
        )

    time = (
        np.asarray(dataset.y[:, 0], dtype=np.float64)
        if dataset.y is not None
        else np.arange(dataset.batch_size, dtype=np.float64)
    )
    invalid_snapshot_indices = [
        index
        for index in params.posterior_snapshot_indices
        if index >= dataset.batch_size
    ]
    if invalid_snapshot_indices:
        raise ValueError(
            "posterior_snapshot_indices are outside the valid, preprocessed "
            f"time axis of length {dataset.batch_size}: {invalid_snapshot_indices}"
        )
    return context, time


def _build_posterior_result(
    *,
    context: PEMFCDatasetContext,
    bundle: PEMFCCINNBundle,
    params: PEMFCPosteriorPredictorParams,
    time: np.ndarray,
    statistics: dict[str, dict],
    snapshots: dict[int, dict],
    inference_seconds: float,
) -> dict:
    return PEMFCPosteriorResult(
        time=time.tolist(),
        time_label=context.time_label,
        time_unit=context.time_unit,
        parameter_names=list(bundle.parameter_names),
        parameters=bundle.parameters,
        statistics=statistics,
        snapshots=[snapshots[index] for index in params.posterior_snapshot_indices],
        metadata=PEMFCPosteriorMetadata(
            dataset_id=context.dataset_id,
            dataset_title=context.dataset_title,
            dataset_source=context.dataset_source,
            model_id=bundle.model_id,
            num_samples=params.num_samples,
            seed=params.seed,
            quantiles=params.quantiles,
            common_latent_samples=params.common_latent_samples,
            histogram_bins=params.histogram_bins,
            posterior_snapshot_indices=params.posterior_snapshot_indices,
            inference_seconds=inference_seconds,
            valid_time_steps=len(time),
            dropped_time_step_indices=context.dropped_time_step_indices,
            out_of_range_value_count=context.out_of_range_value_count,
        ),
    ).as_vipr_payload()


@discover_predictor("pemfc_posterior", PEMFCPosteriorPredictorParams)
class PEMFCPosteriorPredictor(PredictorHandler):
    """Reconstruct posterior summaries for all valid sensor time steps."""

    class Meta:
        label = "pemfc_posterior"

    def _predict(self, dataset, model, params):
        # Validate the VIPR boundary and recover the aligned valid time axis.
        params = PEMFCPosteriorPredictorParams.model_validate(params)
        bundle = as_pemfc_bundle(model)
        context, time = _validate_prediction_inputs(dataset, bundle, params)
        n_steps = dataset.batch_size

        # Keep the random sampling state consistent across time batches.
        sampler = PosteriorSampler(
            bundle,
            num_samples=params.num_samples,
            seed=params.seed,
            common_latent_samples=params.common_latent_samples,
        )
        # Reduce each sample batch to parameter-wise statistical time series.
        accumulator = PosteriorAccumulator(
            bundle.parameter_names,
            params.quantiles,
        )
        snapshots: dict[int, dict] = {}

        started = perf_counter()
        # Batch time steps to limit the memory used by posterior samples.
        for start in range(0, n_steps, params.time_batch_size):
            stop = min(start + params.time_batch_size, n_steps)
            # Run the inverse cINN to sample operating-parameter posteriors.
            samples = sampler.samples_for(dataset.x[start:stop])
            accumulator.add(samples)

            # Retain empirical distributions only for requested time steps.
            for index in params.posterior_snapshot_indices:
                if start <= index < stop:
                    snapshots[index] = build_posterior_snapshot(
                        samples[index - start],
                        valid_time_step_index=index,
                        time=time[index],
                        parameter_names=bundle.parameter_names,
                        bins=params.histogram_bins,
                    )

        elapsed = perf_counter() - started
        self.app.log.info(
            f"Reconstructed {n_steps} PEMFC time steps with {params.num_samples} "
            f"posterior samples each in {elapsed:.3f} s"
        )
        return _build_posterior_result(
            context=context,
            bundle=bundle,
            params=params,
            time=time,
            statistics=accumulator.result(),
            snapshots=snapshots,
            inference_seconds=elapsed,
        )
