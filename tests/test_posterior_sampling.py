import numpy as np
import pytest
import torch

from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.load_model.manifest import VariableDescriptor
from vipr_fuel_cell.load_model.scaler import MinMaxScaler
from vipr_fuel_cell.predict.posterior_sampling import (
    PosteriorAccumulator,
    PosteriorSampler,
    build_posterior_snapshot,
)


class NonFiniteInverseModel:
    parameter_names = ["p1", "p2"]
    condition_names = ["c1", "c2"]

    def inverse(self, latent, _conditions):
        return torch.full_like(latent, torch.nan)


def _bundle(model) -> PEMFCCINNBundle:
    return PEMFCCINNBundle(
        model=model,
        condition_scaler=MinMaxScaler(
            np.array([[0.0, 1.0], [0.0, 1.0]])
        ),
        parameter_scaler=MinMaxScaler(
            np.array([[0.0, 1.0], [0.0, 1.0]])
        ),
        conditions={
            name: VariableDescriptor(name=name, id=name, label=name, unit="")
            for name in ("c1", "c2")
        },
        parameters={
            name: VariableDescriptor(name=name, id=name, label=name, unit="")
            for name in ("p1", "p2")
        },
        device=torch.device("cpu"),
        model_id="test_model",
    )


def test_sampler_rejects_non_finite_model_output():
    sampler = PosteriorSampler(
        _bundle(NonFiniteInverseModel()),
        num_samples=5,
        seed=1,
        common_latent_samples=True,
    )

    with pytest.raises(ValueError, match="non-finite posterior samples"):
        sampler.samples_for(np.array([[0.1, 0.2]], dtype=np.float32))


def test_accumulator_and_snapshot_preserve_empirical_samples():
    samples = torch.tensor(
        [
            [[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0]],
            [[4.0, 50.0], [5.0, 60.0], [6.0, 70.0], [7.0, 80.0]],
        ]
    )
    accumulator = PosteriorAccumulator(["p1", "p2"], [0.0, 0.5, 1.0])

    accumulator.add(samples)
    statistics = accumulator.result()
    snapshot = build_posterior_snapshot(
        samples[0],
        valid_time_step_index=3,
        time=1.5,
        parameter_names=["p1", "p2"],
        bins=2,
    )

    assert statistics["p1"]["mean"] == [1.5, 5.5]
    assert statistics["p2"]["quantiles"] == {
        "0": [10.0, 50.0],
        "0.5": [25.0, 65.0],
        "1": [40.0, 80.0],
    }
    assert snapshot["valid_time_step_index"] == 3
    assert snapshot["time"] == 1.5
    assert sum(snapshot["histograms"]["p1"]["counts"]) == 4
    assert np.isclose(
        np.sum(
            np.diff(snapshot["histograms"]["p1"]["bin_edges"])
            * snapshot["histograms"]["p1"]["density"]
        ),
        1.0,
    )
