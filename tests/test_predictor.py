from types import SimpleNamespace

import numpy as np
import torch

from vipr.plugins.inference.dataset import DataSet
from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.load_model.scaler import MinMaxScaler
from vipr_fuel_cell.predict.posterior_predictor import PEMFCPosteriorPredictor
from tests.helpers import make_app


class FakeInverseModel:
    def inverse(self, latent, conditions):
        return latent + conditions[:, : latent.shape[1]]


def test_predictor_returns_time_aligned_posterior_statistics():
    bundle = PEMFCCINNBundle(
        model=FakeInverseModel(),
        condition_scaler=MinMaxScaler(np.array([[0.0, 1.0], [0.0, 1.0]])),
        parameter_scaler=MinMaxScaler(np.array([[0.0, 10.0], [100.0, 200.0]])),
        condition_names=["c1", "c2"],
        parameter_names=["p1", "p2"],
        device=torch.device("cpu"),
        checkpoint_path="fake.ckpt",
    )
    dataset = DataSet(
        x=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        y=np.array([[0.0], [0.1]]),
        metadata={
            "conditions_scaled": True,
            "time_name": "tout",
            "time_unit": "s",
            "reference_values": {"p1": 4.0},
        },
    )
    predictor = object.__new__(PEMFCPosteriorPredictor)
    predictor.app = make_app()

    result = predictor._predict(
        dataset,
        bundle,
        {
            "num_samples": 20,
            "seed": 7,
            "quantiles": [0.1, 0.5, 0.9],
            "time_batch_size": 1,
        },
    )

    assert result["prediction_type"] == "pemfc_cinn_posterior"
    assert result["time"] == [0.0, 0.1]
    assert len(result["statistics"]["p1"]["mean"]) == 2
    assert set(result["statistics"]["p2"]["quantiles"]) == {"0.1", "0.5", "0.9"}
    assert result["reference_values"] == {"p1": 4.0}


def test_common_latent_samples_preserve_condition_shift():
    bundle = PEMFCCINNBundle(
        model=FakeInverseModel(),
        condition_scaler=MinMaxScaler(np.array([[0.0, 1.0], [0.0, 1.0]])),
        parameter_scaler=MinMaxScaler(np.array([[0.0, 1.0], [0.0, 1.0]])),
        condition_names=["c1", "c2"],
        parameter_names=["p1", "p2"],
        device=torch.device("cpu"),
        checkpoint_path="fake.ckpt",
    )
    dataset = DataSet(
        x=np.array([[0.1, 0.2], [0.4, 0.2]], dtype=np.float32),
        y=np.array([[0.0], [1.0]]),
        metadata={"conditions_scaled": True},
    )
    predictor = object.__new__(PEMFCPosteriorPredictor)
    predictor.app = make_app()
    result = predictor._predict(
        dataset,
        bundle,
        {"num_samples": 50, "seed": 1, "common_latent_samples": True},
    )

    means = result["statistics"]["p1"]["mean"]
    assert abs((means[1] - means[0]) - 0.3) < 1e-6
