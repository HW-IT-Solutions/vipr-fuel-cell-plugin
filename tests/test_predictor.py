from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from vipr.plugins.inference.dataset import DataSet
from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.load_model.scaler import MinMaxScaler
from vipr_fuel_cell.predict.posterior_predictor import PEMFCPosteriorPredictor
from tests.helpers import dataset_metadata, make_app
from vipr_fuel_cell.contracts import PEMFCPosteriorResult, ParameterDescriptor


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
        parameters={
            name: ParameterDescriptor(name=name, id=name, label=name, unit="")
            for name in ("p1", "p2")
        },
        device=torch.device("cpu"),
        checkpoint_path="fake.ckpt",
    )
    dataset = DataSet(
        x=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        y=np.array([[0.0], [0.1]]),
        metadata={
            **dataset_metadata(["c1", "c2"], conditions_scaled=True),
            "time_label": "tout",
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
            "posterior_snapshot_indices": [0],
            "histogram_bins": 5,
        },
    )

    assert result["prediction_type"] == "pemfc_cinn_posterior"
    assert result["time"] == [0.0, 0.1]
    assert len(result["statistics"]["p1"]["mean"]) == 2
    assert set(result["statistics"]["p2"]["quantiles"]) == {"0.1", "0.5", "0.9"}
    assert result["metadata"]["posterior_snapshot_indices"] == [0]
    assert len(result["snapshots"]) == 1
    histogram = result["snapshots"][0]["histograms"]["p1"]
    assert len(histogram["bin_edges"]) == 6
    assert sum(histogram["counts"]) == 20
    assert np.isclose(
        np.sum(np.diff(histogram["bin_edges"]) * histogram["density"]),
        1.0,
    )
    assert "reference_values" not in result

    with pytest.raises(ValueError, match="outside the valid, preprocessed time axis"):
        predictor._predict(
            dataset,
            bundle,
            {"num_samples": 20, "posterior_snapshot_indices": [2]},
        )


def test_common_latent_samples_preserve_condition_shift():
    bundle = PEMFCCINNBundle(
        model=FakeInverseModel(),
        condition_scaler=MinMaxScaler(np.array([[0.0, 1.0], [0.0, 1.0]])),
        parameter_scaler=MinMaxScaler(np.array([[0.0, 1.0], [0.0, 1.0]])),
        condition_names=["c1", "c2"],
        parameter_names=["p1", "p2"],
        parameters={
            name: ParameterDescriptor(name=name, id=name, label=name, unit="")
            for name in ("p1", "p2")
        },
        device=torch.device("cpu"),
        checkpoint_path="fake.ckpt",
    )
    dataset = DataSet(
        x=np.array([[0.1, 0.2], [0.4, 0.2]], dtype=np.float32),
        y=np.array([[0.0], [1.0]]),
        metadata=dataset_metadata(["c1", "c2"], conditions_scaled=True),
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


def test_posterior_contract_rejects_quantiles_not_declared_in_metadata():
    payload = {
        "time": [0.0],
        "time_label": "Time",
        "time_unit": "s",
        "parameter_names": ["p"],
        "parameters": {
            "p": {"name": "p", "id": "p", "label": "P", "unit": ""}
        },
        "statistics": {
            "p": {
                "mean": [1.0],
                "std": [0.1],
                "min": [0.8],
                "max": [1.2],
                "quantiles": {"0.1": [0.9]},
            }
        },
        "metadata": {
            "dataset_id": "test",
            "dataset_title": "Test",
            "num_samples": 10,
            "seed": 1,
            "quantiles": [0.5],
            "common_latent_samples": True,
            "inference_seconds": 0.01,
            "valid_time_steps": 1,
        },
    }

    with pytest.raises(ValidationError, match="do not match metadata"):
        PEMFCPosteriorResult.model_validate(payload)
