from types import SimpleNamespace

import numpy as np
import torch

from vipr.plugins.inference.dataset import DataSet
from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.load_model.scaler import MinMaxScaler
from vipr_fuel_cell.preprocess.condition_preprocessor import PEMFCConditionPreprocessor
from tests.helpers import dataset_metadata, make_app
from vipr_fuel_cell.contracts import ParameterDescriptor


def test_preprocessor_reorders_drops_invalid_and_scales():
    app = make_app()
    condition_scaler = MinMaxScaler(np.array([[0.0, 10.0], [10.0, 20.0]]))
    app.inference.model = PEMFCCINNBundle(
        model=SimpleNamespace(),
        condition_scaler=condition_scaler,
        parameter_scaler=MinMaxScaler(np.array([[0.0, 1.0]])),
        condition_names=["a", "b"],
        parameter_names=["p"],
        conditions={
            "a": ParameterDescriptor(name="a", id="signal_a", label="A", unit=""),
            "b": ParameterDescriptor(name="b", id="signal_b", label="B", unit=""),
        },
        parameters={
            "p": ParameterDescriptor(name="p", id="p", label="P", unit="")
        },
        device=torch.device("cpu"),
        checkpoint_path="model.ckpt",
    )
    dataset = DataSet(
        x=np.array(
            [
                [15.0, 5.0, 100.0],
                [np.nan, 6.0, 101.0],
                [20.0, 10.0, 102.0],
            ]
        ),
        y=np.array([[0.0], [1.0], [2.0]]),
        metadata={
            **dataset_metadata(["signal_b", "signal_a", "unused"]),
            "original_time_steps": 3,
            "upstream_extension": {"preserve": True},
        },
    )
    preprocessor = PEMFCConditionPreprocessor(app)

    result = preprocessor.preprocess_conditions(dataset)

    np.testing.assert_allclose(result.x, [[0.5, 0.5], [1.0, 1.0]])
    np.testing.assert_allclose(result.y[:, 0], [0.0, 2.0])
    assert result.metadata["dropped_time_step_indices"] == [1]
    assert result.metadata["conditions_scaled"] is True
    assert result.metadata["signal_ids"] == ["signal_a", "signal_b"]
    assert result.metadata["upstream_extension"] == {"preserve": True}
