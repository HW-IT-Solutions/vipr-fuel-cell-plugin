from types import SimpleNamespace

import numpy as np
import torch

from vipr.plugins.inference.dataset import DataSet
from vipr_fuel_cell.load_model.bundle import PEMFCCINNBundle
from vipr_fuel_cell.load_model.scaler import MinMaxScaler
from vipr_fuel_cell.preprocess.condition_preprocessor import PEMFCConditionPreprocessor
from tests.helpers import make_app


def test_preprocessor_reorders_drops_invalid_and_scales():
    app = make_app()
    condition_scaler = MinMaxScaler(np.array([[0.0, 10.0], [10.0, 20.0]]))
    app.inference.model = PEMFCCINNBundle(
        model=SimpleNamespace(),
        condition_scaler=condition_scaler,
        parameter_scaler=MinMaxScaler(np.array([[0.0, 1.0]])),
        condition_names=["a", "b"],
        parameter_names=["p"],
        device=torch.device("cpu"),
        checkpoint_path="model.ckpt",
    )
    dataset = DataSet(
        x=np.array([[15.0, 5.0], [np.nan, 6.0], [20.0, 10.0]]),
        y=np.array([[0.0], [1.0], [2.0]]),
        metadata={"condition_names": ["b", "a"], "conditions_scaled": False},
    )
    preprocessor = PEMFCConditionPreprocessor(app)

    result = preprocessor.preprocess_conditions(dataset)

    np.testing.assert_allclose(result.x, [[0.5, 0.5], [1.0, 1.0]])
    np.testing.assert_allclose(result.y[:, 0], [0.0, 2.0])
    assert result.metadata["dropped_time_step_indices"] == [1]
    assert result.metadata["conditions_scaled"] is True
