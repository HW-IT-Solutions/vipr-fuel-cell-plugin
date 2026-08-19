import json

import numpy as np
import torch

from vipr_fuel_cell.load_model.cinn_loader import (
    PEMFCCINNModelLoader,
    PEMFCCINNModelLoaderParams,
    _resolve_model_files,
)
from vipr_fuel_cell.load_model.model import PEMFCCINN
from tests.helpers import make_app


def test_built_in_model_alias_resolves_packaged_artifacts():
    checkpoint, scaler_x, scaler_y = _resolve_model_files(
        make_app(),
        PEMFCCINNModelLoaderParams(model="test_case_1"),
    )

    assert checkpoint.name == "checkpoint.ckpt"
    assert checkpoint.is_file()
    assert scaler_x.name == "scaler_x.json"
    assert scaler_y.name == "scaler_y.json"


def test_direct_model_loader_reconstructs_checkpoint_and_scalers(tmp_path):
    model = PEMFCCINN(
        parameter_names=["p1", "p2"],
        condition_names=["c1", "c2", "c3"],
        n_blocks=1,
        subnet_hidden_size=4,
    )
    checkpoint_path = tmp_path / "last.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                "input_names": ["p1", "p2"],
                "output_names": ["c1", "c2", "c3"],
                "n_blocks": 1,
                "subnet_hidden_size": 4,
            },
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    scaler_x_path = tmp_path / "scaler_x.json"
    scaler_y_path = tmp_path / "scaler_y.json"
    scaler_x_path.write_text(
        json.dumps({"feature_range": [0, 1], "minmax": [[0, 1], [10, 20]]}),
        encoding="utf-8",
    )
    scaler_y_path.write_text(
        json.dumps({"feature_range": [0, 1], "minmax": [[0, 1], [0, 2], [-1, 1]]}),
        encoding="utf-8",
    )

    loader = object.__new__(PEMFCCINNModelLoader)
    loader.app = make_app()
    bundle = loader._load_model(
        checkpoint_path=str(checkpoint_path),
        scaler_x_path=str(scaler_x_path),
        scaler_y_path=str(scaler_y_path),
        device="cpu",
    )

    assert bundle.parameter_names == ["p1", "p2"]
    assert bundle.condition_names == ["c1", "c2", "c3"]
    assert bundle.parameter_scaler.n_features == 2
    with torch.inference_mode():
        result = bundle.model.inverse(torch.zeros(5, 2), torch.zeros(5, 3))
    assert result.shape == (5, 2)
    assert np.isfinite(result.numpy()).all()
