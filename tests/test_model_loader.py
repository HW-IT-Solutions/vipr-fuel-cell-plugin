import json

import numpy as np
import pytest
import torch

from vipr_fuel_cell.load_model.cinn_loader import (
    PEMFCCINNModelLoader,
    _default_scaler_path,
)
from vipr_fuel_cell.load_model import artifacts
from vipr_fuel_cell.load_model.model import PEMFCCINN
from vipr_fuel_cell.load_model.scaler import MinMaxScaler
from tests.helpers import make_app


@pytest.mark.integration
def test_built_in_model_loads_verified_artifacts_and_presentation_metadata():
    loader = object.__new__(PEMFCCINNModelLoader)
    loader.app = make_app()

    bundle = loader._load_model(model="test_case_1", device="cpu")

    assert bundle.model_id == "test_case_1"
    assert bundle.checkpoint_path.endswith("models/test_case_1/checkpoint.ckpt")
    assert bundle.parameters["T_In_An_Ist"].label == "Anode inlet temperature"
    assert bundle.parameters["T_In_An_Ist"].unit == "K"


def test_model_identifier_is_an_exact_packaged_manifest_name():
    loader = object.__new__(PEMFCCINNModelLoader)
    loader.app = make_app()

    with pytest.raises(ValueError, match="Unknown PEMFC model"):
        loader._load_model(model="../../test_case_1", device="cpu")


def test_packaged_model_metadata_and_checksums_are_available():
    metadata = artifacts._load_model_metadata("test_case_1")
    checksums = artifacts._read_checksums("test_case_1")

    assert metadata.id == "test_case_1"
    assert {target.name for target in metadata.targets} == {
        "J_Cell_A_cm2",
        "RH_In_An_Ist",
        "RH_In_Cath_Ist",
        "T_In_An_Ist",
        "T_In_Cath_Ist",
        "p_In_An_Ist",
        "p_In_Cath_Ist",
        "Stoech_In_H2_An_Ist",
        "Stoech_In_O2_Cath_Ist",
    }
    assert set(checksums) == {"checkpoint.ckpt", "scaler_x.json", "scaler_y.json"}


def test_installed_package_requires_an_explicit_artifact_root(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(artifacts.FUEL_CELL_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(artifacts, "_PROJECT_ROOT", tmp_path / "site-packages")

    with pytest.raises(FileNotFoundError, match="VIPR_FUEL_CELL_ROOT_DIR"):
        artifacts._artifact_directory("test_case_1")


def test_default_scaler_path_supports_legacy_training_run_layout(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    legacy_scaler = tmp_path / "run" / "scalers" / "scaler_x.json"
    legacy_scaler.parent.mkdir(parents=True)
    legacy_scaler.write_text("{}", encoding="utf-8")

    assert _default_scaler_path(checkpoint, "scaler_x.json") == legacy_scaler


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
    assert bundle.parameters["p1"].label == "p1"
    assert bundle.parameter_scaler.n_features == 2
    with torch.inference_mode():
        result = bundle.model.inverse(torch.zeros(5, 2), torch.zeros(5, 3))
    assert result.shape == (5, 2)
    assert np.isfinite(result.numpy()).all()


def test_scaler_rejects_zero_width_feature_range():
    with pytest.raises(ValueError, match="feature_range maximum"):
        MinMaxScaler(np.array([[0.0, 1.0]]), feature_range=(1.0, 1.0))
