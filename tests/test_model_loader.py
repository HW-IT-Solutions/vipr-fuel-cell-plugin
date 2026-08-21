import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

from tests.helpers import make_app
from vipr_fuel_cell import paths as plugin_paths
from vipr_fuel_cell.load_model.cinn_loader import PEMFCCINNModelLoader
from vipr_fuel_cell.load_model.cinn_network import PEMFCCINN
from vipr_fuel_cell.load_model.manifest import (
    PEMFCModelManifest,
    load_model_manifest,
)
from vipr_fuel_cell.load_model.scaler import MinMaxScaler


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _manifest_payload(artifact_hashes: dict[str, str]) -> dict:
    return {
        "schema_version": 1,
        "id": "custom_model",
        "title": "Custom model",
        "description": "Synthetic model for tests",
        "publication": {
            "title": "Test publication",
            "doi": "https://example.test/model",
            "test_case": "Test",
        },
        "conditions": [
            {"name": name, "id": f"signal_{index}", "label": name, "unit": ""}
            for index, name in enumerate(("c1", "c2", "c3"), start=1)
        ],
        "targets": [
            {"name": name, "id": name, "label": name, "unit": ""}
            for name in ("p1", "p2")
        ],
        "artifacts": {
            "checkpoint": {
                "filename": "checkpoint.ckpt",
                "sha256": artifact_hashes["checkpoint.ckpt"],
            },
            "parameter_scaler": {
                "filename": "scaler_x.json",
                "sha256": artifact_hashes["scaler_x.json"],
            },
            "condition_scaler": {
                "filename": "scaler_y.json",
                "sha256": artifact_hashes["scaler_y.json"],
            },
        },
    }


def _write_model_bundle(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = PEMFCCINN(
        parameter_names=["p1", "p2"],
        condition_names=["c1", "c2", "c3"],
        n_blocks=1,
        subnet_hidden_size=4,
    )
    checkpoint_path = model_dir / "checkpoint.ckpt"
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
    (model_dir / "scaler_x.json").write_text(
        json.dumps({"feature_range": [0, 1], "minmax": [[0, 1], [10, 20]]}),
        encoding="utf-8",
    )
    (model_dir / "scaler_y.json").write_text(
        json.dumps({"feature_range": [0, 1], "minmax": [[0, 1], [0, 2], [-1, 1]]}),
        encoding="utf-8",
    )
    hashes = {path.name: _sha256(path) for path in model_dir.iterdir()}
    (model_dir / "model.yaml").write_text(
        yaml.safe_dump(_manifest_payload(hashes), sort_keys=False),
        encoding="utf-8",
    )
    return model_dir


@pytest.mark.integration
def test_local_model_bundle_loads_verified_artifacts_and_semantic_metadata():
    loader = object.__new__(PEMFCCINNModelLoader)
    loader.app = make_app()

    bundle = loader._load_model(
        model_dir="models/test_case_1",
        device="cpu",
    )

    assert bundle.model_id == "test_case_1"
    assert bundle.conditions["U_cell_V"].id == "cell_voltage"
    assert bundle.parameters["T_In_An_Ist"].label == "Anode inlet temperature"
    assert bundle.parameters["T_In_An_Ist"].unit == "K"


def test_model_manifest_defines_complete_contract():
    path = Path(__file__).parents[1] / "models/test_case_1/model.yaml"
    manifest = load_model_manifest(path)

    assert manifest.id == "test_case_1"
    assert len(manifest.conditions) == 11
    assert len(manifest.targets) == 9
    assert manifest.artifacts.checkpoint.filename == "checkpoint.ckpt"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "../../test_case_1", "model id"),
        ("filename", "../checkpoint.ckpt", "basename"),
        ("sha256", "not-a-hash", "64 hexadecimal"),
    ],
)
def test_manifest_rejects_unsafe_identifiers_and_artifacts(field, value, message):
    payload = _manifest_payload(
        {
            "checkpoint.ckpt": "a" * 64,
            "scaler_x.json": "b" * 64,
            "scaler_y.json": "c" * 64,
        }
    )
    if field == "id":
        payload["id"] = value
    else:
        payload["artifacts"]["checkpoint"][field] = value

    with pytest.raises(ValidationError, match=message):
        PEMFCModelManifest.model_validate(payload)


def test_missing_model_directory_reports_resolved_path(tmp_path):
    config_path = tmp_path / "configs/example.yaml"
    expected = tmp_path / "configs/models/test_case_1"

    with pytest.raises(FileNotFoundError, match=str(expected)):
        plugin_paths.resolve_model_directory(
            make_app(config_path),
            "models/test_case_1",
        )


def test_model_directory_is_resolved_relative_to_configuration(tmp_path):
    config_directory = tmp_path / "configs"
    model_dir = config_directory / "models/test_case_1"
    model_dir.mkdir(parents=True)

    resolved = plugin_paths.resolve_model_directory(
        make_app(config_directory / "example.yaml"),
        "models/test_case_1",
    )

    assert resolved == model_dir


def test_absolute_model_directory_is_used_directly(tmp_path):
    model_dir = tmp_path / "models/test_case_1"
    model_dir.mkdir(parents=True)

    resolved = plugin_paths.resolve_model_directory(
        make_app(tmp_path / "elsewhere/example.yaml"),
        str(model_dir),
    )

    assert resolved == model_dir


def test_explicit_model_directory_loads_complete_bundle(tmp_path):
    model_dir = _write_model_bundle(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vipr: {}\n", encoding="utf-8")
    loader = object.__new__(PEMFCCINNModelLoader)
    loader.app = make_app(config_path)

    bundle = loader._load_model(
        model_dir=model_dir.name,
        device="cpu",
    )

    assert bundle.parameter_names == ["p1", "p2"]
    assert bundle.condition_names == ["c1", "c2", "c3"]
    assert bundle.model_id == "custom_model"
    assert bundle.conditions["c1"].id == "signal_1"
    assert bundle.parameter_scaler.n_features == 2
    with torch.inference_mode():
        result = bundle.model.inverse(torch.zeros(5, 2), torch.zeros(5, 3))
    assert result.shape == (5, 2)
    assert np.isfinite(result.numpy()).all()


def test_scaler_rejects_zero_width_feature_range():
    with pytest.raises(ValueError, match="feature_range maximum"):
        MinMaxScaler(np.array([[0.0, 1.0]]), feature_range=(1.0, 1.0))
