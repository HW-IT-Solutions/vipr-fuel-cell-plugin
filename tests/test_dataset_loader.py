from pathlib import Path

import numpy as np

from tests.helpers import make_app
from vipr_fuel_cell.load_data.dataset_loader import PEMFCDatasetLoader


def _loader(config_path: Path | None = None) -> PEMFCDatasetLoader:
    loader = object.__new__(PEMFCDatasetLoader)
    loader.app = make_app(config_path)
    return loader


def test_packaged_dataset_loads_through_explicit_resource_paths():
    dataset = _loader()._load_data(
        data_path=(
            "@vipr_fuel_cell/resources/datasets/operating_profile/sensor_data.csv"
        ),
        metadata_path=(
            "@vipr_fuel_cell/resources/datasets/operating_profile/metadata.yaml"
        ),
    )

    assert dataset.x.shape == (301, 11)
    assert dataset.y.shape == (301, 1)
    assert dataset.metadata["dataset_id"] == "operating_profile"
    assert dataset.metadata["time_label"] == "Time"
    assert "reference_values" not in dataset.metadata


def test_custom_dataset_resolves_files_next_to_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vipr: {}\n", encoding="utf-8")
    (tmp_path / "sensor.csv").write_text(
        "time_s,cell_voltage_V\n0.0,0.7\n0.1,0.6\n",
        encoding="utf-8",
    )
    (tmp_path / "metadata.yaml").write_text(
        """\
schema_version: 1
id: custom_profile
title: Custom profile
description: Test data
time: {column: time_s, label: Time, unit: s}
conditions:
  - {name: U_cell_V, column: cell_voltage_V, label: Cell voltage, unit: V}
""",
        encoding="utf-8",
    )

    loader = _loader(config_path)
    messages = []
    loader.app.log.info = messages.append
    dataset = loader._load_data(
        data_path="sensor.csv",
        metadata_path="metadata.yaml",
    )

    np.testing.assert_allclose(dataset.x[:, 0], [0.7, 0.6])
    np.testing.assert_allclose(dataset.y[:, 0], [0.0, 0.1])
    assert dataset.metadata["condition_names"] == ["U_cell_V"]
    assert messages[-1].startswith("Loaded PEMFC dataset 'custom_profile' from")
