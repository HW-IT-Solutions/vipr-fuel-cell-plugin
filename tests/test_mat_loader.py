from pathlib import Path

import h5py
import numpy as np

from vipr_fuel_cell.load_data.mat_loader import PEMFCMatDataLoader
from tests.helpers import make_app


def _write_mat(path: Path):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("out").create_group("Data")
        group.create_dataset("U_cell_V", data=np.array([[0.7, 0.6, 0.5]]))
        group.create_dataset("E_Con", data=np.array([[-0.1, -0.2, -0.3]]))
        group.create_dataset("tout", data=np.array([[0.0, 0.1, 0.2]]))


def test_mat_loader_returns_time_aligned_condition_matrix(tmp_path):
    mat_path = tmp_path / "acceptance.mat"
    config_path = tmp_path / "config.yaml"
    reference_path = tmp_path / "INI.txt"
    _write_mat(mat_path)
    config_path.write_text("vipr: {}\n", encoding="utf-8")
    reference_path.write_text(
        "Temperature.InletAnodeINI = [0 343.15];\n"
        "Temperature.InletCathodeINI = Temperature.InletAnodeINI;\n"
        "Pressure.InletCathodeINI = [0 1.8];\n"
        "Pressure.InletAnodeINI = Pressure.InletCathodeINI + 0.2;\n",
        encoding="utf-8",
    )

    loader = object.__new__(PEMFCMatDataLoader)
    loader.app = make_app(config_path)
    dataset = loader._load_data(
        data_path="acceptance.mat",
        reference_path="INI.txt",
        condition_names=["U_cell_V", "E_Con"],
    )

    assert dataset.x.shape == (3, 2)
    assert dataset.y.shape == (3, 1)
    np.testing.assert_allclose(dataset.y[:, 0], [0.0, 0.1, 0.2])
    assert dataset.metadata["condition_names"] == ["U_cell_V", "E_Con"]
    assert dataset.metadata["reference_values"]["T_In_Cath_Ist"] == 343.15
    assert dataset.metadata["reference_values"]["p_In_An_Ist"] == 2.0


def test_mat_loader_reports_missing_conditions(tmp_path):
    mat_path = tmp_path / "acceptance.mat"
    _write_mat(mat_path)
    loader = object.__new__(PEMFCMatDataLoader)
    loader.app = make_app()

    try:
        loader._load_data(data_path=str(mat_path), condition_names=["not_present"])
    except ValueError as exc:
        assert "missing configured conditions" in str(exc)
    else:
        raise AssertionError("missing condition must fail")
