from types import SimpleNamespace

from vipr.plugins.api.data_collector import DataCollector
from vipr_fuel_cell.postprocess.collector import PEMFCDataCollector
from tests.helpers import NullLog


class HookStub:
    def register(self, *_args, **_kwargs):
        pass


def test_collector_builds_table_and_parameter_diagram():
    app = SimpleNamespace(
        log=NullLog(),
        hook=HookStub(),
        datacollector=DataCollector(),
    )
    collector = PEMFCDataCollector(app)
    prediction = {
        "prediction_type": "pemfc_cinn_posterior",
        "time": [0.0, 0.1],
        "time_label": "tout",
        "time_unit": "s",
        "parameter_names": ["T_In_An_Ist"],
        "parameters": {
            "T_In_An_Ist": {
                "name": "T_In_An_Ist",
                "id": "anode_inlet_temperature",
                "label": "Anode inlet temperature",
                "unit": "K",
            }
        },
        "statistics": {
            "T_In_An_Ist": {
                "mean": [340.0, 341.0],
                "std": [1.0, 1.1],
                "min": [337.0, 338.0],
                "max": [343.0, 344.0],
                "quantiles": {
                    "0.025": [338.0, 339.0],
                    "0.5": [340.0, 341.0],
                    "0.975": [342.0, 343.0],
                },
            }
        },
        "snapshots": [
            {
                "valid_time_step_index": 0,
                "time": 0.0,
                "histograms": {
                    "T_In_An_Ist": {
                        "bin_edges": [338.0, 340.0, 342.0],
                        "density": [0.25, 0.25],
                        "counts": [500, 500],
                    }
                },
            }
        ],
        "metadata": {
            "dataset_id": "test_profile",
            "dataset_title": "Test profile",
            "dataset_source": {},
            "model_id": "test_case_1",
            "quantiles": [0.025, 0.5, 0.975],
            "num_samples": 1000,
            "seed": 42,
            "common_latent_samples": True,
            "histogram_bins": 2,
            "posterior_snapshot_indices": [0],
            "inference_seconds": 0.1,
            "valid_time_steps": 2,
        },
    }

    collector._collect(app, prediction)

    item = app.datacollector.data.items[0]
    assert item.tables[0].key_column == "parameter"
    assert item.tables[0].data[0]["parameter"] == "Anode inlet temperature"
    assert item.tables[0].data[0]["parameter_id"] == "anode_inlet_temperature"
    assert "reference" not in item.tables[0].data[0]
    assert item.diagrams[0].title == "Anode inlet temperature"
    assert item.diagrams[0].id == "pemfc_anode_inlet_temperature"
    assert item.diagrams[0].data["posterior_mean"] == [340.0, 341.0]
    assert [series.label for series in item.diagrams[0].series] == ["Posterior mean"]
    assert len(item.images) == 1
    assert item.images[0].format == "svg"
    assert item.images[0].data
    histogram_table = item.find_table_by_id("pemfc_posterior_snapshot_histograms")
    assert histogram_table is not None
    assert len(histogram_table.data) == 2
    assert histogram_table.data[0]["posterior_mean"] == 340.0
    image_key = "0:pemfc_posterior_distributions_step_0"
    assert image_key in app.datacollector._transient_plot_scripts
    assert image_key in app.datacollector._transient_plot_data
    assert app.datacollector._transient_plot_scripts[image_key]["data_format"] == "npz"
    assert app.datacollector.data.batch_metadata["domain"] == "pemfc"


def test_collector_keeps_quantiles_out_of_standard_diagram():
    app = SimpleNamespace(
        log=NullLog(),
        hook=HookStub(),
        datacollector=DataCollector(),
    )
    collector = PEMFCDataCollector(app)
    prediction = {
        "prediction_type": "pemfc_cinn_posterior",
        "time": [0.0],
        "time_label": "Time",
        "time_unit": "s",
        "parameter_names": ["p"],
        "parameters": {
            "p": {"name": "p", "id": "p", "label": "P", "unit": "K"}
        },
        "statistics": {
            "p": {
                "mean": [1.0],
                "std": [0.1],
                "min": [0.8],
                "max": [1.2],
                "quantiles": {"0.5": [1.0]},
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

    collector._collect(app, prediction)

    series = app.datacollector.data.items[0].diagrams[0].series
    assert [entry.label for entry in series] == ["Posterior mean"]
