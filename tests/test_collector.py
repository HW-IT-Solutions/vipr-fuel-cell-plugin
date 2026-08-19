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
        "time_name": "tout",
        "time_unit": "s",
        "condition_names": ["U_cell_V"],
        "parameter_names": ["T_In_An_Ist"],
        "parameter_labels": {"T_In_An_Ist": "Anode inlet temperature"},
        "parameter_units": {"T_In_An_Ist": "K"},
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
        "reference_values": {"T_In_An_Ist": 343.15},
        "metadata": {"quantiles": [0.025, 0.5, 0.975], "num_samples": 1000},
    }

    collector._collect(app, prediction)

    item = app.datacollector.data.items[0]
    assert item.tables[0].key_column == "parameter"
    assert item.tables[0].data[0]["parameter"] == "Anode inlet temperature"
    assert item.tables[0].data[0]["parameter_id"] == "anode_inlet_temperature"
    assert item.tables[0].data[0]["reference"] == 343.15
    assert item.diagrams[0].title == "Anode inlet temperature"
    assert item.diagrams[0].id == "pemfc_anode_inlet_temperature"
    assert item.diagrams[0].data["posterior_mean"] == [340.0, 341.0]
    assert len(item.diagrams[0].series) == 4
    assert item.diagrams[0].series[-1].label == "Reference"
    assert app.datacollector.data.batch_metadata["domain"] == "pemfc"
