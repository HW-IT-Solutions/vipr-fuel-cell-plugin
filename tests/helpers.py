from pathlib import Path
from types import SimpleNamespace


class NullLog:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass

    def exception(self, *_args, **_kwargs):
        pass


def dataset_metadata(
    condition_names: list[str], *, conditions_scaled: bool = False
) -> dict:
    return {
        "domain": "pemfc",
        "dataset_id": "test_profile",
        "dataset_title": "Test profile",
        "dataset_description": "Synthetic unit-test profile",
        "dataset_source": {},
        "source": "sensor.csv",
        "metadata_source": "metadata.yaml",
        "condition_names": condition_names,
        "condition_labels": {name: name for name in condition_names},
        "condition_units": {name: "" for name in condition_names},
        "time_label": "Time",
        "time_unit": "s",
        "reference_values": {},
        "original_time_steps": 2,
        "conditions_scaled": conditions_scaled,
    }


def make_app(config_path: Path | None = None):
    return SimpleNamespace(
        log=NullLog(),
        pargs=SimpleNamespace(vipr_config=str(config_path) if config_path else None),
        inference=SimpleNamespace(model=None),
    )
