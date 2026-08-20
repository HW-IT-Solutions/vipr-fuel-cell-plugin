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
    condition_ids: list[str], *, conditions_scaled: bool = False
) -> dict:
    return {
        "domain": "pemfc",
        "dataset_id": "test_profile",
        "dataset_title": "Test profile",
        "dataset_description": "Synthetic unit-test profile",
        "dataset_source": {},
        "source": "sensor.csv",
        "profile_source": "profile.yaml",
        "condition_ids": condition_ids,
        "time_label": "Time",
        "time_unit": "s",
        "original_time_steps": 2,
        "conditions_scaled": conditions_scaled,
    }


def make_app(config_path: Path | None = None):
    return SimpleNamespace(
        log=NullLog(),
        pargs=SimpleNamespace(vipr_config=str(config_path) if config_path else None),
        inference=SimpleNamespace(model=None),
    )
