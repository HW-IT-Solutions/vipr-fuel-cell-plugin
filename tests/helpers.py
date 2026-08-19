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


def make_app(config_path: Path | None = None):
    return SimpleNamespace(
        log=NullLog(),
        pargs=SimpleNamespace(vipr_config=str(config_path) if config_path else None),
        inference=SimpleNamespace(model=None),
    )
