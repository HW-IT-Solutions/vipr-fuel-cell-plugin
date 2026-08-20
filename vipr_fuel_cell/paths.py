"""Path resolution shared by fuel-cell handlers."""

from pathlib import Path

from vipr.vipr_paths import resolve_file_path


def config_base_dir(app) -> Path:
    """Return the directory containing the active VIPR YAML configuration."""
    pargs = getattr(app, "pargs", None)
    config_path = getattr(pargs, "vipr_config", None) if pargs else None
    return Path(config_path).resolve().parent if config_path else Path.cwd()


def resolve_required_file(app, value: str, label: str) -> Path:
    """Resolve a VIPR path and raise an actionable error when it is missing."""
    resolved = resolve_file_path(value, base_dir=config_base_dir(app))
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(f"Could not resolve {label}: {value!r}")
    return resolved


def resolve_required_directory(app, value: str, label: str) -> Path:
    """Resolve an external directory relative to the active VIPR config."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_base_dir(app) / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Could not resolve {label}: {value!r}")
    return path
