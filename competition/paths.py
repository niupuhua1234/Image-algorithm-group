"""Central project paths and configuration helpers."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "project_config.yaml"


def load_project_config():
    """Load project_config.yaml and return a plain dictionary."""
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def prepared_profile(profile):
    return ROOT / "datasets" / "prepared" / profile


def require_path(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path
