"""Centralized loading of config/seasons.yaml.

Every script in the project imports configuration through this module
instead of reading the file directly, ensuring any changes to the YAML
structure only need to be updated in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "seasons.yaml"


def load_config(config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Loads and performs minimal validation on config/seasons.yaml."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_top_level = {"team", "seasons", "fpl", "reddit"}
    missing = required_top_level - cfg.keys()
    if missing:
        raise ValueError(f"Missing keys in seasons.yaml: {missing}")

    for season_id, season_cfg in cfg["seasons"].items():
        required = {"start_date", "end_date", "fpl_season_dir"}
        missing_season = required - season_cfg.keys()
        if missing_season:
            raise ValueError(f"Season {season_id}: missing keys {missing_season}")

    return cfg


def season_ids(cfg: dict[str, Any]) -> list[str]:
    """Returns season IDs in chronological order (e.g. ['2024-25', '2025-26'])."""
    return sorted(cfg["seasons"].keys())


def data_dir(kind: str, season_id: str | None = None) -> Path:
    """Standard path under data/raw|silver|gold, optionally for a specific season."""
    base = PROJECT_ROOT / "data" / kind
    return base / season_id if season_id else base
