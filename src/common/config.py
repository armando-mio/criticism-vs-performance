"""Caricamento centralizzato di config/seasons.yaml.

Ogni script del progetto passa da qui invece di leggere il file
direttamente, cosi' se cambia la struttura dello yaml si aggiorna
in un solo punto.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "seasons.yaml"


def load_config(config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Carica e valida (minimamente) config/seasons.yaml."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config non trovata: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_top_level = {"team", "seasons", "fpl", "reddit"}
    missing = required_top_level - cfg.keys()
    if missing:
        raise ValueError(f"Chiavi mancanti in seasons.yaml: {missing}")

    for season_id, season_cfg in cfg["seasons"].items():
        required = {"start_date", "end_date", "fpl_season_dir"}
        missing_season = required - season_cfg.keys()
        if missing_season:
            raise ValueError(f"Stagione {season_id}: chiavi mancanti {missing_season}")

    return cfg


def season_ids(cfg: dict[str, Any]) -> list[str]:
    """Ritorna gli id di stagione in ordine cronologico (es. ['2024-25', '2025-26'])."""
    return sorted(cfg["seasons"].keys())


def data_dir(kind: str, season_id: str | None = None) -> Path:
    """Path standard sotto data/raw|silver|gold, opzionalmente per una stagione."""
    base = PROJECT_ROOT / "data" / kind
    return base / season_id if season_id else base
