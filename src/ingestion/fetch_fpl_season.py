"""Downloads performance data from vaastav/Fantasy-Premier-League and
filters for the configured team only (default: Liverpool).

Data source: static files on raw.githubusercontent.com, no API key,
no rate limits to handle (not a live endpoint).

Usage:
    python -m src.ingestion.fetch_fpl_season
    python -m src.ingestion.fetch_fpl_season --season 2025-26
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import PROJECT_ROOT, data_dir, load_config, season_ids  # noqa: E402

REQUEST_TIMEOUT = 30


def _download_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def fetch_season(season_id: str, cfg: dict) -> dict[str, Path]:
    """Downloads and filters FPL data for a single season.

    Returns the paths of the written files.
    """
    season_cfg = cfg["seasons"][season_id]
    fpl_dir = season_cfg["fpl_season_dir"]
    team_name = cfg["team"]["fpl_team_name"]
    base_url = cfg["fpl"]["base_url"]

    out_dir = data_dir("raw", None) / "fpl" / season_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) players_raw.csv -> squad players master data
    #    "team" column here is the numeric team id, not the name.
    players = _download_csv(f"{base_url}/{fpl_dir}/players_raw.csv")
    teams = _download_csv(f"{base_url}/{fpl_dir}/teams.csv")
    team_row = teams[teams["name"] == team_name]
    if team_row.empty:
        raise ValueError(f"Team '{team_name}' not found in teams.csv for {season_id}")
    team_id = int(team_row.iloc[0]["id"])

    team_players = players[players["team"] == team_id].copy()
    keep_cols = [
        "id", "web_name", "first_name", "second_name", "known_name",
        "element_type", "team", "now_cost", "total_points", "minutes",
        "goals_scored", "assists", "yellow_cards", "red_cards",
    ]
    keep_cols = [c for c in keep_cols if c in team_players.columns]
    team_players = team_players[keep_cols]

    players_path = out_dir / "players.csv"
    team_players.to_csv(players_path, index=False)

    # 2) merged_gw.csv -> gameweek-by-gameweek statistics.
    #    here the "team" column is already the team name in plain text, filtered directly.
    gws = _download_csv(f"{base_url}/{fpl_dir}/gws/merged_gw.csv")
    team_gws = gws[gws["team"] == team_name].copy()

    gws_path = out_dir / "gameweeks.csv"
    team_gws.to_csv(gws_path, index=False)

    print(
        f"[{season_id}] {len(team_players)} players, "
        f"{len(team_gws)} gameweek rows -> {out_dir}"
    )
    return {"players": players_path, "gameweeks": gws_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", help="Only this season (e.g. 2024-25). Default: all.")
    args = parser.parse_args()

    cfg = load_config()
    seasons = [args.season] if args.season else season_ids(cfg)

    for season_id in seasons:
        if season_id not in cfg["seasons"]:
            raise SystemExit(f"Unknown season: {season_id}")
        fetch_season(season_id, cfg)


if __name__ == "__main__":
    main()
