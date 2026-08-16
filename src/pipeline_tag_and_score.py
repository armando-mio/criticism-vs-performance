"""Legge data/raw/reddit/<season>/comments.jsonl, associa ogni commento
ai giocatori citati e ne calcola il sentiment, scrive il risultato in
data/silver/<season>/tagged_comments.csv.

Un commento che cita N giocatori genera N righe (una per player_id),
cosi' l'aggregazione a valle resta un semplice group-by.

Uso:
    python -m src.pipeline_tag_and_score --season 2024-25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification.sentiment_baseline import score_comment  # noqa: E402
from src.common.config import data_dir  # noqa: E402
from src.entity_resolution.player_matcher import build_alias_index, match_players  # noqa: E402


def load_raw_comments(season_id: str) -> list[dict]:
    path = data_dir("raw", season_id).parent / "reddit" / season_id / "comments.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} non esiste. Esegui prima "
            f"'python -m src.ingestion.fetch_reddit_dump --season {season_id}' "
            "(o genera un dataset demo, vedi scripts/make_demo_reddit_data.py)."
        )
    comments = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                comments.append(json.loads(line))
    return comments


def tag_and_score(
    comments: list[dict], players_df: pd.DataFrame, custom_aliases: dict | None = None
) -> pd.DataFrame:
    alias_index = build_alias_index(players_df, custom_aliases=custom_aliases)

    rows = []
    for comment in comments:
        body = comment.get("body", "")
        player_ids = match_players(body, alias_index)
        if not player_ids:
            continue

        sentiment = score_comment(body)
        for pid in player_ids:
            rows.append(
                {
                    "comment_id": comment.get("id"),
                    "player_id": pid,
                    "player_name": alias_index.display_name.get(pid),
                    "created_utc": comment.get("created_utc"),
                    "score": comment.get("score"),
                    "sentiment_compound": sentiment["compound"],
                    "sentiment_label": sentiment["label"],
                }
            )
    return pd.DataFrame(rows)


def run(season_id: str, custom_aliases: dict | None = None) -> Path:
    players_path = data_dir("raw", season_id).parent / "fpl" / season_id / "players.csv"
    players_df = pd.read_csv(players_path)

    comments = load_raw_comments(season_id)
    tagged = tag_and_score(comments, players_df, custom_aliases=custom_aliases)

    out_dir = data_dir("silver", season_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tagged_comments.csv"
    tagged.to_csv(out_path, index=False)

    print(
        f"[{season_id}] {len(comments)} commenti letti, "
        f"{len(tagged)} righe taggate (commento x giocatore) -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True)
    args = parser.parse_args()
    run(args.season)
