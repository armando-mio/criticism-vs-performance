"""Aggrega sentiment (mensile, per giocatore) e performance (per gameweek,
convertita a mensile) in un unico dataset 'gold' per stagione.

Input attesi:
  - data/raw/fpl/<season>/gameweeks.csv   (da fetch_fpl_season.py)
  - data/silver/<season>/tagged_comments.csv
      colonne minime: player_id, created_utc, sentiment_compound

Output:
  - data/gold/<season>/player_month_summary.csv
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import data_dir  # noqa: E402


def _month_key(dt: pd.Series) -> pd.Series:
    # tolgo il timezone prima di derivare il periodo: ci interessa solo
    # l'etichetta "anno-mese", la tz non serve piu' a valle e altrimenti
    # pandas emette un warning ad ogni chiamata
    return dt.dt.tz_localize(None).dt.to_period("M").astype(str)


def aggregate_performance(gameweeks_df: pd.DataFrame) -> pd.DataFrame:
    """Da righe per-gameweek a righe per-giocatore-mese."""
    df = gameweeks_df.copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    df["month"] = _month_key(df["kickoff_time"])

    agg = (
        df.groupby(["element", "name", "month"])
        .agg(
            total_points=("total_points", "sum"),
            goals_scored=("goals_scored", "sum"),
            assists=("assists", "sum"),
            minutes=("minutes", "sum"),
            appearances=("total_points", "count"),
        )
        .reset_index()
        .rename(columns={"element": "player_id", "name": "player_name"})
    )
    return agg


def aggregate_sentiment(tagged_comments_df: pd.DataFrame) -> pd.DataFrame:
    """Da commenti taggati per-giocatore a sentiment medio per-giocatore-mese."""
    df = tagged_comments_df.copy()
    df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df["month"] = _month_key(df["created_dt"])

    agg = (
        df.groupby(["player_id", "month"])
        .agg(
            avg_sentiment=("sentiment_compound", "mean"),
            n_comments=("sentiment_compound", "count"),
            negative_share=("sentiment_compound", lambda s: float((s <= -0.05).mean())),
        )
        .reset_index()
    )
    return agg


def build_gold_dataset(
    performance_df: pd.DataFrame, sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge esterno performance + sentiment su (player_id, month).

    Outer join deliberato: un mese senza commenti resta nel dataset con
    sentiment nullo (non lo si butta via), un mese senza partite idem.
    """
    merged = performance_df.merge(
        sentiment_df, on=["player_id", "month"], how="outer"
    )
    merged = merged.sort_values(["player_id", "month"]).reset_index(drop=True)
    return merged


def run(season_id: str) -> Path:
    gw_path = data_dir("raw", season_id).parent / "fpl" / season_id / "gameweeks.csv"
    tagged_path = data_dir("silver", season_id) / "tagged_comments.csv"

    gameweeks_df = pd.read_csv(gw_path)
    performance = aggregate_performance(gameweeks_df)

    if tagged_path.exists():
        tagged_df = pd.read_csv(tagged_path)
        sentiment = aggregate_sentiment(tagged_df)
    else:
        print(f"[{season_id}] nessun tagged_comments.csv trovato, gold solo performance")
        sentiment = pd.DataFrame(columns=["player_id", "month", "avg_sentiment", "n_comments", "negative_share"])

    gold = build_gold_dataset(performance, sentiment)

    out_dir = data_dir("gold", season_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "player_month_summary.csv"
    gold.to_csv(out_path, index=False)
    print(f"[{season_id}] gold dataset: {len(gold)} righe -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    from src.common.config import load_config, season_ids

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", help="Solo questa stagione. Default: tutte.")
    args = parser.parse_args()

    cfg = load_config()
    seasons = [args.season] if args.season else season_ids(cfg)
    for s in seasons:
        run(s)
