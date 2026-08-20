"""Aggrega sentiment (mensile, per giocatore) e performance (per gameweek,
convertita a mensile) in un unico dataset 'gold' per stagione.

Input attesi:
  - data/raw/fpl/<season>/gameweeks.csv   (da fetch_fpl_season.py)
  - data/silver/<season>/tagged_comments.csv
      colonne minime: player_id, created_utc, sentiment_compound

Output:
  - data/gold/<season>/player_month_summary.csv
  - data/gold/<season>/player_gameweek_summary.csv
  - data/gold/<season>/player_daily_sentiment.csv
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import data_dir  # noqa: E402


def _month_key(dt: pd.Series) -> pd.Series:
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
    """Merge esterno performance + sentiment su (player_id, month)."""
    merged = performance_df.merge(
        sentiment_df, on=["player_id", "month"], how="outer"
    )
    merged = merged.sort_values(["player_id", "month"]).reset_index(drop=True)
    return merged


def _gameweek_calendar(gameweeks_df: pd.DataFrame) -> pd.DataFrame:
    """Una riga per round, con la data di kickoff della gara Liverpool di quel round."""
    df = gameweeks_df.copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    return (
        df.groupby("round")["kickoff_time"]
        .min()
        .reset_index()
        .sort_values("round")
        .rename(columns={"kickoff_time": "gw_date"})
    )


def aggregate_performance_by_gameweek(gameweeks_df: pd.DataFrame) -> pd.DataFrame:
    """Da righe per-gameweek a righe per-giocatore-round."""
    df = gameweeks_df.copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    agg = (
        df.groupby(["element", "name", "round"])
        .agg(
            gw_date=("kickoff_time", "min"),
            total_points=("total_points", "sum"),
            goals_scored=("goals_scored", "sum"),
            assists=("assists", "sum"),
            minutes=("minutes", "sum"),
        )
        .reset_index()
        .rename(columns={"element": "player_id", "name": "player_name"})
    )
    return agg


def aggregate_sentiment_by_gameweek(
    tagged_comments_df: pd.DataFrame, gameweeks_df: pd.DataFrame
) -> pd.DataFrame:
    """Assegna ogni commento alla gameweek piu' recente gia' iniziata
    (finestra: dal kickoff della gameweek N a un istante prima del kickoff
    N+1), poi calcola sentiment medio per-giocatore-round."""
    comments = tagged_comments_df.copy()
    comments["created_dt"] = pd.to_datetime(
        comments["created_utc"], unit="s", utc=True
    ).astype("datetime64[us, UTC]")

    calendar = _gameweek_calendar(gameweeks_df)
    comments = comments.sort_values("created_dt")

    tagged_with_gw = pd.merge_asof(
        comments, calendar.sort_values("gw_date"),
        left_on="created_dt", right_on="gw_date", direction="backward",
    )
    tagged_with_gw = tagged_with_gw.dropna(subset=["round"])
    tagged_with_gw["round"] = tagged_with_gw["round"].astype(int)

    agg = (
        tagged_with_gw.groupby(["player_id", "player_name", "round"])
        .agg(
            avg_sentiment=("sentiment_compound", "mean"),
            n_comments=("sentiment_compound", "count"),
            negative_share=("sentiment_compound", lambda s: float((s <= -0.05).mean())),
        )
        .reset_index()
    )
    return agg


def build_gold_dataset_by_gameweek(
    performance_df: pd.DataFrame, sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    merged = performance_df.merge(
        sentiment_df, on=["player_id", "player_name", "round"], how="outer"
    )
    return merged.sort_values(["player_id", "round"]).reset_index(drop=True)


def aggregate_sentiment_by_day(tagged_comments_df: pd.DataFrame) -> pd.DataFrame:
    """Sentiment medio per-giocatore-giorno. Nessuna colonna di performance:
    i punti FPL non hanno granularita' giornaliera."""
    df = tagged_comments_df.copy()
    df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df["date"] = df["created_dt"].dt.tz_localize(None).dt.date.astype(str)

    agg = (
        df.groupby(["player_id", "player_name", "date"])
        .agg(
            avg_sentiment=("sentiment_compound", "mean"),
            n_comments=("sentiment_compound", "count"),
            negative_share=("sentiment_compound", lambda s: float((s <= -0.05).mean())),
        )
        .reset_index()
    )
    return agg


def run(season_id: str) -> Path:
    gw_path = data_dir("raw", season_id).parent / "fpl" / season_id / "gameweeks.csv"
    tagged_path = data_dir("silver", season_id) / "tagged_comments.csv"

    gameweeks_df = pd.read_csv(gw_path)
    performance = aggregate_performance(gameweeks_df)
    performance_gw = aggregate_performance_by_gameweek(gameweeks_df)

    if tagged_path.exists():
        tagged_df = pd.read_csv(tagged_path)
        sentiment = aggregate_sentiment(tagged_df)
        sentiment_gw = aggregate_sentiment_by_gameweek(tagged_df, gameweeks_df)
        sentiment_day = aggregate_sentiment_by_day(tagged_df)
    else:
        print(f"[{season_id}] nessun tagged_comments.csv trovato, gold solo performance")
        sentiment = pd.DataFrame(columns=["player_id", "month", "avg_sentiment", "n_comments", "negative_share"])
        sentiment_gw = pd.DataFrame(columns=["player_id", "player_name", "round", "avg_sentiment", "n_comments", "negative_share"])
        sentiment_day = pd.DataFrame(columns=["player_id", "player_name", "date", "avg_sentiment", "n_comments", "negative_share"])

    gold = build_gold_dataset(performance, sentiment)
    gold_gw = build_gold_dataset_by_gameweek(performance_gw, sentiment_gw)

    out_dir = data_dir("gold", season_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold.to_csv(out_dir / "player_month_summary.csv", index=False)
    gold_gw.to_csv(out_dir / "player_gameweek_summary.csv", index=False)
    sentiment_day.to_csv(out_dir / "player_daily_sentiment.csv", index=False)

    print(f"[{season_id}] gold mensile: {len(gold)} righe")
    print(f"[{season_id}] gold match week: {len(gold_gw)} righe")
    print(f"[{season_id}] gold giornaliero: {len(sentiment_day)} righe")
    return out_dir / "player_month_summary.csv"


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