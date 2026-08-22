"""Aggregates sentiment (monthly, per player) and performance (per gameweek,
converted to monthly) into a single 'gold' dataset per season.

Expected inputs:
  - data/raw/fpl/<season>/gameweeks.csv   (from fetch_fpl_season.py)
  - data/silver/<season>/tagged_comments.csv
      minimum columns: player_id, created_utc, sentiment_compound

Outputs:
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
    """Converts per-gameweek rows to per-player-month rows."""
    df = gameweeks_df.copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True).astype("datetime64[ns, UTC]")
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
    """Converts per-player tagged comments to average sentiment per player-month."""
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
    """Outer merge of performance + sentiment on (player_id, month)."""
    merged = performance_df.merge(
        sentiment_df, on=["player_id", "month"], how="outer"
    )
    merged = merged.sort_values(["player_id", "month"]).reset_index(drop=True)
    return merged


def _gameweek_calendar(gameweeks_df: pd.DataFrame) -> pd.DataFrame:
    """One row per round, with the kickoff date of Liverpool's match for that round."""
    df = gameweeks_df.copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True).astype("datetime64[ns, UTC]")
    return (
        df.groupby("round")["kickoff_time"]
        .min()
        .reset_index()
        .sort_values("round")
        .rename(columns={"kickoff_time": "gw_date"})
    )


def aggregate_performance_by_gameweek(gameweeks_df: pd.DataFrame) -> pd.DataFrame:
    """Converts per-gameweek rows to per-player-round rows."""
    df = gameweeks_df.copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True).astype("datetime64[ns, UTC]")
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
    """Assigns each comment to the most recent already-started gameweek
    (window: from kickoff of gameweek N to just before kickoff of
    N+1), then computes average sentiment per player-round."""
    comments = tagged_comments_df.copy()
    comments["created_dt"] = pd.to_datetime(
        comments["created_utc"], unit="s", utc=True
    ).astype("datetime64[ns, UTC]")

    calendar = _gameweek_calendar(gameweeks_df)
    comments = comments.sort_values("created_dt")

    tagged_with_gw = pd.merge_asof(
        comments, calendar.sort_values("gw_date"),
        left_on="created_dt", right_on="gw_date", direction="backward",
    )
    tagged_with_gw = tagged_with_gw.dropna(subset=["round"])
    tagged_with_gw["round"] = tagged_with_gw["round"].astype(int)

    agg = (
        tagged_with_gw.groupby(["player_id", "round"])
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
        sentiment_df, on=["player_id", "round"], how="outer"
    )
    return merged.sort_values(["player_id", "round"]).reset_index(drop=True)


def aggregate_sentiment_by_day(tagged_comments_df: pd.DataFrame) -> pd.DataFrame:
    """Average sentiment per player-day. No performance columns:
    FPL points do not have daily granularity."""
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
        print(f"[{season_id}] no tagged_comments.csv found, gold performance only")
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

    print(f"[{season_id}] gold monthly: {len(gold)} rows")
    print(f"[{season_id}] gold match week: {len(gold_gw)} rows")
    print(f"[{season_id}] gold daily: {len(sentiment_day)} rows")
    return out_dir / "player_month_summary.csv"


if __name__ == "__main__":
    import argparse

    from src.common.config import load_config, season_ids

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", help="Only this season. Default: all.")
    args = parser.parse_args()

    cfg = load_config()
    seasons = [args.season] if args.season else season_ids(cfg)
    for s in seasons:
        run(s)