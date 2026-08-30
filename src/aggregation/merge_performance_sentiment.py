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
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import data_dir  # noqa: E402


def _month_key(dt: pd.Series) -> pd.Series:
    return dt.dt.tz_localize(None).dt.to_period("M").astype(str)


EXCLUDED_PERFORMANCE_NAMES = {"Diogo Teixeira da Silva"}  # Diogo Jota, out of respect - see silver

NAME_CANONICALIZATION = {
    # same person, name spelled differently across seasons in FPL data
    "Luis Díaz Marulanda": "Luis Díaz",
    "Alisson Ramses Becker": "Alisson Becker",
    "Treymaurice Nyoni": "Trey Nyoni",
}


def _clean_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["name"].isin(EXCLUDED_PERFORMANCE_NAMES)].copy()
    df["name"] = df["name"].replace(NAME_CANONICALIZATION)
    return df


def aggregate_performance(gameweeks_df: pd.DataFrame) -> pd.DataFrame:
    """From per-gameweek rows to per-player-month rows."""
    df = _clean_names(gameweeks_df)
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
    """From tagged comments per player to average sentiment per player-month."""
    df = tagged_comments_df.copy()
    df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df["month"] = _month_key(df["created_dt"])

    group_cols = ["player_id", "player_name", "month"] if "player_name" in df.columns else ["player_id", "month"]
    agg = (
        df.groupby(group_cols)
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
    merged = performance_df.merge(
        sentiment_df, on=["player_id", "month"], how="outer", suffixes=("", "_sent")
    )
    if "player_name_sent" in merged.columns:
        merged["player_name"] = merged["player_name"].fillna(merged["player_name_sent"])
        merged = merged.drop(columns=["player_name_sent"])
    merged = merged.sort_values(["player_id", "month"]).reset_index(drop=True)
    return merged


def _gameweek_calendar(gameweeks_df: pd.DataFrame) -> pd.DataFrame:
    df = gameweeks_df.copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True).astype("datetime64[ns, UTC]")
    cal = (
        df.groupby("round")["kickoff_time"]
        .min()
        .reset_index()
        .sort_values("round")
        .rename(columns={"kickoff_time": "gw_date"})
    )
    # Temporal partitioning using midpoint between consecutive fixtures (handles midweek games without overlaps)
    cal["prev_gw"] = cal["gw_date"].shift(1)
    cal["window_start"] = cal["gw_date"] - (cal["gw_date"] - cal["prev_gw"]) / 2
    min_r = cal["round"].min()
    cal.loc[cal["round"] == min_r, "window_start"] = cal.loc[cal["round"] == min_r, "gw_date"] - pd.Timedelta(days=4)
    return cal.drop(columns=["prev_gw"])


def aggregate_performance_by_gameweek(gameweeks_df, teams_df=None):
    """teams_df: FPL teams.csv DataFrame (columns id, name) to resolve
    opponent_team (numeric id) into a readable name."""
    df = _clean_names(gameweeks_df)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True).astype("datetime64[ns, UTC]")
    agg = (
        df.groupby(["element", "name", "round"])
        .agg(
            gw_date=("kickoff_time", "min"),
            total_points=("total_points", "sum"),
            goals_scored=("goals_scored", "sum"),
            assists=("assists", "sum"),
            minutes=("minutes", "sum"),
            opponent_team=("opponent_team", "first"),
            was_home=("was_home", "first"),
        )
        .reset_index()
        .rename(columns={"element": "player_id", "name": "player_name"})
    )
    if teams_df is not None:
        id_to_name = dict(zip(teams_df["id"], teams_df["name"]))
        agg["opponent"] = agg["opponent_team"].map(id_to_name)
    agg = agg.drop(columns=["opponent_team"])

    # Compute per-90 metrics (only for meaningful appearances >= 15 min, otherwise scaled)
    agg["points_per_90"] = agg.apply(
        lambda r: (r["total_points"] / (r["minutes"] / 90.0)) if r["minutes"] >= 15 else (float(r["total_points"]) if r["minutes"] > 0 else 0.0),
        axis=1,
    )
    return agg


def _calc_weighted_sentiment(sub_df: pd.DataFrame) -> float:
    if sub_df.empty:
        return 0.0
    if "score" in sub_df.columns:
        # Weights bounded >= 1 to prevent negative/zero weights from silencing comments
        weights = sub_df["score"].clip(lower=1).astype(float)
        return float((sub_df["sentiment_compound"] * weights).sum() / weights.sum())
    return float(sub_df["sentiment_compound"].mean())


def aggregate_sentiment_by_gameweek(tagged_comments_df, gameweeks_df):
    comments = tagged_comments_df.copy()
    comments["created_dt"] = pd.to_datetime(comments["created_utc"], unit="s", utc=True).astype("datetime64[ns, UTC]")
    calendar = _gameweek_calendar(gameweeks_df)
    comments = comments.sort_values("created_dt")
    cal_sorted = calendar[["round", "gw_date", "window_start"]].sort_values("window_start")
    tagged_with_gw = pd.merge_asof(
        comments, cal_sorted, left_on="created_dt", right_on="window_start", direction="backward"
    )
    tagged_with_gw = tagged_with_gw.dropna(subset=["round"])
    tagged_with_gw["round"] = tagged_with_gw["round"].astype(int)

    group_cols = ["player_id", "player_name", "round"] if "player_name" in tagged_with_gw.columns else ["player_id", "round"]
    
    records = []
    for keys, group in tagged_with_gw.groupby(group_cols):
        pid = keys[0] if isinstance(keys, tuple) else keys
        pname = keys[1] if isinstance(keys, tuple) and len(keys) > 2 else None
        rnd = keys[-1] if isinstance(keys, tuple) else None

        n_tot = len(group)
        avg_sent = float(group["sentiment_compound"].mean())
        weighted_sent = _calc_weighted_sentiment(group)
        neg_share = float((group["sentiment_compound"] <= -0.05).mean())
        pos_share = float((group["sentiment_compound"] >= 0.05).mean())

        # Pre vs Post match sentiment breakdown
        pre_group = group[group["created_dt"] < group["gw_date"]]
        post_group = group[group["created_dt"] >= group["gw_date"]]

        pre_sent = float(pre_group["sentiment_compound"].mean()) if len(pre_group) > 0 else avg_sent
        post_sent = float(post_group["sentiment_compound"].mean()) if len(post_group) > 0 else avg_sent

        rec = {
            "player_id": pid,
            "round": rnd,
            "avg_sentiment": avg_sent,
            "weighted_sentiment": weighted_sent,
            "n_comments": n_tot,
            "negative_share": neg_share,
            "positive_share": pos_share,
            "pre_sentiment": pre_sent,
            "post_sentiment": post_sent,
            "n_comments_pre": len(pre_group),
            "n_comments_post": len(post_group),
            "low_sample_flag": (n_tot < 3),
        }
        if pname:
            rec["player_name"] = pname
        records.append(rec)

    return pd.DataFrame(records)


def build_gold_dataset_by_gameweek(performance_df, sentiment_df):
    merged = performance_df.merge(
        sentiment_df, on=["player_id", "round"], how="outer", suffixes=("", "_sent")
    )
    if "player_name_sent" in merged.columns:
        merged["player_name"] = merged["player_name"].fillna(merged["player_name_sent"])
        merged = merged.drop(columns=["player_name_sent"])

    # Calculate standardized Z-scores (per season) for direct statistical comparability
    if not merged.empty:
        pts = merged["total_points"].dropna()
        if len(pts) > 1 and pts.std() > 0:
            merged["points_zscore"] = (merged["total_points"] - pts.mean()) / pts.std()
        else:
            merged["points_zscore"] = 0.0

        sent = merged["avg_sentiment"].dropna()
        if len(sent) > 1 and sent.std() > 0:
            merged["sentiment_zscore"] = (merged["avg_sentiment"] - sent.mean()) / sent.std()
        else:
            merged["sentiment_zscore"] = 0.0

        # Divergence / Scapegoat Index: sentiment_zscore - points_zscore
        merged["divergence_zscore"] = merged["sentiment_zscore"] - merged["points_zscore"]

    return merged.sort_values(["player_id", "round"]).reset_index(drop=True)




def aggregate_sentiment_by_day(tagged_comments_df):
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

    teams_path = data_dir("raw", season_id).parent / "fpl" / season_id / "teams.csv"
    teams_df = pd.read_csv(teams_path) if teams_path.exists() else None
    performance_gw = aggregate_performance_by_gameweek(gameweeks_df, teams_df)

    if tagged_path.exists():
        tagged_df = pd.read_csv(tagged_path)
        sentiment = aggregate_sentiment(tagged_df)
        sentiment_gw = aggregate_sentiment_by_gameweek(tagged_df, gameweeks_df)
        sentiment_day = aggregate_sentiment_by_day(tagged_df)
    else:
        sentiment = pd.DataFrame(columns=["player_id", "month", "avg_sentiment", "n_comments", "negative_share"])
        sentiment_gw = pd.DataFrame(columns=["player_id", "round", "avg_sentiment", "n_comments", "negative_share"])
        sentiment_day = pd.DataFrame(columns=["player_id", "player_name", "date", "avg_sentiment", "n_comments", "negative_share"])

    gold = build_gold_dataset(performance, sentiment)
    gold_gw = build_gold_dataset_by_gameweek(performance_gw, sentiment_gw)

    out_dir = data_dir("gold", season_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold.to_csv(out_dir / "player_month_summary.csv", index=False)
    gold_gw.to_csv(out_dir / "player_gameweek_summary.csv", index=False)
    sentiment_day.to_csv(out_dir / "player_daily_sentiment.csv", index=False)

    print(f"[{season_id}] monthly gold: {len(gold)} rows")
    print(f"[{season_id}] match week gold: {len(gold_gw)} rows")
    print(f"[{season_id}] daily gold: {len(sentiment_day)} rows")
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