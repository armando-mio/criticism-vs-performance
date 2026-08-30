"""Dashboard to explore the performance vs. criticism relationship for Liverpool
players across the two configured seasons.

Usage:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from databricks import sql
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, season_ids  # noqa: E402

load_dotenv()

st.set_page_config(page_title="Performance vs Criticism - Liverpool", layout="wide")

# light -> dark shade per season index, one family per metric
POINTS_SHADES = ["#8FD9BE", "#1D9E75"]
SENTIMENT_SHADES = ["#F0B499", "#D85A30"]

GRANULARITY_CONFIG = {
    "Match week": {"table": "player_gameweek_summary", "x_col": "round", "has_points": True},
    "Daily": {"table": "player_daily_sentiment", "x_col": "date", "has_points": False},
}

CORR_COLUMNS = ["total_points", "goals_scored", "assists", "minutes",
                "avg_sentiment", "n_comments", "negative_share"]

def _friendly_error_message(exc: Exception) -> str:
    msg = str(exc).lower()
    if any(kw in msg for kw in ["expired", "invalid access token", "unauthorized", "401", "authentication"]):
        return "🔒 Databricks token expired. Please contact the page admin."
    return "⚠️ Could not connect to the data source. Please contact the page admin."

@st.cache_data(ttl=3600)
def load_gold_data(table_name: str) -> pd.DataFrame:
    # Try Databricks if credentials are configured
    if all(os.getenv(k) for k in ["DATABRICKS_SERVER_HOSTNAME", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN"]):
        try:
            with sql.connect(
                server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
                http_path=os.environ["DATABRICKS_HTTP_PATH"],
                access_token=os.environ["DATABRICKS_TOKEN"],
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT * FROM performance_vs_toxicity.gold.{table_name}")
                    return cur.fetchall_arrow().to_pandas()
        except Exception as exc:
            pass

    # Fallback to local Gold CSV datasets
    cfg = load_config()
    seasons = season_ids(cfg)
    frames = []
    for s in seasons:
        csv_path = Path("data/gold") / s / f"{table_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "season" not in df.columns:
                df["season"] = s
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_cross_season_players(seasons: list[str]) -> set[str]:
    """Players with minutes > 0 in configured seasons."""
    gw = load_gold_data("player_gameweek_summary")
    if gw.empty or "player_name" not in gw.columns or "minutes" not in gw.columns:
        return set()
    minutes_by_season = gw.groupby(["player_name", "season"])["minutes"].sum()
    played = minutes_by_season[minutes_by_season > 0].reset_index()
    counts = played.groupby("player_name")["season"].nunique()
    cross = set(counts[counts >= len(seasons)].index)
    if not cross:
        # Fallback to any active players if cross-season set is small
        return set(gw["player_name"].dropna().unique())
    return cross


def _season_shade(shades: list[str], seasons: list[str], season: str) -> str:
    idx = seasons.index(season) if season in seasons else 0
    return shades[min(idx, len(shades) - 1)]


def render_correlation_matrix(player_df: pd.DataFrame, player: str) -> None:
    available_cols = [c for c in CORR_COLUMNS if c in player_df.columns]
    numeric = player_df[available_cols].dropna(subset=["avg_sentiment"])
    varying_cols = [c for c in available_cols if numeric[c].nunique(dropna=True) > 1]
    if len(varying_cols) < 2:
        st.info(f"Not enough variation in {player}'s stats to compute a correlation matrix.")
        return
    corr = numeric[varying_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        text=corr.round(2).values, texttemplate="%{text}",
    ))
    fig.update_layout(title=f"Correlation matrix — {player}", height=420)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    cfg = load_config()
    seasons = season_ids(cfg)
    team_name = cfg["team"]["name"]

    st.title(f"{team_name}: Performance vs Criticism on Reddit")
    st.caption(
        " / ".join(f"{s} — {cfg['seasons'][s]['label']}" for s in seasons)
    )

    col1, col2, col3 = st.columns([1.5, 2, 2])
    with col1:
        granularity = st.radio("Granularity", list(GRANULARITY_CONFIG.keys()), horizontal=True)
    gconf = GRANULARITY_CONFIG[granularity]

    try:
        data = load_gold_data(gconf["table"])
    except Exception as exc:
        st.error(_friendly_error_message(exc))
        st.stop()
    if data.empty:
        st.warning(f"No data found in table '{gconf['table']}'.")
        return

    with col2:
        view_mode = st.selectbox(
            "Metric View Mode",
            ["Standard (Raw Points & Sentiment)", "Z-Score Standardized (Direct Alignment)", "Upvote-Weighted Sentiment", "Points Per 90 Min"] if gconf["has_points"] else ["Standard Average Sentiment", "Upvote-Weighted Sentiment"],
        )
    with col3:
        filter_low_sample = st.checkbox("Filter low-sample (< 3 comments)", value=False)

    if filter_low_sample and "low_sample_flag" in data.columns:
        data = data[~data["low_sample_flag"]].copy()

    try:
        eligible = get_cross_season_players(seasons)
    except Exception as exc:
        st.error(_friendly_error_message(exc))
        st.stop()
    all_players = sorted(p for p in data["player_name"].dropna().unique() if p in eligible)
    if not all_players:
        all_players = sorted(data["player_name"].dropna().unique())
    default_player = "Mohamed Salah" if "Mohamed Salah" in all_players else ("M.Salah" if "M.Salah" in all_players else (all_players[0] if all_players else None))

    selected_players = st.multiselect(
        "Players", all_players, default=[default_player] if default_player else []
    )

    x_col = gconf["x_col"]
    filtered = data[data["player_name"].isin(selected_players)].sort_values(
        ["player_name", "season", x_col]
    )

    if filtered.empty:
        st.info("Select at least one player to view charts.")
        return

    filtered = filtered.copy()
    x_label = "season_" + x_col
    filtered[x_label] = filtered["season"] + " · " + filtered[x_col].astype(str)

    if gconf["has_points"]:
        filtered["opponent_label"] = (
            "vs " + filtered["opponent"].fillna("?")
            + " (" + filtered["was_home"].map({True: "H", False: "A"}).fillna("?") + ")"
        )

    for player in selected_players:
        player_df = filtered[filtered["player_name"] == player]
        if player_df.empty:
            continue

        fig = go.Figure()
        for season in seasons:
            season_df = player_df[player_df["season"] == season]
            if season_df.empty:
                continue

            hover_extra = season_df["opponent_label"] if gconf["has_points"] else ""

            # Determine fields based on view mode
            if "Z-Score" in view_mode and "points_zscore" in season_df.columns:
                y_pts = season_df["points_zscore"]
                y_sent = season_df["sentiment_zscore"]
                pts_title = "Standardized Output (Z-Score)"
                sent_title = "Standardized Sentiment (Z-Score)"
                dual_axis = False
            elif "Per 90" in view_mode and "points_per_90" in season_df.columns:
                y_pts = season_df["points_per_90"]
                y_sent = season_df["weighted_sentiment"] if "weighted_sentiment" in season_df.columns else season_df["avg_sentiment"]
                pts_title = "Points Per 90"
                sent_title = "Sentiment"
                dual_axis = True
            elif "Weighted" in view_mode and "weighted_sentiment" in season_df.columns:
                y_pts = season_df["total_points"] if gconf["has_points"] else None
                y_sent = season_df["weighted_sentiment"]
                pts_title = "FPL Points"
                sent_title = "Upvote-Weighted Sentiment"
                dual_axis = True
            else:
                y_pts = season_df["total_points"] if gconf["has_points"] else None
                y_sent = season_df["avg_sentiment"]
                pts_title = "FPL Points"
                sent_title = "Avg Sentiment"
                dual_axis = True

            if gconf["has_points"] and y_pts is not None:
                fig.add_trace(go.Scatter(
                    x=season_df[x_label], y=y_pts,
                    name=f"{pts_title} ({season})", mode="lines+markers",
                    line=dict(color=_season_shade(POINTS_SHADES, seasons, season)),
                    text=hover_extra,
                    hovertemplate="%{x}<br>%{text}<br>" + pts_title + ": %{y:.2f}<extra></extra>",
                    yaxis="y1",
                ))

            fig.add_trace(go.Scatter(
                x=season_df[x_label], y=y_sent,
                name=f"{sent_title} ({season})", mode="lines+markers",
                line=dict(color=_season_shade(SENTIMENT_SHADES, seasons, season)),
                text=hover_extra,
                hovertemplate="%{x}<br>%{text}<br>" + sent_title + ": %{y:.2f}<extra></extra>",
                yaxis="y2" if (gconf["has_points"] and dual_axis) else "y1",
            ))

        fig.update_layout(
            title=f"{player} — {view_mode}",
            xaxis=dict(title=f"Season · {x_col}"),
            yaxis=dict(title=pts_title if (gconf["has_points"] and dual_axis) else sent_title, side="left"),
            yaxis2=dict(
                title=sent_title,
                side="right",
                overlaying="y",
                autorange=True,
                showgrid=False,
            ) if (gconf["has_points"] and dual_axis) else None,
            legend=dict(orientation="h", y=1.15),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        if gconf["has_points"]:
            render_correlation_matrix(player_df, player)


if __name__ == "__main__":
    main()