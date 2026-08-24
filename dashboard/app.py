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


@st.cache_data(ttl=3600)
def load_gold_data(table_name: str) -> pd.DataFrame:
    with sql.connect(
        server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM performance_vs_toxicity.gold.{table_name}")
            return cur.fetchall_arrow().to_pandas()


@st.cache_data(ttl=3600)
def get_cross_season_players(seasons: list[str]) -> set[str]:
    """Players with minutes > 0 in every configured season (e.g. excludes
    Isak, who only joined mid 2025-26)."""
    gw = load_gold_data("player_gameweek_summary")
    minutes_by_season = gw.groupby(["player_name", "season"])["minutes"].sum()
    played = minutes_by_season[minutes_by_season > 0].reset_index()
    counts = played.groupby("player_name")["season"].nunique()
    return set(counts[counts >= len(seasons)].index)


def _season_shade(shades: list[str], seasons: list[str], season: str) -> str:
    idx = seasons.index(season) if season in seasons else 0
    return shades[min(idx, len(shades) - 1)]


def render_correlation_matrix(player_df: pd.DataFrame, player: str) -> None:
    corr = player_df[CORR_COLUMNS].corr()
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

    granularity = st.radio("Granularity", list(GRANULARITY_CONFIG.keys()), horizontal=True)
    gconf = GRANULARITY_CONFIG[granularity]

    data = load_gold_data(gconf["table"])
    if data.empty:
        st.warning(f"No data found in table '{gconf['table']}'.")
        return

    eligible = get_cross_season_players(seasons)
    all_players = sorted(p for p in data["player_name"].dropna().unique() if p in eligible)
    default_player = all_players[0] if all_players else None

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

            hover_extra = season_df["opponent_label"] if gconf["has_points"] else None

            if gconf["has_points"]:
                fig.add_trace(go.Scatter(
                    x=season_df[x_label], y=season_df["total_points"],
                    name=f"FPL Points ({season})", mode="lines+markers",
                    line=dict(color=_season_shade(POINTS_SHADES, seasons, season)),
                    text=hover_extra,
                    hovertemplate="%{x}<br>%{text}<br>Points: %{y}<extra></extra>",
                    yaxis="y1",
                ))
            fig.add_trace(go.Scatter(
                x=season_df[x_label], y=season_df["avg_sentiment"],
                name=f"Avg Sentiment ({season})", mode="lines+markers",
                line=dict(color=_season_shade(SENTIMENT_SHADES, seasons, season)),
                text=hover_extra,
                hovertemplate=("%{x}<br>%{text}<br>Sentiment: %{y:.2f}<extra></extra>"
                               if gconf["has_points"] else "%{x}<br>Sentiment: %{y:.2f}<extra></extra>"),
                yaxis="y2" if gconf["has_points"] else "y1",
            ))

        fig.update_layout(
            title=player,
            xaxis=dict(title=f"Season · {x_col}"),
            yaxis=dict(title="FPL Points" if gconf["has_points"] else "Sentiment (-1..1)", side="left"),
            yaxis2=dict(title="Sentiment (-1..1)", side="right", overlaying="y", range=[-1, 1]) if gconf["has_points"] else None,
            legend=dict(orientation="h", y=1.15),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        if gconf["has_points"]:
            render_correlation_matrix(player_df, player)


if __name__ == "__main__":
    main()