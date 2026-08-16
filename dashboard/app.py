"""Dashboard per esplorare la relazione performance/critica dei giocatori
del Liverpool nelle due stagioni configurate.

Uso:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import data_dir, load_config, season_ids  # noqa: E402

st.set_page_config(page_title="Performance vs Toxicity - Liverpool", layout="wide")


@st.cache_data
def load_gold_data(seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season_id in seasons:
        path = data_dir("gold", season_id) / "player_month_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["season"] = season_id
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cfg = load_config()
    seasons = season_ids(cfg)
    team_name = cfg["team"]["name"]

    st.title(f"{team_name}: performance vs critica su Reddit")
    st.caption(
        " / ".join(f"{s} — {cfg['seasons'][s]['label']}" for s in seasons)
    )

    data = load_gold_data(seasons)
    if data.empty:
        st.warning(
            "Nessun dataset gold trovato. Esegui prima la pipeline "
            "(fetch -> tag_and_score -> merge_performance_sentiment)."
        )
        return

    all_players = sorted(data["player_name"].dropna().unique())
    default_player = all_players[0] if all_players else None

    col_a, col_b = st.columns([1, 2])
    with col_a:
        selected_seasons = st.multiselect("Stagioni", seasons, default=seasons)
    with col_b:
        selected_players = st.multiselect(
            "Giocatori", all_players, default=[default_player] if default_player else []
        )

    filtered = data[
        data["season"].isin(selected_seasons) & data["player_name"].isin(selected_players)
    ].sort_values(["player_name", "season", "month"])

    if filtered.empty:
        st.info("Seleziona almeno un giocatore per vedere i grafici.")
        return

    for player in selected_players:
        player_df = filtered[filtered["player_name"] == player]
        if player_df.empty:
            continue

        player_df = player_df.copy()
        player_df["season_month"] = player_df["season"] + " · " + player_df["month"]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=player_df["season_month"],
                y=player_df["total_points"],
                name="Punti FPL (mese)",
                marker_color="#1D9E75",
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=player_df["season_month"],
                y=player_df["avg_sentiment"],
                name="Sentiment medio Reddit",
                mode="lines+markers",
                line=dict(color="#D85A30"),
                yaxis="y2",
            )
        )
        fig.update_layout(
            title=player,
            xaxis=dict(title="Stagione · mese"),
            yaxis=dict(title="Punti FPL", side="left"),
            yaxis2=dict(title="Sentiment (-1..1)", side="right", overlaying="y", range=[-1, 1]),
            legend=dict(orientation="h", y=1.15),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander(f"Dati grezzi — {player}"):
            st.dataframe(player_df, use_container_width=True)


if __name__ == "__main__":
    main()
