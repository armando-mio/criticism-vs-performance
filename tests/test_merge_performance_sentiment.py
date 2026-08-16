import pandas as pd

from src.aggregation.merge_performance_sentiment import (
    aggregate_performance,
    aggregate_sentiment,
    build_gold_dataset,
)


def test_aggregate_performance_on_real_liverpool_data():
    gws = pd.read_csv("data/raw/fpl/2024-25/gameweeks.csv")
    agg = aggregate_performance(gws)

    assert not agg.empty
    assert {"player_id", "player_name", "month", "total_points"}.issubset(agg.columns)
    # Salah e' stato capocannoniere della stagione 2024-25: deve comparire
    # con piu' di un mese di dati e punti totali complessivamente alti
    salah_rows = agg[agg["player_name"].str.contains("Salah", case=False, na=False)]
    assert len(salah_rows) > 1
    assert salah_rows["total_points"].sum() > 100


def test_aggregate_sentiment_basic():
    tagged = pd.DataFrame(
        [
            {"player_id": 1, "created_utc": 1723852800, "sentiment_compound": 0.8},
            {"player_id": 1, "created_utc": 1723852900, "sentiment_compound": -0.6},
            {"player_id": 1, "created_utc": 1723853000, "sentiment_compound": 0.1},
        ]
    )
    agg = aggregate_sentiment(tagged)
    assert len(agg) == 1  # tutti e tre i commenti nello stesso mese
    row = agg.iloc[0]
    assert row["n_comments"] == 3
    assert abs(row["negative_share"] - (1 / 3)) < 1e-6


def test_build_gold_dataset_outer_join_keeps_unmatched_rows():
    perf = pd.DataFrame(
        [{"player_id": 1, "player_name": "Salah", "month": "2024-09", "total_points": 20}]
    )
    sent = pd.DataFrame(
        [{"player_id": 2, "month": "2024-10", "avg_sentiment": -0.3, "n_comments": 5, "negative_share": 0.4}]
    )
    gold = build_gold_dataset(perf, sent)
    # entrambe le righe devono sopravvivere, anche se non condividono la chiave
    assert len(gold) == 2
    assert set(gold["player_id"]) == {1, 2}
