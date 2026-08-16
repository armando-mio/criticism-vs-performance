"""ATTENZIONE: genera commenti SINTETICI (frasi template, non veri commenti
Reddit) per poter testare l'intera pipeline end-to-end senza dover
raggiungere Arctic Shift.

Non e' una fonte dati reale. Serve solo a verificare che
pipeline_tag_and_score.py e merge_performance_sentiment.py producano un
gold dataset sensato, prima di collegare i dati veri raccolti con
fetch_reddit_dump.py.

Uso:
    python scripts/make_demo_reddit_data.py --season 2024-25
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import data_dir, load_config  # noqa: E402

POSITIVE_TEMPLATES = [
    "{player} was outstanding today, best on the pitch",
    "{player} is having an incredible season, so consistent",
    "what a finish from {player}, world class",
    "{player} covered every blade of grass, immense performance",
]
NEGATIVE_TEMPLATES = [
    "{player} was poor again, can't keep his place in the team",
    "{player} gave the ball away constantly, really disappointing",
    "not good enough from {player} today, way off the pace",
    "{player} looks completely out of form, needs dropping",
]
NEUTRAL_TEMPLATES = [
    "{player} started on the right wing today",
    "{player} came on as a substitute in the second half",
    "{player} played the full 90 minutes",
]


def generate(season_id: str, n_per_player: int = 20, seed: int = 42) -> list[dict]:
    cfg = load_config()
    season_cfg = cfg["seasons"][season_id]
    players_path = data_dir("raw", season_id).parent / "fpl" / season_id / "players.csv"
    players_df = pd.read_csv(players_path)

    start = datetime.strptime(season_cfg["start_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(season_cfg["end_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    span_days = (end - start).days

    rng = random.Random(seed)
    comments = []
    cid = 0
    for _, row in players_df.iterrows():
        name = row["web_name"]
        for _ in range(n_per_player):
            bucket = rng.choices(
                ["pos", "neg", "neu"], weights=[0.4, 0.35, 0.25]
            )[0]
            template = rng.choice(
                {"pos": POSITIVE_TEMPLATES, "neg": NEGATIVE_TEMPLATES, "neu": NEUTRAL_TEMPLATES}[bucket]
            )
            offset_days = rng.randint(0, max(span_days, 1))
            created = start + timedelta(days=offset_days, seconds=rng.randint(0, 86400))
            cid += 1
            comments.append(
                {
                    "author": f"demo_user_{cid % 50}",
                    "body": template.format(player=name),
                    "created_utc": int(created.timestamp()),
                    "id": f"demo{season_id.replace('-', '')}{cid}",
                    "link_id": "t3_demo",
                    "score": rng.randint(-5, 40),
                    "subreddit": cfg["team"]["reddit_subreddit"],
                }
            )
    return comments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True)
    parser.add_argument("--n-per-player", type=int, default=20)
    args = parser.parse_args()

    comments = generate(args.season, n_per_player=args.n_per_player)

    out_dir = data_dir("raw", args.season).parent / "reddit" / args.season
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comments.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in comments:
            f.write(json.dumps(c) + "\n")

    print(f"[{args.season}] {len(comments)} commenti SINTETICI (demo) -> {out_path}")


if __name__ == "__main__":
    main()
