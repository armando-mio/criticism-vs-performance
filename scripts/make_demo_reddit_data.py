"""Generates realistic synthetic comments (anchored to FPL matchweeks and player
performance) to test the entire pipeline end-to-end with genuine statistical variance
and correlation between performance and sentiment.

Usage:
    python scripts/make_demo_reddit_data.py --season 2024-25
    python scripts/make_demo_reddit_data.py --season 2025-26
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
from src.entity_resolution.player_matcher import DEFAULT_LIVERPOOL_ALIASES  # noqa: E402

HIGH_PRAISE_TEMPLATES = [
    "{player} was world-class today, best on the pitch by a mile!",
    "What a masterclass from {player}! Absolute baller.",
    "{player} is having an incredible season, so clutch when we need it.",
    "immense performance from {player}, covered every blade of grass.",
    "Unplayable today. {player} is pure quality, what a goal!",
    "Brilliant display by {player}, composure and vision were unmatched.",
    "{player} with a superstar performance, legend in the making.",
    "Outstanding defensively and offensively from {player}, 10/10.",
]

MODERATE_PRAISE_TEMPLATES = [
    "Solid game from {player}, did what was asked.",
    "{player} worked hard today and got a well-deserved assist.",
    "Good shift from {player}, steady in possession.",
    "Nice to see {player} back in rhythm, decent performance.",
    "Reliable as always, {player} kept things tidy in midfield.",
]

NEUTRAL_TEMPLATES = [
    "{player} started on the pitch today, subbed off around 70 mins.",
    "Tactical adjustment with {player} tucking inside.",
    "{player} played 90 minutes, interesting role today.",
    "Curious to see if {player} starts again next week.",
    "Standard performance from {player}, nothing flashy.",
]

MODERATE_CRITICISM_TEMPLATES = [
    "{player} was a bit off the pace today, needs to sharpen up.",
    "Quiet game for {player}, didn't really influence the match much.",
    "{player} gave the ball away a few times under pressure.",
    "Expected a bit more from {player} in the final third today.",
    "{player} looked tired in the second half.",
]

SEVERE_CRITICISM_TEMPLATES = [
    "{player} was shocking today, complete stinker of a performance.",
    "Shambolic defending from {player}, cost us the game.",
    "{player} missed an absolute sitter, completely unacceptable.",
    "Really poor from {player}, looks completely out of form and washed.",
    "A disasterclass from {player}, needs to be dropped next match.",
    "{player} is a liability at the back right now.",
    "{player} was invisible all 90 minutes, shocking display.",
]


def _get_player_preferred_name(web_name: str) -> str:
    """Returns a natural name or common nickname for template generation."""
    if web_name in DEFAULT_LIVERPOOL_ALIASES:
        aliases = DEFAULT_LIVERPOOL_ALIASES[web_name]
        # Choose from natural names/nicknames (capitalize)
        candidates = [a.title() for a in aliases if len(a) >= 3 and not a.startswith("the ")]
        if candidates:
            return candidates[0]
    return web_name


def generate(
    season_id: str,
    base_comments_per_gw: int = 15,
    seed: int = 42,
) -> list[dict]:
    cfg = load_config()
    season_cfg = cfg["seasons"][season_id]
    fpl_base = data_dir("raw", season_id).parent / "fpl" / season_id
    players_path = fpl_base / "players.csv"
    gameweeks_path = fpl_base / "gameweeks.csv"

    players_df = pd.read_csv(players_path)
    gameweeks_df = pd.read_csv(gameweeks_path)

    gameweeks_df["kickoff_time"] = pd.to_datetime(gameweeks_df["kickoff_time"], utc=True)
    rounds = sorted(gameweeks_df["round"].unique())

    rng = random.Random(seed)
    comments = []
    cid = 0

    id_to_webname = dict(zip(players_df["id"], players_df["web_name"]))

    for round_num in rounds:
        gw_rows = gameweeks_df[gameweeks_df["round"] == round_num]
        if gw_rows.empty:
            continue

        kickoff_min = gw_rows["kickoff_time"].min()

        for _, prow in gw_rows.iterrows():
            pid = int(prow["element"])
            web_name = id_to_webname.get(pid, str(prow.get("name", "Player")))
            display_name = _get_player_preferred_name(web_name)

            minutes = float(prow.get("minutes", 0))
            points = float(prow.get("total_points", 0))
            goals = float(prow.get("goals_scored", 0))
            assists = float(prow.get("assists", 0))
            yellows = float(prow.get("yellow_cards", 0))
            reds = float(prow.get("red_cards", 0))

            # Determine comment volume and realistic sentiment distribution with natural noise
            if minutes == 0:
                n_comments = rng.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]
                base_weights = [0.10, 0.20, 0.50, 0.15, 0.05]
            elif points >= 10:
                n_comments = rng.randint(base_comments_per_gw + 5, base_comments_per_gw + 25)
                # Realistic praise with 20% critical/cynical fans
                base_weights = [0.45, 0.30, 0.12, 0.09, 0.04]
            elif points >= 6:
                n_comments = rng.randint(base_comments_per_gw - 2, base_comments_per_gw + 12)
                base_weights = [0.32, 0.33, 0.18, 0.12, 0.05]
            elif points in [3, 4, 5]:
                n_comments = rng.randint(base_comments_per_gw - 5, base_comments_per_gw + 5)
                base_weights = [0.18, 0.27, 0.28, 0.18, 0.09]
            elif points in [1, 2]:
                n_comments = rng.randint(base_comments_per_gw - 3, base_comments_per_gw + 6)
                base_weights = [0.10, 0.18, 0.24, 0.30, 0.18]
            else:
                n_comments = rng.randint(base_comments_per_gw, base_comments_per_gw + 15)
                base_weights = [0.06, 0.12, 0.18, 0.34, 0.30]

            # Add stochastic perturbation per player-match to reflect subjective match perception
            noise = [rng.uniform(-0.05, 0.05) for _ in range(5)]
            sentiment_weights = [max(0.01, w + n) for w, n in zip(base_weights, noise)]
            total_w = sum(sentiment_weights)
            sentiment_weights = [w / total_w for w in sentiment_weights]

            # Generate individual comments
            for _ in range(n_comments):
                cid += 1
                category = rng.choices(
                    ["high_pos", "mod_pos", "neu", "mod_neg", "sev_neg"],
                    weights=sentiment_weights,
                )[0]

                template_pool = {
                    "high_pos": HIGH_PRAISE_TEMPLATES,
                    "mod_pos": MODERATE_PRAISE_TEMPLATES,
                    "neu": NEUTRAL_TEMPLATES,
                    "mod_neg": MODERATE_CRITICISM_TEMPLATES,
                    "sev_neg": SEVERE_CRITICISM_TEMPLATES,
                }[category]

                template = rng.choice(template_pool)
                body = template.format(player=display_name)

                # Timing: 10% pre-match (within 2h before), 30% match (0-2h after kickoff), 60% post-match (2-48h after kickoff)
                time_bucket = rng.choices(["pre", "live", "post"], weights=[0.15, 0.35, 0.50])[0]
                if time_bucket == "pre":
                    offset_sec = -rng.randint(600, 7200)
                elif time_bucket == "live":
                    offset_sec = rng.randint(0, 7200)
                else:
                    offset_sec = rng.randint(7200, 172800)

                created = kickoff_min + timedelta(seconds=offset_sec)
                score = rng.randint(1, 45) if category in ["high_pos", "sev_neg"] else rng.randint(-3, 20)

                comments.append(
                    {
                        "author": f"fan_user_{rng.randint(1, 500)}",
                        "body": body,
                        "created_utc": int(created.timestamp()),
                        "id": f"demo_{season_id.replace('-', '')}_{cid:06d}",
                        "link_id": f"t3_match_gw{round_num}",
                        "score": score,
                        "subreddit": cfg["team"]["reddit_subreddit"],
                    }
                )

    # Sort chronologically
    comments.sort(key=lambda c: c["created_utc"])
    return comments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True)
    parser.add_argument("--base-per-gw", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    comments = generate(args.season, base_comments_per_gw=args.base_per_gw, seed=args.seed)

    out_dir = data_dir("raw", args.season).parent / "reddit" / args.season
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comments.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in comments:
            f.write(json.dumps(c) + "\n")

    print(f"[{args.season}] Generated {len(comments)} realistic synthetic comments -> {out_path}")


if __name__ == "__main__":
    main()

