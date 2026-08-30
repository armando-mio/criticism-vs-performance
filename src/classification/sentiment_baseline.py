"""Baseline sentiment classification for a comment.

Uses VADER (vaderSentiment), designed for social/informal text - the same
family of lexicons often used as a starting point before moving to a
transformer model fine-tuned on manually labeled data.

Requires no training or labeled data: works out-of-the-box, making it
well-suited as an initial step for a small scope like this (one team, two
seasons) before investing in labeling + validation (kappa) + fine-tuning,
which remain the natural next step if higher accuracy is needed for football
fan-specific slang.
"""
from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Domain-specific football sentiment terms (lexicon enrichment)
FOOTBALL_LEXICON = {
    # High positive praise
    "masterclass": 3.4,
    "world-class": 3.4,
    "worldclass": 3.4,
    "baller": 2.8,
    "clutch": 3.0,
    "clinical": 2.6,
    "unplayable": 3.2,
    "immense": 2.8,
    "beast": 2.4,
    "maestro": 2.7,
    "class": 2.2,
    "legend": 2.8,
    "goat": 3.0,
    "golazo": 3.2,
    "solid": 2.0,
    "composed": 2.0,
    "workhorse": 2.2,
    "superb": 3.0,
    # High negative criticism
    "shambolic": -3.2,
    "disasterclass": -3.5,
    "abysmal": -3.3,
    "bottled": -2.8,
    "bottler": -2.8,
    "bottling": -2.8,
    "stinker": -2.9,
    "flop": -2.7,
    "fraud": -3.0,
    "dreadful": -3.0,
    "washed": -2.8,
    "sitter": -2.5,
    "liability": -3.0,
    "clueless": -2.8,
    "invisible": -2.2,
    "shocking": -2.8,
    "horrendous": -3.2,
    "useless": -3.0,
    "pathetic": -3.2,
    "atrocious": -3.4,
    "woeful": -3.0,
    "subpar": -2.0,
}
_analyzer.lexicon.update(FOOTBALL_LEXICON)

# standard thresholds recommended by VADER authors
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


import re


def _preprocess_football_text(text: str) -> str:
    """Normalizes multi-word football expressions and handles inverting idioms."""
    cleaned = str(text)
    cleaned = re.sub(r"\bworld\s+class\b", "world-class", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfar\s+from\s+", "not ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhardly\s+a\s+", "not a ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bnothing\s+short\s+of\s+a\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned



def score_comment(text: str) -> dict:
    """Returns compound score (-1..1) and discrete label for a comment."""
    if not text or not str(text).strip():
        return {"compound": 0.0, "label": "neutral"}

    processed = _preprocess_football_text(text)
    scores = _analyzer.polarity_scores(processed)
    compound = scores["compound"]

    if compound >= POSITIVE_THRESHOLD:
        label = "positive"
    elif compound <= NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return {"compound": compound, "label": label}


def score_comments_batch(texts: list[str]) -> list[dict]:
    return [score_comment(t) for t in texts]

