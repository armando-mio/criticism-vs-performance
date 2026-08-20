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

# standard thresholds recommended by VADER authors
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def score_comment(text: str) -> dict:
    """Returns compound score (-1..1) and discrete label for a comment."""
    if not text or not str(text).strip():
        return {"compound": 0.0, "label": "neutral"}

    scores = _analyzer.polarity_scores(str(text))
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
