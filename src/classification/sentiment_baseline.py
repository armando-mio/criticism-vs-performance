"""Classificazione baseline del sentiment di un commento.

Usa VADER (vaderSentiment), pensato per testo social/informale - stessa
famiglia di lessici usata spesso come punto di partenza prima di passare
a un modello transformer fine-tuned su dati etichettati manualmente.

Non richiede training ne' dati etichettati: funziona out-of-the-box,
il che lo rende adatto come primo passaggio per uno scope piccolo come
questo (una squadra, due stagioni) prima di investire in labeling +
validazione (kappa) + fine-tuning, che restano lo step successivo se
si vuole piu' accuratezza sullo slang specifico dei tifosi.
"""
from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# soglie standard raccomandate dagli autori di VADER
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def score_comment(text: str) -> dict:
    """Ritorna compound score (-1..1) ed etichetta discreta per un commento."""
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
