from src.classification.sentiment_baseline import score_comment, score_comments_batch


def test_clearly_positive_comment():
    result = score_comment("Salah was absolutely brilliant, best player in the league!")
    assert result["label"] == "positive"
    assert result["compound"] > 0


def test_clearly_negative_comment():
    result = score_comment("Salah is finished, embarrassing performance, get him off")
    assert result["label"] == "negative"
    assert result["compound"] < 0


def test_neutral_factual_comment():
    result = score_comment("Salah started the match against Chelsea")
    assert result["label"] == "neutral"


def test_empty_comment_is_neutral():
    result = score_comment("")
    assert result["label"] == "neutral"
    assert result["compound"] == 0.0


def test_batch_matches_single_scoring():
    texts = ["Great goal!", "Terrible performance", ""]
    batch = score_comments_batch(texts)
    singles = [score_comment(t) for t in texts]
    assert batch == singles
