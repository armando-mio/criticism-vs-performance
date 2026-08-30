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


def test_football_lexicon_enrichment():
    res_pos = score_comment("What a masterclass from Trent, absolute baller!")
    assert res_pos["label"] == "positive"
    assert res_pos["compound"] > 0.5

    res_neg = score_comment("Shambolic defending from Konate, complete disasterclass")
    assert res_neg["label"] == "negative"
    assert res_neg["compound"] < -0.5


def test_negations_and_idiom_inversion():
    res_not_masterclass = score_comment("Salah did not have a masterclass today")
    assert res_not_masterclass["label"] == "negative"
    assert res_not_masterclass["compound"] < 0

    res_far_from_clinical = score_comment("Darwin was far from clinical today")
    assert res_far_from_clinical["label"] == "negative"
    assert res_far_from_clinical["compound"] < 0

    res_hardly_baller = score_comment("Hardly a baller performance")
    assert res_hardly_baller["label"] == "negative"
    assert res_hardly_baller["compound"] < 0

