import pandas as pd

from src.entity_resolution.player_matcher import build_alias_index, match_players


def make_roster() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": 1, "web_name": "Salah", "first_name": "Mohamed", "second_name": "Salah"},
            {"id": 2, "web_name": "Van Dijk", "first_name": "Virgil", "second_name": "van Dijk"},
            {"id": 3, "web_name": "Alexander-Arnold", "first_name": "Trent", "second_name": "Alexander-Arnold"},
        ]
    )


def test_exact_match_web_name():
    idx = build_alias_index(make_roster())
    assert match_players("Salah was unplayable today", idx) == [1]


def test_exact_match_full_name():
    idx = build_alias_index(make_roster())
    assert match_players("Mohamed Salah scored again", idx) == [1]


def test_multiple_players_same_comment():
    idx = build_alias_index(make_roster())
    result = match_players("Salah and Van Dijk were both poor", idx)
    assert result == [1, 2]


def test_custom_alias():
    custom = {"Salah": ["mo salah", "the egyptian king"]}
    idx = build_alias_index(make_roster(), custom_aliases=custom)
    assert match_players("mo salah is finished", idx) == [1]


def test_no_match_returns_empty():
    idx = build_alias_index(make_roster())
    assert match_players("what a boring 0-0 draw", idx) == []


def test_fuzzy_catches_common_typo():
    idx = build_alias_index(make_roster())
    # "salahh" non e' un match esatto, deve arrivarci il fallback fuzzy
    assert match_players("salahh should start every game", idx) == [1]


def test_fuzzy_disabled_misses_typo():
    idx = build_alias_index(make_roster())
    assert match_players("salahh should start every game", idx, use_fuzzy=False) == []


def test_web_name_ending_in_period_still_matches():
    # regressione: alias che finisce con punteggiatura (es. "Diogo J.")
    # non deve rompere il controllo di confine-parola
    roster = pd.DataFrame(
        [{"id": 4, "web_name": "Diogo J.", "first_name": "Diogo", "second_name": "Jota"}]
    )
    idx = build_alias_index(roster)
    assert match_players("Diogo J. was electric off the bench", idx) == [4]


def test_substring_does_not_false_positive_on_unrelated_word():
    idx = build_alias_index(make_roster())
    # "arnold" e' cognome di Alexander-Arnold: deve matchare solo l'alias giusto,
    # non confondersi con parole che contengono "van" (es. "advantage")
    assert match_players("what an advantage we have", idx) == []
