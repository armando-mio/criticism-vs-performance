"""Matches a Reddit comment to one or more squad players.

Approach: player alias dictionary built automatically from the FPL roster
(web_name, first+last name) + optional custom aliases for nicknames not
present in FPL data (e.g. "mo salah", "vvd").

Two-level matching:
  1. exact word-boundary match (high precision, fast)
  2. fuzzy fallback (rapidfuzz) on individual tokens, only for aliases
     of at least 4 characters, to absorb common typos ("salahh")

Honest note: no entity resolver of this type is perfect. Ambiguous nicknames,
sarcasm, and comments discussing "the team" without naming anyone remain
out of scope - see README.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz

MIN_FUZZY_ALIAS_LEN = 4
FUZZY_THRESHOLD = 88


@dataclass
class PlayerAliasIndex:
    # player_id -> set of lowercase aliases
    aliases: dict[int, set[str]] = field(default_factory=dict)
    # player_id -> display name (web_name)
    display_name: dict[int, str] = field(default_factory=dict)

    def all_alias_pairs(self) -> list[tuple[int, str]]:
        return [(pid, alias) for pid, al_set in self.aliases.items() for alias in al_set]


def _clean(name: str) -> str:
    text = re.sub(r"\s+", " ", str(name)).strip().lower()
    # strip leading/trailing punctuation (e.g. "Diogo J." -> "diogo j"):
    # if an alias ends with a non-alphanumeric character, the word-boundary
    # \b in regex matching will not trigger and the alias will never match
    return text.strip(".-'")


def build_alias_index(
    players_df: pd.DataFrame,
    custom_aliases: dict[str, list[str]] | None = None,
) -> PlayerAliasIndex:
    """Builds the alias -> player_id index from an FPL roster.

    players_df must have at least the columns: id, web_name, first_name, second_name.
    custom_aliases: mapping web_name -> list of extra aliases, e.g.
        {"Salah": ["mo salah", "the egyptian king"]}
    """
    idx = PlayerAliasIndex()
    custom_aliases = custom_aliases or {}

    for _, row in players_df.iterrows():
        pid = int(row["id"])
        web_name = str(row["web_name"])
        aliases = {_clean(web_name)}

        if "first_name" in row and "second_name" in row:
            full_name = f"{row['first_name']} {row['second_name']}"
            aliases.add(_clean(full_name))
            if pd.notna(row.get("second_name")):
                aliases.add(_clean(row["second_name"]))

        for extra in custom_aliases.get(web_name, []):
            aliases.add(_clean(extra))

        # discard aliases that are too short/generic (e.g. 2-letter surnames)
        aliases = {a for a in aliases if len(a) >= 3}

        if pid in idx.aliases:
            idx.aliases[pid] |= aliases
        else:
            idx.aliases[pid] = aliases
            idx.display_name[pid] = web_name

    return idx


def match_players(
    comment_text: str,
    index: PlayerAliasIndex,
    use_fuzzy: bool = True,
    fuzzy_threshold: int = FUZZY_THRESHOLD,
) -> list[int]:
    """Returns the list of player_ids mentioned in the comment (can be empty or multiple)."""
    text = _clean(comment_text)
    if not text:
        return []

    matched: set[int] = set()

    # 1) exact word-boundary match, longer aliases first
    #    (prevents a shorter alias from consuming part of a more specific one)
    pairs = sorted(index.all_alias_pairs(), key=lambda p: -len(p[1]))
    for pid, alias in pairs:
        if re.search(rf"\b{re.escape(alias)}\b", text):
            matched.add(pid)

    if matched or not use_fuzzy:
        return sorted(matched)

    # 2) fuzzy fallback on individual tokens, only if exact match found nothing
    tokens = re.findall(r"[a-zà-ÿ]+", text)
    for pid, alias in pairs:
        if len(alias) < MIN_FUZZY_ALIAS_LEN or " " in alias:
            continue
        for token in tokens:
            if len(token) < MIN_FUZZY_ALIAS_LEN:
                continue
            # first letter must match: a genuine typo (e.g. "salahh") never
            # changes the first letter, while a common word that happens to
            # closely resemble a short surname (e.g. "right" vs "Wright")
            # almost always does - this is the most frequent false-positive case
            if token[0] != alias[0]:
                continue
            if fuzz.ratio(alias, token) >= fuzzy_threshold:
                matched.add(pid)
                break

    return sorted(matched)
