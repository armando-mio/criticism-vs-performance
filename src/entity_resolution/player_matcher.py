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

import unicodedata

MIN_FUZZY_ALIAS_LEN = 4
FUZZY_THRESHOLD = 88

# Unambiguous nicknames and aliases for Liverpool squad
DEFAULT_LIVERPOOL_ALIASES: dict[str, list[str]] = {
    "M.Salah": ["mo salah", "salah", "the egyptian king", "egyptian king"],
    "Salah": ["mo salah", "salah", "the egyptian king", "egyptian king"],
    "A.Becker": ["alisson", "becker", "alisson becker"],
    "Alexander-Arnold": ["trent", "trent alexander-arnold", "taa", "alexander arnold"],
    "Virgil": ["virgil", "van dijk", "vvd", "virgil van dijk"],
    "Darwin": ["darwin", "nunez", "núñez", "darwin nunez", "darwin núñez"],
    "Diogo J.": ["jota", "diogo jota", "diogo j"],
    "Luis Díaz": ["luis diaz", "luis díaz", "diaz", "díaz", "lucho"],
    "Mac Allister": ["mac allister", "macca", "alexis mac allister"],
    "Szoboszlai": ["szoboszlai", "szobo", "dominik szoboszlai"],
    "Gravenberch": ["gravenberch", "ryan gravenberch"],
    "Konaté": ["konate", "konaté", "ibou", "ibrahima konate", "ibrahima konaté"],
    "Gakpo": ["gakpo", "cody gakpo"],
    "Robertson": ["robertson", "robbo", "andy robertson"],
    "Jones": ["curtis", "curtis jones"],
    "C.Jones": ["curtis", "curtis jones"],
    "Elliott": ["elliott", "harvey elliott"],
    "Endo": ["endo", "wataru endo"],
    "Gomez": ["joe gomez"],
    "Bradley": ["conor bradley", "bradley"],
    "Tsimikas": ["tsimikas", "kostas", "greek scouser"],
    "Kelleher": ["kelleher", "caoimhin", "kweev"],
    "Quansah": ["quansah", "jarell quansah"],
    "Chiesa": ["chiesa", "federico chiesa"],
    "Wirtz": ["wirtz", "florian wirtz", "flo wirtz"],
    "Isak": ["isak", "alexander isak"],
    "Frimpong": ["frimpong", "jeremie frimpong"],
    "Kerkez": ["kerkez", "milos kerkez"],
    "Slot": ["arne slot", "slot"],
}


@dataclass
class PlayerAliasIndex:
    # player_id -> set of lowercase aliases
    aliases: dict[int, set[str]] = field(default_factory=dict)
    # player_id -> display name (web_name)
    display_name: dict[int, str] = field(default_factory=dict)

    def all_alias_pairs(self) -> list[tuple[int, str]]:
        return [(pid, alias) for pid, al_set in self.aliases.items() for alias in al_set]


def _strip_accents(text: str) -> str:
    """Removes diacritics / accents from text (e.g. Núñez -> Nunez)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _clean(name: str) -> str:
    text = re.sub(r"\s+", " ", str(name)).strip().lower()
    # strip leading/trailing punctuation
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
    combined_custom = dict(DEFAULT_LIVERPOOL_ALIASES)
    if custom_aliases:
        for k, v in custom_aliases.items():
            if k in combined_custom:
                combined_custom[k] = list(set(combined_custom[k] + v))
            else:
                combined_custom[k] = v

    # Common generic words/names that should not be used as standalone single-token aliases
    GENERIC_TOKENS = {
        "will", "ben", "joe", "harvey", "james", "ryan", "conor", "calvin",
        "mo", "ali", "dom", "dan", "rob", "lee", "sam", "tony", "may", "long", "short"
    }

    for _, row in players_df.iterrows():
        pid = int(row["id"])
        web_name = str(row["web_name"])
        raw_aliases = set()

        raw_aliases.add(web_name)
        raw_aliases.add(_strip_accents(web_name))

        first = str(row.get("first_name", "")) if pd.notna(row.get("first_name")) else ""
        second = str(row.get("second_name", "")) if pd.notna(row.get("second_name")) else ""

        if first and second:
            raw_aliases.add(f"{first} {second}")
            raw_aliases.add(_strip_accents(f"{first} {second}"))

        if second:
            raw_aliases.add(second)
            raw_aliases.add(_strip_accents(second))
            # Split compound surnames (e.g. "Núñez Ribeiro" -> "Núñez", "Ribeiro")
            for part in second.split():
                if len(part) >= 3 and part.lower() not in GENERIC_TOKENS:
                    raw_aliases.add(part)
                    raw_aliases.add(_strip_accents(part))

        if first and len(first) >= 3 and first.lower() not in GENERIC_TOKENS:
            raw_aliases.add(first)
            raw_aliases.add(_strip_accents(first))

        # Check default / custom aliases matching web_name or second_name or first_name
        for lookup_key in [web_name, second, first, f"{first} {second}"]:
            if lookup_key in combined_custom:
                for extra in combined_custom[lookup_key]:
                    raw_aliases.add(extra)
                    raw_aliases.add(_strip_accents(extra))

        cleaned_aliases = set()
        for a in raw_aliases:
            cl = _clean(a)
            if len(cl) >= 3 and cl not in GENERIC_TOKENS and cl not in {"in", "at", "on", "to", "he", "is", "a", "an", "the"}:
                cleaned_aliases.add(cl)

        if pid in idx.aliases:
            idx.aliases[pid] |= cleaned_aliases
        else:
            idx.aliases[pid] = cleaned_aliases
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
