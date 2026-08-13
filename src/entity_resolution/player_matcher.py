"""Associa un commento Reddit a uno o piu' giocatori del roster.

Approccio: dizionario di alias per giocatore costruito automaticamente
dal roster FPL (web_name, nome+cognome) + alias custom opzionali per
soprannomi che i dati FPL non contengono (es. "mo salah", "vvd").

Matching a due livelli:
  1. match esatto su confine di parola (alta precisione, veloce)
  2. fallback fuzzy (rapidfuzz) sui singoli token, solo per alias
     di almeno 4 caratteri, per assorbire refusi comuni ("salahh")

Nota onesta: nessun entity resolver di questo tipo e' perfetto.
Soprannomi ambigui, sarcasmo, e commenti che parlano "della squadra"
senza nominare nessuno restano fuori scope - vedi README.
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
    # player_id -> set di alias in lowercase
    aliases: dict[int, set[str]] = field(default_factory=dict)
    # player_id -> nome da mostrare (web_name)
    display_name: dict[int, str] = field(default_factory=dict)

    def all_alias_pairs(self) -> list[tuple[int, str]]:
        return [(pid, alias) for pid, al_set in self.aliases.items() for alias in al_set]


def _clean(name: str) -> str:
    text = re.sub(r"\s+", " ", str(name)).strip().lower()
    # tolgo punteggiatura iniziale/finale (es. "Diogo J." -> "diogo j"):
    # se un alias finisce con un carattere non alfanumerico, il \b di
    # confine-parola nel matching non scatta piu' e l'alias non matcha mai
    return text.strip(".-'")


def build_alias_index(
    players_df: pd.DataFrame,
    custom_aliases: dict[str, list[str]] | None = None,
) -> PlayerAliasIndex:
    """Costruisce l'indice alias -> player_id da un roster FPL.

    players_df deve avere almeno le colonne: id, web_name, first_name, second_name.
    custom_aliases: mappa web_name -> lista di alias extra, es.
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

        # scarta alias troppo corti/generici (es. cognomi di 2 lettere)
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
    """Ritorna la lista di player_id citati nel commento (puo' essere vuota o multipla)."""
    text = _clean(comment_text)
    if not text:
        return []

    matched: set[int] = set()

    # 1) match esatto a confine di parola, alias piu' lunghi prima
    #    (evita che un alias corto "mangi" parte di uno piu' specifico)
    pairs = sorted(index.all_alias_pairs(), key=lambda p: -len(p[1]))
    for pid, alias in pairs:
        if re.search(rf"\b{re.escape(alias)}\b", text):
            matched.add(pid)

    if matched or not use_fuzzy:
        return sorted(matched)

    # 2) fallback fuzzy sui singoli token, solo se il match esatto non ha trovato nulla
    tokens = re.findall(r"[a-zà-ÿ]+", text)
    for pid, alias in pairs:
        if len(alias) < MIN_FUZZY_ALIAS_LEN or " " in alias:
            continue
        for token in tokens:
            if len(token) < MIN_FUZZY_ALIAS_LEN:
                continue
            if fuzz.ratio(alias, token) >= fuzzy_threshold:
                matched.add(pid)
                break

    return sorted(matched)
