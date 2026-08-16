# performance-vs-toxicity

Pipeline batch one-shot: relazione tra performance in campo e critica su
Reddit per i giocatori del Liverpool, in due stagioni chiuse:

- **2024-25** — stagione dello scudetto (16 ago 2024 – 25 mag 2025)
- **2025-26** — stagione attuale, chiusa al 5° posto (15 ago 2025 – 24 mag 2026)

Nessuna raccolta continua, nessuno scheduler: si scarica una volta, si
processa, si esplora in dashboard.

## Cosa e' stato testato davvero e cosa no

Per essere onesti su cosa puoi fidarti al 100% e cosa va verificato tu:

| Componente | Stato |
|---|---|
| `fetch_fpl_season.py` | **Eseguito con dati reali** in questo ambiente. `data/raw/fpl/` contiene gia' i CSV veri di entrambe le stagioni (roster + gameweek), scaricati da vaastav/Fantasy-Premier-League. |
| `player_matcher.py` (entity resolution) | Testato con 22 unit test, incluso un bug reale trovato e corretto (alias che finiscono con un punto, es. "Diogo J.", rompevano il match — vedi `test_web_name_ending_in_period_still_matches`). |
| `sentiment_baseline.py` | Testato, usa VADER (nessun training necessario). |
| `merge_performance_sentiment.py` | Testato anche contro i dati FPL reali del Liverpool. |
| `fetch_reddit_dump.py` | Scritto contro la documentazione ufficiale dell'API di Arctic Shift e verificato interrogando l'endpoint reale per confermare lo schema di risposta. La logica di paginazione e' coperta da test con risposte mock. **Non eseguito end-to-end**: `arctic-shift.photon-reddit.com` non e' raggiungibile dalla rete dell'ambiente in cui questo progetto e' stato scritto. Vedi sotto per come lanciarlo. |
| `dashboard/app.py` | Avviato per davvero (`streamlit run`), risponde HTTP 200 senza errori. Il contenuto visivo va verificato da te con dati reali. |

Nella repo consegnata **non ci sono commenti Reddit**, ne' veri ne' finti:
solo i CSV FPL reali. `scripts/make_demo_reddit_data.py` genera dati
sintetici (frasi template, chiaramente etichettate) solo per farti
vedere la pipeline girare end-to-end prima di collegare i dati veri.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Come eseguire tutto

```bash
# 1. Performance (reale, gia' incluso, ma rieseguibile)
python -m src.ingestion.fetch_fpl_season

# 2. Reddit — scarica i commenti veri di r/LiverpoolFC per le due stagioni
#    (nessun account/OAuth richiesto, solo l'API pubblica di Arctic Shift)
python -m src.ingestion.fetch_reddit_dump --season 2024-25
python -m src.ingestion.fetch_reddit_dump --season 2025-26

# 3. Entity resolution + sentiment (silver)
python -m src.pipeline_tag_and_score --season 2024-25
python -m src.pipeline_tag_and_score --season 2025-26

# 4. Aggregazione (gold)
python -m src.aggregation.merge_performance_sentiment

# 5. Dashboard
streamlit run dashboard/app.py
```

Per provare tutto SENZA aspettare il passo 2 (dati sintetici, vedi sopra):

```bash
python scripts/make_demo_reddit_data.py --season 2024-25
python scripts/make_demo_reddit_data.py --season 2025-26
python -m src.pipeline_tag_and_score --season 2024-25
python -m src.pipeline_tag_and_score --season 2025-26
python -m src.aggregation.merge_performance_sentiment
streamlit run dashboard/app.py
```

## Test

```bash
pytest tests/ -v
```

## Struttura

```
performance-vs-toxicity/
├── config/seasons.yaml          # date stagioni, squadra, endpoint
├── data/
│   ├── raw/fpl/<season>/        # CSV reali (gia' inclusi)
│   ├── raw/reddit/<season>/     # comments.jsonl (da generare, vedi sopra)
│   ├── silver/<season>/         # tagged_comments.csv
│   └── gold/<season>/           # player_month_summary.csv
├── src/
│   ├── ingestion/                # fetch_fpl_season.py, fetch_reddit_dump.py
│   ├── entity_resolution/        # player_matcher.py
│   ├── classification/           # sentiment_baseline.py
│   ├── aggregation/              # merge_performance_sentiment.py
│   └── pipeline_tag_and_score.py # collega entity resolution + sentiment
├── scripts/make_demo_reddit_data.py   # dati sintetici, solo per testare
├── dashboard/app.py              # Streamlit
└── tests/
```

## Limitazioni note (leggi prima di fidarti troppo dei numeri)

- **Entity resolution non e' perfetta.** Alias derivati dal roster FPL +
  fallback fuzzy per refusi comuni. Sarcasmo, nickname non ufficiali,
  commenti che criticano "la squadra" senza nominare nessuno restano
  fuori scope. Se vuoi piu' precisione, arricchisci `custom_aliases` in
  `config/seasons.yaml` (attualmente non popolato — aggiungilo se serve).
- **VADER e' un baseline, non un modello fine-tuned.** Buono per farsi
  un'idea rapida, meno accurato dello slang specifico dei tifosi di
  calcio rispetto a un transformer addestrato su dati etichettati (lo
  step naturale successivo, se vuoi investire in labeling + kappa).
- **`avg_sentiment` per mese puo' avere pochissimi commenti** in mesi con
  poca attivita' — controlla sempre `n_comments` prima di trarre
  conclusioni da una media.
- **Arctic Shift e' un servizio gratuito senza garanzie di uptime.** Se
  `fetch_reddit_dump.py` va in timeout su una finestra ampia, riducila
  o riprova — non e' un bug dello script.
