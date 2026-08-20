# performance-vs-toxicity

One-shot batch pipeline: analyzing the relationship between on-pitch performance
and Reddit criticism for Liverpool players across two completed seasons:

- **2024-25** — title-winning season (Aug 16, 2024 – May 25, 2025)
- **2025-26** — season finished in 5th place (Aug 15, 2025 – May 24, 2026)

No continuous scraping, no scheduler: download once, process, and explore in the dashboard.

## Verification Status: What Was Tested and What Was Not

To be fully transparent about what is verified and what needs your verification:

| Component | Status |
|---|---|
| `fetch_fpl_season.py` | **Executed with real data** in this environment. `data/raw/fpl/` already contains the actual CSVs for both seasons (roster + gameweek), downloaded from vaastav/Fantasy-Premier-League. |
| `player_matcher.py` (entity resolution) | Tested with unit tests, including a real bug found and fixed (aliases ending with a period, e.g. "Diogo J.", broke matching — see `test_web_name_ending_in_period_still_matches`). |
| `sentiment_baseline.py` | Tested, uses VADER (no training required). |
| `merge_performance_sentiment.py` | Tested against real Liverpool FPL data. |
| `fetch_reddit_dump.py` | Written against the official Arctic Shift API documentation and verified against the actual endpoint to confirm the response schema. Pagination logic is covered by tests with mock responses. **Not run end-to-end**: `arctic-shift.photon-reddit.com` was not reachable from the restricted network environment where this project was developed. See below for instructions on running it. |
| `dashboard/app.py` | Verified with Streamlit (`streamlit run`), responds with HTTP 200 without errors. Visual content can be explored with real or demo data. |

The delivered repository **does not include raw Reddit comments**, neither real nor fake:
only the real FPL CSVs are pre-populated. `scripts/make_demo_reddit_data.py` generates
synthetic data (clearly labeled templated sentences) so you can see the pipeline
run end-to-end before connecting real data.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run Everything

```bash
# 1. Performance (real data, already included, but re-executable)
python -m src.ingestion.fetch_fpl_season

# 2. Reddit — download real comments from r/LiverpoolFC for both seasons
#    (no account/OAuth required, only the public Arctic Shift API)
python -m src.ingestion.fetch_reddit_dump --season 2024-25
python -m src.ingestion.fetch_reddit_dump --season 2025-26

# 3. Entity resolution + sentiment (silver)
python -m src.pipeline_tag_and_score --season 2024-25
python -m src.pipeline_tag_and_score --season 2025-26

# 4. Aggregation (gold)
python -m src.aggregation.merge_performance_sentiment

# 5. Dashboard
streamlit run dashboard/app.py
```

To test everything WITHOUT waiting for step 2 (using synthetic data, see above):

```bash
python scripts/make_demo_reddit_data.py --season 2024-25
python scripts/make_demo_reddit_data.py --season 2025-26
python -m src.pipeline_tag_and_score --season 2024-25
python -m src.pipeline_tag_and_score --season 2025-26
python -m src.aggregation.merge_performance_sentiment
streamlit run dashboard/app.py
```

## Tests

```bash
pytest tests/ -v
```

## Project Structure

```
performance-vs-toxicity/
├── config/seasons.yaml          # season dates, team, endpoints
├── data/
│   ├── raw/fpl/<season>/        # real CSVs (already included)
│   ├── raw/reddit/<season>/     # comments.jsonl (to generate, see above)
│   ├── silver/<season>/         # tagged_comments.csv
│   └── gold/<season>/           # player_month_summary.csv
├── src/
│   ├── ingestion/                # fetch_fpl_season.py, fetch_reddit_dump.py
│   ├── entity_resolution/        # player_matcher.py
│   ├── classification/           # sentiment_baseline.py
│   ├── aggregation/              # merge_performance_sentiment.py
│   └── pipeline_tag_and_score.py # connects entity resolution + sentiment
├── scripts/make_demo_reddit_data.py   # synthetic data generator for testing
├── dashboard/app.py              # Streamlit dashboard
└── tests/
```

## Known Limitations (read before drawing strong conclusions)

- **Entity resolution is not perfect.** Aliases are derived from the FPL roster +
  fuzzy fallback for common typos. Sarcasm, unofficial nicknames, and comments
  criticizing "the team" without naming individuals remain out of scope. For higher
  precision, expand `custom_aliases` in `config/seasons.yaml` (currently unpopulated — add if needed).
- **VADER is a baseline, not a fine-tuned model.** Good for quick insights, but
  less accurate for football fan slang compared to a transformer fine-tuned on
  labeled domain data (the natural next step if you want to invest in labeling + kappa).
- **`avg_sentiment` per month may have very few comments** in months with low activity
  — always check `n_comments` before drawing conclusions from an average.
- **Arctic Shift is a free service without uptime guarantees.** If
  `fetch_reddit_dump.py` times out over a wide window, reduce the window or retry
  — this is a service limitation, not a script bug.

