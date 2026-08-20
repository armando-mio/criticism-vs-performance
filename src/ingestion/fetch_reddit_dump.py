"""Downloads all comments from a subreddit within a date window,
using the Arctic Shift search endpoint (not Reddit's live API).

Official docs: https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md
Endpoint used: GET /api/comments/search
  parameters: subreddit, after, before, limit (max 100), sort=asc, fields

Why this approach and not PRAW/OAuth:
  - this is historical data for two seasons that are already over, nothing live is needed
  - Arctic Shift doesn't require a Reddit account or OAuth
  - the "serious" limits of the service apply to very heavy queries; for
    a single subreddit over a ~9 month window, requests of 100
    comments with a small pause between requests is enough
    and stays within the spirit of "few requests per second" of the service

Note for whoever runs this script: arctic-shift.photon-reddit.com is not
reachable from the sandbox this project was written and tested in
(network restricted to a domain allowlist). The script is written
and the pagination logic is tested with mock responses that replicate
the real schema confirmed via a direct query to the API (see tests/), but
it needs to be run on your own machine for the first real data collection.

Usage:
    python -m src.ingestion.fetch_reddit_dump --season 2024-25
    python -m src.ingestion.fetch_reddit_dump --season 2024-25 --restart

By default it resumes from where a previous run was interrupted (reads
the last comment already saved in comments.jsonl and continues from there),
so a transient error from the service doesn't force you to redo
thousands of requests from scratch. Use --restart to ignore the existing
file and start over from the beginning of the season.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import data_dir, load_config  # noqa: E402

FIELDS = "author,body,created_utc,id,link_id,score,subreddit"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 3.0  # seconds: attempt n waits base * 2^(n-1)


def _to_epoch(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _last_created_utc(jsonl_path: Path) -> int | None:
    """Reads the last comment already saved to figure out where to resume from.

    Tolerates a possibly truncated last line (e.g. process interrupted
    mid-write) by discarding it and looking at the previous one.
    """
    if not jsonl_path.exists():
        return None

    with open(jsonl_path, "rb") as f:
        try:
            f.seek(-4096, 2)
        except OSError:
            f.seek(0)
        tail = f.read().decode("utf-8", errors="ignore")

    for line in reversed(tail.strip().splitlines()):
        try:
            return int(json.loads(line)["created_utc"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return None


def fetch_page(
    session: requests.Session,
    base_url: str,
    subreddit: str,
    after: int,
    before: int,
    limit: int,
) -> list[dict]:
    """A single call to the search endpoint, with retry on transient errors.

    Isolated (one call = one full attempt including retry) so it
    stays easily mockable in tests: whoever mocks it doesn't need to know
    anything about the internal retry logic.
    """
    params = {
        "subreddit": subreddit,
        "after": after,
        "before": before,
        "limit": limit,
        "sort": "asc",
        "fields": FIELDS,
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except (requests.exceptions.RequestException,) as exc:
            last_error = exc
            body_preview = ""
            if getattr(exc, "response", None) is not None:
                body_preview = exc.response.text[:300]
            wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            print(
                f"  attempt {attempt}/{MAX_RETRIES} failed ({exc}). "
                f"Response: {body_preview!r}. Retrying in {wait:.0f}s..."
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    raise RuntimeError(
        f"Unable to fetch page after {MAX_RETRIES} attempts "
        f"(after={after}, before={before}): {last_error}"
    )


def fetch_subreddit_comments(
    subreddit: str,
    start_date: str,
    end_date: str,
    base_url: str,
    page_size: int = 100,
    delay_seconds: float = 1.0,
    session: requests.Session | None = None,
    start_override_epoch: int | None = None,
) -> Iterator[dict]:
    """Generates all comments from a subreddit between start_date and end_date (inclusive).

    Paginates by advancing the 'after' cursor to the created_utc of the last
    comment received + 1 second, until a page comes back with fewer results
    than the requested page_size (signal that data has ended).

    start_override_epoch, if passed, overrides start_date as the starting
    cursor (used to resume an interrupted download).
    """
    session = session or requests.Session()
    after = start_override_epoch if start_override_epoch is not None else _to_epoch(start_date)
    before = _to_epoch(end_date) + 24 * 3600  # include the entire final day

    while True:
        page = fetch_page(session, base_url, subreddit, after, before, page_size)
        if not page:
            return

        for comment in page:
            yield comment

        if len(page) < page_size:
            return

        after = int(page[-1]["created_utc"]) + 1
        time.sleep(delay_seconds)


def run(season_id: str, restart: bool = False) -> Path:
    cfg = load_config()
    season_cfg = cfg["seasons"][season_id]
    reddit_cfg = cfg["reddit"]
    subreddit = cfg["team"]["reddit_subreddit"]

    out_dir = data_dir("raw", season_id).parent / "reddit" / season_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comments.jsonl"

    resume_from = None if restart else _last_created_utc(out_path)
    start_date = season_cfg["start_date"]

    if resume_from is not None:
        print(
            f"[{season_id}] found previous progress, resuming from "
            f"created_utc={resume_from} instead of season start"
        )
        file_mode = "a"
        start_override_epoch = resume_from + 1
    else:
        if restart:
            print(f"[{season_id}] --restart: restarting from season start")
        file_mode = "w"
        start_override_epoch = None

    count = 0
    with open(out_path, file_mode, encoding="utf-8") as f:
        comments_iter = fetch_subreddit_comments(
            subreddit=subreddit,
            start_date=start_date,
            end_date=season_cfg["end_date"],
            base_url=reddit_cfg["base_url"],
            page_size=reddit_cfg["page_size"],
            delay_seconds=reddit_cfg["request_delay_seconds"],
            start_override_epoch=start_override_epoch,
        )
        for comment in comments_iter:
            f.write(json.dumps(comment) + "\n")
            f.flush()  # so a crash mid-way doesn't lose more than the few ms in progress
            count += 1
            if count % 500 == 0:
                print(f"[{season_id}] {count} comments downloaded in this run...")

    print(f"[{season_id}] total {count} new comments in this run -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="e.g. 2024-25")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore saved progress and restart from the beginning of the season",
    )
    args = parser.parse_args()
    run(args.season, restart=args.restart)