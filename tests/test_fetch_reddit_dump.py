import json
from unittest.mock import MagicMock, patch

import requests

from src.ingestion.fetch_reddit_dump import (
    _last_created_utc,
    _to_epoch,
    fetch_page,
    fetch_subreddit_comments,
    run,
)


def _fake_comment(created_utc: int, cid: str) -> dict:
    # same schema confirmed by querying the real API (see README):
    # author, body, created_utc, id, link_id, score, subreddit
    return {
        "author": "some_user",
        "body": f"comment {cid}",
        "created_utc": created_utc,
        "id": cid,
        "link_id": "t3_abc123",
        "score": 1,
        "subreddit": "LiverpoolFC",
    }


def test_pagination_stops_when_page_smaller_than_limit():
    """With page_size=2: first page full (2), second page partial (1) -> stop."""
    page1 = [_fake_comment(1000, "a"), _fake_comment(1001, "b")]
    page2 = [_fake_comment(1002, "c")]

    call_count = {"n": 0}

    def fake_fetch_page(session, base_url, subreddit, after, before, limit):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return page1
        elif call_count["n"] == 2:
            return page2
        return []

    with patch("src.ingestion.fetch_reddit_dump.fetch_page", side_effect=fake_fetch_page):
        results = list(
            fetch_subreddit_comments(
                subreddit="LiverpoolFC",
                start_date="2024-08-16",
                end_date="2024-08-17",
                base_url="https://fake",
                page_size=2,
                delay_seconds=0,
            )
        )

    assert [c["id"] for c in results] == ["a", "b", "c"]
    assert call_count["n"] == 2  # should not make an unnecessary third call


def test_pagination_advances_cursor_past_last_comment():
    """The 'after' cursor of the second call must be last_created_utc + 1."""
    page1 = [_fake_comment(1000, "a"), _fake_comment(1001, "b")]
    seen_afters = []

    def fake_fetch_page(session, base_url, subreddit, after, before, limit):
        seen_afters.append(after)
        if len(seen_afters) == 1:
            return page1
        return []  # second call: no more data

    with patch("src.ingestion.fetch_reddit_dump.fetch_page", side_effect=fake_fetch_page):
        list(
            fetch_subreddit_comments(
                subreddit="LiverpoolFC",
                start_date="2024-08-16",
                end_date="2024-08-17",
                base_url="https://fake",
                page_size=2,
                delay_seconds=0,
            )
        )

    assert seen_afters[1] == 1001 + 1


def test_empty_first_page_yields_nothing():
    def fake_fetch_page(session, base_url, subreddit, after, before, limit):
        return []

    with patch("src.ingestion.fetch_reddit_dump.fetch_page", side_effect=fake_fetch_page):
        results = list(
            fetch_subreddit_comments(
                subreddit="LiverpoolFC",
                start_date="2024-08-16",
                end_date="2024-08-17",
                base_url="https://fake",
                page_size=100,
                delay_seconds=0,
            )
        )
    assert results == []


def test_to_epoch_is_utc():
    # 2024-08-16 00:00:00 UTC
    assert _to_epoch("2024-08-16") == 1723766400


# --- retry on transient errors (the real 422 case in production) ---


def _mock_response(status_code: int, json_data: dict | None = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        error = requests.exceptions.HTTPError(f"{status_code} error", response=resp)
        resp.raise_for_status.side_effect = error
    else:
        resp.raise_for_status.side_effect = None
    return resp


def test_fetch_page_retries_then_succeeds(monkeypatch):
    """A transient 422/500 followed by a success should not make fetch_page fail."""
    monkeypatch.setattr("time.sleep", lambda _: None)  # no real waiting in tests

    fail_response = _mock_response(422, text='{"error": "temporary"}')
    ok_response = _mock_response(200, {"data": [{"id": "a", "created_utc": 1000}]})

    session = MagicMock()
    session.get.side_effect = [fail_response, ok_response]

    result = fetch_page(session, "https://fake", "LiverpoolFC", after=1, before=2, limit=100)

    assert result == [{"id": "a", "created_utc": 1000}]
    assert session.get.call_count == 2


def test_fetch_page_raises_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("src.ingestion.fetch_reddit_dump.MAX_RETRIES", 2)

    always_fail = _mock_response(500, text="server error")
    session = MagicMock()
    session.get.return_value = always_fail

    try:
        fetch_page(session, "https://fake", "LiverpoolFC", after=1, before=2, limit=100)
        assert False, "should have raised RuntimeError"
    except RuntimeError as exc:
        assert "2 tentativi" in str(exc)
    assert session.get.call_count == 2


# --- resuming an interrupted download ---


def test_last_created_utc_missing_file_returns_none(tmp_path):
    assert _last_created_utc(tmp_path / "does_not_exist.jsonl") is None


def test_last_created_utc_reads_last_line(tmp_path):
    path = tmp_path / "comments.jsonl"
    lines = [
        json.dumps({"id": "a", "created_utc": 1000}),
        json.dumps({"id": "b", "created_utc": 2000}),
        json.dumps({"id": "c", "created_utc": 3000}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert _last_created_utc(path) == 3000


def test_last_created_utc_skips_truncated_last_line(tmp_path):
    """Simulates a crash mid-write: the last line is truncated JSON."""
    path = tmp_path / "comments.jsonl"
    good_line = json.dumps({"id": "b", "created_utc": 2000})
    truncated = '{"id": "c", "created_utc": 30'  # truncated, no closing
    path.write_text(good_line + "\n" + truncated, encoding="utf-8")

    assert _last_created_utc(path) == 2000


# --- run() end-to-end: exactly the 422-mid-download scenario ---


def test_run_resumes_from_existing_partial_file(tmp_path, monkeypatch):
    """Simulates exactly the real case: 2 comments already saved, then an
    error midway through a previous run. run() must resume from the
    correct cursor and APPEND to the file, not overwrite it."""
    season_id = "2024-25"

    fake_cfg = {
        "team": {"reddit_subreddit": "LiverpoolFC"},
        "seasons": {
            season_id: {"start_date": "2024-08-16", "end_date": "2024-08-17"}
        },
        "reddit": {
            "base_url": "https://fake",
            "page_size": 100,
            "request_delay_seconds": 0,
        },
    }
    monkeypatch.setattr("src.ingestion.fetch_reddit_dump.load_config", lambda: fake_cfg)

    # data_dir("raw", season_id).parent / "reddit" / season_id / "comments.jsonl"
    # -> rebuild the same path used by run(), but inside tmp_path
    fake_raw_dir = tmp_path / "data" / "raw"
    monkeypatch.setattr(
        "src.ingestion.fetch_reddit_dump.data_dir",
        lambda kind, sid=None: (fake_raw_dir / sid) if sid else fake_raw_dir,
    )

    out_path = fake_raw_dir / "reddit" / season_id / "comments.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"id": "old1", "created_utc": 1723895741}) + "\n",
        encoding="utf-8",
    )

    seen_afters = []

    def fake_fetch_page(session, base_url, subreddit, after, before, limit):
        seen_afters.append(after)
        if len(seen_afters) == 1:
            return [{"id": "new1", "created_utc": 1723895800}]
        return []

    with patch("src.ingestion.fetch_reddit_dump.fetch_page", side_effect=fake_fetch_page):
        run(season_id)

    # the cursor must resume right after the last comment already saved
    assert seen_afters[0] == 1723895741 + 1

    # the file must contain both the old comment and the new one (append, not overwrite)
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == ["old1", "new1"]


def test_run_restart_ignores_existing_file(tmp_path, monkeypatch):
    season_id = "2024-25"
    fake_cfg = {
        "team": {"reddit_subreddit": "LiverpoolFC"},
        "seasons": {
            season_id: {"start_date": "2024-08-16", "end_date": "2024-08-17"}
        },
        "reddit": {
            "base_url": "https://fake",
            "page_size": 100,
            "request_delay_seconds": 0,
        },
    }
    monkeypatch.setattr("src.ingestion.fetch_reddit_dump.load_config", lambda: fake_cfg)

    fake_raw_dir = tmp_path / "data" / "raw"
    monkeypatch.setattr(
        "src.ingestion.fetch_reddit_dump.data_dir",
        lambda kind, sid=None: (fake_raw_dir / sid) if sid else fake_raw_dir,
    )

    out_path = fake_raw_dir / "reddit" / season_id / "comments.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"id": "old1", "created_utc": 999}) + "\n", encoding="utf-8")

    def fake_fetch_page(session, base_url, subreddit, after, before, limit):
        return []

    with patch("src.ingestion.fetch_reddit_dump.fetch_page", side_effect=fake_fetch_page):
        run(season_id, restart=True)

    # with --restart the old content must disappear (file truncated, not appended)
    assert out_path.read_text(encoding="utf-8").strip() == ""