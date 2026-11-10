"""
Fetch public Steam reviews for V Rising (appid 1604030).

Note on endpoint: Valve's historically-documented public endpoint
(store.steampowered.com/appreviews/<id>?json=1) now 302-redirects to the
store homepage -- it appears to have been quietly retired. Inspecting the
network traffic of the actual V Rising store page (via browser devtools)
shows the review widget itself now calls a differently-named endpoint on
the same storefront host:

    store.steampowered.com/ajaxappreviews/<id>

This is still Valve's own public storefront code calling its own storefront
host to render reviews on the page anyone can view -- no login, no API key,
no private data. It just requires a `sessionid` cookie (handed out to any
visitor) and a Referer header, which is normal browser behavior, not an
authentication bypass. We mirror exactly what the page itself does: load
the store page once to receive a sessionid cookie, then page through
reviews with that session.

Only the fields needed for analysis are kept: review text, recommended
flag, timestamp, playtime at the time of review, and vote counts. No
author name, SteamID, profile URL, or avatar is stored anywhere.

Usage:
    python fetch_reviews.py --max-reviews 10000 --out ../data/reviews_raw.jsonl
"""
import argparse
import json
import sys
import time

import requests

APP_ID = 1604030
STORE_PAGE_URL = f"https://store.steampowered.com/app/{APP_ID}/"
REVIEWS_URL = f"https://store.steampowered.com/ajaxappreviews/{APP_ID}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": STORE_PAGE_URL,
        }
    )
    # Visiting the store page once hands out the sessionid cookie that
    # ajaxappreviews expects, exactly as a real browser would get it.
    s.get(STORE_PAGE_URL, timeout=30)
    return s


def fetch_page(session: requests.Session, cursor: str, num_per_page: int = 100) -> dict:
    params = {
        "filter": "recent",
        "language": "english",
        "cursor": cursor,
        "num_per_page": num_per_page,
        "purchase_type": "all",
        "review_type": "all",
        "playtime_type": "all",
        "filter_offtopic_activity": 1,
        "use_review_quality": 1,
    }
    resp = session.get(REVIEWS_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def slim(review: dict) -> dict:
    """Keep only the fields the analysis needs. No author/ID data."""
    return {
        "review_text": review.get("review", ""),
        "recommended": review.get("voted_up"),
        "timestamp_created": review.get("timestamp_created"),
        "playtime_at_review_minutes": review.get("author", {}).get(
            "playtime_at_review"
        ),
        "votes_up": review.get("votes_up"),
        "votes_funny": review.get("votes_funny"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-reviews", type=int, default=10000)
    ap.add_argument("--num-per-page", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--out", type=str, default="../data/reviews_raw.jsonl")
    args = ap.parse_args()

    session = make_session()
    cursor = "*"
    seen_cursors = set()
    collected = 0

    with open(args.out, "w", encoding="utf-8") as fh:
        while collected < args.max_reviews:
            try:
                data = fetch_page(session, cursor, args.num_per_page)
            except requests.RequestException as e:
                print(f"Request failed: {e}. Stopping.", file=sys.stderr)
                break

            if data.get("success") != 1:
                print(f"Stopping: API returned success={data.get('success')}", file=sys.stderr)
                break

            reviews = data.get("reviews", [])
            if not reviews:
                print("Stopping: no more reviews returned.", file=sys.stderr)
                break

            for r in reviews:
                fh.write(json.dumps(slim(r), ensure_ascii=False) + "\n")
            collected += len(reviews)
            fh.flush()

            print(f"Fetched {collected} reviews so far (cursor={cursor!r})", file=sys.stderr)

            next_cursor = data.get("cursor")
            if not next_cursor or next_cursor == cursor or next_cursor in seen_cursors:
                print("Stopping: cursor did not advance (end of pages).", file=sys.stderr)
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

            time.sleep(args.sleep)

    print(f"Done. Wrote {collected} reviews to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
