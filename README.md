# V Rising — Public Steam Review Analysis

A small, honest portfolio analysis of public Steam reviews for *V Rising*
(Steam app id `1604030`). Built as a thesis-application portfolio piece — not
an official or endorsed analysis.

**This project uses only public data and is not affiliated with, endorsed
by, or produced in cooperation with Stunlock Studios.** All review text
belongs to the Steam users who wrote it; it is used here transiently (not
redistributed) for aggregate, anonymized analysis.

## What this is

- Review volume and % positive, by week, with the two major content
  updates marked (1.0 Launch — 8 May 2024, Invaders of Oakveil — 28 Apr 2025)
- Recurring topics in negative reviews via TF-IDF + NMF, with short
  representative phrases per topic (not full quotes)
- How playtime-at-review differs between positive and negative reviews

See `notebook/analysis.ipynb` for the full analysis and
`v_rising_review_analysis_summary.pdf` for a one-page summary.

## What this explicitly is not

- Not a representative survey of the playerbase — Steam reviewers are a
  self-selected, vocal subset.
- English-language reviews only.
- Review text is stated opinion at a point in time, not behavioral
  telemetry. It can't show silent churn, actual session structure, or
  drop-off points the way internal analytics could.

No numbers in this repo are estimated or invented — everything is computed
directly from the fetched dataset, and the fetch/analysis code is included
so every number is reproducible.

## Data

Reviews are fetched directly from Steam's own storefront (no login, no API
key, no scraping of private data) via `scripts/fetch_reviews.py`. Only
review text, recommended/not, timestamp, playtime at time of review, and
vote counts are kept — no usernames, SteamIDs, or other author identifiers.

Raw review data is **not committed to this repo** (see `.gitignore`) since
it's third-party user-generated content pulled live from Steam, not
something to redistribute wholesale. Regenerate it locally:

```bash
pip install -r requirements.txt
cd scripts
python fetch_reviews.py --max-reviews 8000 --out ../data/reviews_raw.jsonl
```

This respects rate limits (a configurable sleep between paginated requests,
default 1.5s) and paginates via Steam's cursor mechanism.

## Reproducing the analysis

```bash
cd scripts
python build_notebook.py        # (re)generates notebook/analysis.ipynb
jupyter notebook ../notebook/analysis.ipynb
python make_pdf_summary.py      # regenerates the one-page PDF + images/
```

## Repo structure

```
scripts/
  fetch_reviews.py      # pulls reviews from Steam, strips to needed fields
  analysis.py           # shared analysis functions (volume, topics, playtime)
  build_notebook.py      # generates the notebook programmatically
  make_pdf_summary.py    # builds the one-page PDF summary
notebook/
  analysis.ipynb         # the full walkthrough
images/                  # chart PNGs used in the notebook and PDF
data/                    # raw review data (gitignored, regenerate locally)
```
