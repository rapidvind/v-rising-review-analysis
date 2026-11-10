"""Generates notebook/analysis.ipynb programmatically so cell content stays
in sync with scripts/analysis.py. Run once; the notebook is then a normal
editable .ipynb from there on."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# V Rising: Public Steam Review Analysis

A small, honest look at public Steam reviews for *V Rising* (Steam app id
`1604030`), built as a portfolio piece.

**Data source:** Steam's own storefront review data, fetched with
`scripts/fetch_reviews.py` (see that file for a note on the endpoint).
English-language reviews only. No usernames, SteamIDs, or other author
identifiers are collected or stored: only review text, recommended
yes/no, timestamp, playtime at time of review, and vote counts.

**Not affiliated with or endorsed by Stunlock Studios.** This is an
independent analysis of public data, built to demonstrate an analysis
workflow, not an official or authoritative account of the game's
reception.
"""
))

cells.append(nbf.v4.new_code_cell(
"""import sys
sys.path.append("../scripts")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from analysis import (
    MAJOR_UPDATES,
    MINOR_EVENTS,
    load_reviews,
    weekly_volume_and_positivity,
    playtime_by_recommendation,
    topic_model_negative_reviews,
)

pd.set_option("display.max_colwidth", 120)
DATA_PATH = "../data/reviews_raw.jsonl"
"""
))

cells.append(nbf.v4.new_markdown_cell("## Load data"))

cells.append(nbf.v4.new_code_cell(
"""df = load_reviews(DATA_PATH)
print(f"{len(df):,} reviews loaded")
print(f"Date range: {df['created_at'].min().date()} to {df['created_at'].max().date()}")
print(f"Overall % positive: {df['recommended'].mean() * 100:.1f}%")
df.head(3)
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 1. Review volume and % positive per week

Dashed red lines mark the two major content updates: **1.0 Launch** (8 May
2024) and **Invaders of Oakveil** (28 April 2025). Dotted gray lines mark
other volume spikes large enough to need an explanation, labeled in the
legend only where we could verify a cause against Steam/patch-note
sources (left to right):

- **1.0.9.0 patch**: "Spells Balance, Quality of Life & Gamepad
  Improvements", released 17 Sep 2024.
- **Steam Autumn Sale**: ran 27 Nov-4 Dec 2024 (platform-wide, not
  V Rising-specific).
- **Free Weekend + 1.1.1 patch**: Free Weekend promotion ran 14-18 May
  2026 alongside patch 1.1.1; the dip in % positive that week is
  consistent with a large wave of new/trial players.

Any other, smaller fluctuations in the chart are *not* individually
explained here: we only label what we could verify, per
`MINOR_EVENTS`/`MINOR_EVENTS_DETAIL` in `analysis.py`."""
))

cells.append(nbf.v4.new_code_cell(
"""weekly = weekly_volume_and_positivity(df)

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax1.bar(weekly["created_at"], weekly["num_reviews"], width=5.5, color="#6B6E76", alpha=0.35)
ax2.plot(weekly["created_at"], weekly["pct_positive_rolling4"], color="#3D5A80", linewidth=1.8)

for _, date_str in MAJOR_UPDATES:
    d = mdates.datestr2num(date_str)
    ax1.axvline(d, color="#B3493B", linestyle="--", linewidth=1.2)
for _, date_str in MINOR_EVENTS:
    d = mdates.datestr2num(date_str)
    ax1.axvline(d, color="#6B6E76", linestyle=":", linewidth=1)

legend_handles = [
    Line2D([0], [0], color="#B3493B", linestyle="--", linewidth=1.2, label="Major content update"),
    Line2D([0], [0], color="#6B6E76", linestyle=":", linewidth=1, label="Other verified spike"),
]
ax1.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=8, framealpha=0.85)

ax1.set_ylabel("Reviews / week")
ax2.set_ylabel("% positive")
ax2.set_ylim(0, 100)
ax1.set_title("V Rising: weekly review volume and % positive")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate(rotation=30)
fig.tight_layout()
fig.savefig("../images/weekly_volume_positivity.png", dpi=160)
plt.show()
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""**Reading this honestly:** the line above is a **4-week rolling %
positive**, not the raw per-week value: a plain per-week line would
occasionally show a misleading spike or dip just from a small sample in a
quiet week (weeks below 20 reviews are masked out of the raw series for
the same reason; the final, still-accumulating week is dropped entirely
rather than shown as a fake near-100%/0% point).

Volume spikes are the clearest, least ambiguous signal of "something
happened" around an update: a launch, a big content drop, a sale, or a
free weekend reliably pulls in a burst of reviews, and that burst decays
back to baseline within roughly 6-8 weeks. The %-positive line, by
contrast, moves surprisingly little across the same events and stays in a
fairly narrow band, so volume is the more informative signal here than
sentiment."""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2. Recurring topics in negative reviews

TF-IDF + NMF topic modeling over negative (`recommended == False`)
reviews only. For each topic: a short descriptive label (derived from the
top weighted words, not a separately-sourced claim), the top weighted
words themselves, and 2-3 short representative *phrases* (not full review
quotes) pulled from the reviews that load most heavily on that topic.
Profanity is filtered out of both the model vocabulary and the phrases."""
))

cells.append(nbf.v4.new_code_cell(
"""topics = topic_model_negative_reviews(df, n_topics=6, n_top_words=8, n_examples=3)

for t in topics:
    print(f"{t['label']}")
    print(f"  top words: " + ", ".join(t["top_words"]))
    for p in t["example_phrases"]:
        print(f"    - \\"{p}\\"")
    print()
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""**Caveat on topics:** NMF topics are unsupervised clusters of co-occurring
words, not ground truth categories: they need a human glance to confirm
they're coherent, and the short labels above are a plain-language gloss on
the model's top words, not independently verified category names. Treat
this as "here's where to look," not a definitive breakdown of why people
dislike the game. These same six themes, with one phrase each, are also on
the one-page PDF summary."""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 3. Playtime at review: positive vs. negative"""
))

cells.append(nbf.v4.new_code_cell(
"""playtime_stats = playtime_by_recommendation(df)
playtime_stats
"""
))

cells.append(nbf.v4.new_code_cell(
"""fig, ax = plt.subplots(figsize=(9, 3.5))
pos_hours = df.loc[df["recommended"], "playtime_at_review_hours"].clip(upper=400)
neg_hours = df.loc[~df["recommended"], "playtime_at_review_hours"].clip(upper=400)

bp = ax.boxplot([pos_hours, neg_hours], vert=False, tick_labels=["Positive", "Negative"],
                 patch_artist=True, widths=0.5, showfliers=False)
for patch, color in zip(bp["boxes"], ["#2E7D5B", "#B3493B"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

ax.set_xlabel("Playtime at time of review (hours, clipped at 400 for readability)")
ax.set_title("Playtime at review: positive vs. negative")
fig.tight_layout()
fig.savefig("../images/playtime_by_recommendation.png", dpi=160)
plt.show()
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""**Reading this honestly:** median playtime at review is higher for
positive reviews than negative ones. That's consistent with people who
dislike the game saying so relatively early, while people who stick
around long enough to write a glowing review have already gotten a lot
of hours out of it. It is *not* proof of why: reviews are a snapshot of
stated opinion at one point in time, not a measurement of play behavior
over time."""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Limitations (please read before citing any of this)

- **Self-selection bias.** Steam reviewers are a vocal, self-selected
  subset of players. People with mild or no opinion rarely write a
  review at all, so this is not representative of the full playerbase.
- **English-only.** Reviews in every other language are excluded
  entirely, which skews the sample toward certain regions/communities.
- **Reviews are not telemetry.** A review is a stated opinion at a single
  moment, shaped by recency, mood, and what the reviewer chooses to
  mention. It is not a measurement of actual play behavior, retention,
  or churn.
- **Numbers here are computed directly from the fetched dataset**: see
  `scripts/fetch_reviews.py` and `scripts/analysis.py` for exactly how.
  Nothing in this notebook is estimated or invented.

**What internal telemetry could add:** the things reviews structurally
can't show, exact in-session drop-off points, return cadence after a
patch, and the large silent population that never writes a review at
all but still churns or sticks around.
"""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("../notebook/analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote notebook/analysis.ipynb")
