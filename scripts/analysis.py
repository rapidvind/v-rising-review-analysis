"""
Shared analysis functions for the V Rising review dataset. Kept outside
the notebook so the logic is testable and the notebook stays readable.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Major content updates being marked on the timeline. Dates are UTC.
MAJOR_UPDATES = [
    ("1.0 Launch", "2024-05-08"),
    ("Invaders of Oakveil", "2025-04-28"),
]

# Secondary volume spikes that are large enough to need explanation but
# aren't one of the two major content updates above. Only events we could
# verify against Steam news / third-party patch-note trackers are labeled
# with a cause; everything else is left as "likely sales or events
# (unverified)" rather than guessed at.
MINOR_EVENTS = [
    ("1.0.9.0 patch (Sep '24)", "2024-09-17"),
    ("Steam Autumn Sale (Dec '24)", "2024-11-27"),
    ("Free Weekend + patch (May '26)", "2026-05-14"),
]

# Full detail behind each MINOR_EVENTS entry, for the notebook/README --
# kept out of the chart label itself to avoid clutter.
MINOR_EVENTS_DETAIL = [
    (
        "1.0.9.0 patch (Sep '24)",
        "Patch 1.0.9.0, \"Spells Balance, Quality of Life & Gamepad Improvements\", "
        "released 17 Sep 2024.",
    ),
    (
        "Steam Autumn Sale (Dec '24)",
        "Steam Autumn Sale ran 27 Nov - 4 Dec 2024 (platform-wide, not V Rising-specific).",
    ),
    (
        "Free Weekend + patch (May '26)",
        "V Rising Free Weekend promotion ran 14-18 May 2026, alongside patch 1.1.1; the "
        "dip in % positive that week is consistent with a large wave of new/trial players.",
    ),
]

MIN_REVIEWS_PER_WEEK = 20

STOPWORDS_EXTRA = {
    "game", "games", "rising", "v", "vrising", "play", "played", "playing",
    "player", "players", "just", "really", "like", "also", "even", "would",
    "get", "got", "still", "much", "many", "lot", "one", "time", "dont",
    "didnt", "im", "ive", "youre", "thats", "it", "https", "http", "com",
}

# Excluded from topic vocabulary entirely so profanity never becomes a
# topic-defining word or gets pulled into a representative phrase on the
# public-facing summary. Matched as a prefix (so "shitty"/"fucking"/etc.
# are caught too) -- deliberately excludes mild words like "ass"/"hell"
# that are also common substrings of harmless words (class, hello, ...).
PROFANITY_STEMS = ("shit", "fuck", "bullshit", "bitch", "piss")


def _has_profanity(word: str) -> bool:
    return any(word.startswith(stem) for stem in PROFANITY_STEMS)


def _is_mostly_latin(text: str, min_ratio: float = 0.9) -> bool:
    """Steam's language=english filter lets some non-English reviews through
    (mixed-language or mislabeled). Drop reviews that are mostly non-Latin
    script so topic modeling isn't contaminated by text it can't read."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    latin = sum(1 for c in letters if ord(c) < 592)  # covers Latin + extended Latin
    return (latin / len(letters)) >= min_ratio


def load_reviews(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["created_at"] = pd.to_datetime(df["timestamp_created"], unit="s", utc=True)
    df["playtime_at_review_hours"] = df["playtime_at_review_minutes"] / 60.0
    df["recommended"] = df["recommended"].astype(bool)
    # Basic cleaning: drop empty/near-empty review text and reviews that are
    # mostly non-Latin script (Steam's language=english filter isn't perfect).
    df = df[df["review_text"].str.strip().str.len() > 0].copy()
    df = df[df["review_text"].map(_is_mostly_latin)].copy()
    return df


def weekly_volume_and_positivity(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly review volume and % positive.

    The trailing week is dropped if it's a partial calendar week (fewer
    days of data than the review feed has actually reached), since a
    half-collected week can show an extreme, misleading % positive from a
    tiny sample. Any remaining week below MIN_REVIEWS_PER_WEEK reviews has
    its raw pct_positive masked (NaN) for the same reason; a 4-week
    trailing rolling % positive is provided as the noise-reduced series to
    plot instead.
    """
    weekly = (
        df.set_index("created_at")
        .resample("W")
        .agg(
            num_reviews=("recommended", "size"),
            num_positive=("recommended", "sum"),
        )
    )
    weekly["pct_positive"] = weekly["num_positive"] / weekly["num_reviews"] * 100
    weekly = weekly.reset_index()

    # Drop the final bin if it's a partial week (bin end is after the last
    # timestamp actually present in the data).
    if len(weekly) and weekly["created_at"].iloc[-1] > df["created_at"].max():
        weekly = weekly.iloc[:-1].copy()

    weekly["pct_positive_masked"] = weekly["pct_positive"].where(
        weekly["num_reviews"] >= MIN_REVIEWS_PER_WEEK
    )
    weekly["pct_positive_rolling4"] = (
        weekly["num_positive"].rolling(4, min_periods=1).sum()
        / weekly["num_reviews"].rolling(4, min_periods=1).sum()
        * 100
    )
    return weekly


def playtime_by_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("recommended")["playtime_at_review_hours"]
        .agg(["count", "median", "mean", "std"])
        .rename(index={True: "positive", False: "negative"})
    )


_token_re = re.compile(r"[a-zA-Z]{3,}")


def _tokenize(text: str) -> str:
    tokens = [t.lower() for t in _token_re.findall(text)]
    tokens = [t for t in tokens if t not in STOPWORDS_EXTRA and not _has_profanity(t)]
    return " ".join(tokens)


def topic_model_negative_reviews(
    df: pd.DataFrame, n_topics: int = 6, n_top_words: int = 8, n_examples: int = 3
):
    """TF-IDF + NMF topic model over negative reviews only.

    Returns a list of dicts: {topic_id, top_words, example_phrases}.
    example_phrases are short (<12 word) snippets pulled from the reviews
    that load most heavily on that topic -- not full review text.
    """
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    neg = df[~df["recommended"]].copy()
    neg["clean"] = neg["review_text"].map(_tokenize)
    neg = neg[neg["clean"].str.split().str.len() >= 3]

    vectorizer = TfidfVectorizer(
        max_df=0.5, min_df=5, ngram_range=(1, 2), max_features=5000,
        stop_words="english",
    )
    X = vectorizer.fit_transform(neg["clean"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    model = NMF(n_components=n_topics, random_state=42, max_iter=400)
    W = model.fit_transform(X)
    H = model.components_

    topics = []
    for k in range(n_topics):
        top_idx = H[k].argsort()[::-1][:n_top_words]
        top_words = feature_names[top_idx].tolist()

        top_doc_idx = W[:, k].argsort()[::-1][:20]
        candidates = neg.iloc[top_doc_idx]["review_text"].tolist()

        phrases = []
        for text in candidates:
            snippet = _short_phrase(text, top_words)
            if snippet and snippet not in phrases:
                phrases.append(snippet)
            if len(phrases) >= n_examples:
                break

        topics.append({
            "topic_id": k,
            "label": _label_topic(top_words),
            "top_words": top_words,
            "example_phrases": phrases,
        })
    return topics


# Short, human-readable one-line labels for the most common negative-review
# themes, matched by keyword overlap with each topic's top words rather than
# topic position (NMF topic order isn't stable across reruns). These are
# descriptive summaries of what the top words/phrases already show, not a
# separately-sourced claim.
_TOPIC_LABEL_RULES: list[tuple[set[str], str]] = [
    ({"boss", "bosses", "gear", "fight", "level"}, "Boss fights and gear progression feel unbalanced"),
    ({"save", "new content", "content", "update", "updates"}, "Updates sometimes force a new save to access new content"),
    ({"pvp", "pve", "server", "servers"}, "PvP/PvE server balance and solo-play friction"),
    ({"boring", "slow", "grindy", "repetitive"}, "Combat and the core loop feel slow, repetitive, or grindy"),
    ({"camera", "controller", "controls"}, "Camera and controller control issues"),
    ({"fun", "solo", "friends", "grinding", "grind"}, "Solo play feels like more of a grind than with friends"),
]


def _label_topic(top_words: list[str]) -> str:
    wordset = set(top_words)
    best_label, best_score = None, 0
    for keys, label in _TOPIC_LABEL_RULES:
        score = len(keys & wordset)
        if score > best_score:
            best_score, best_label = score, label
    return best_label if best_label else " / ".join(top_words[:3])


def _short_phrase(text: str, keywords: list[str], max_words: int = 12) -> str | None:
    """Pull a short, clean snippet (<= max_words) around the first matching
    keyword. Rejects any snippet containing profanity, even if the match
    itself is clean -- these phrases are meant to be quotable as-is."""
    words = text.replace("\n", " ").split()
    lower = [w.lower().strip(".,!?;:\"'()") for w in words]
    for kw in keywords:
        kw_first = kw.split()[0]
        for i, w in enumerate(lower):
            if w == kw_first:
                start = max(0, i - 3)
                end = min(len(words), i + max_words - 3)
                snippet_words = lower[start:end]
                if any(_has_profanity(sw) for sw in snippet_words):
                    continue
                snippet = " ".join(words[start:end]).strip()
                # Steam self-censors profanity in review text as repeated
                # "♥" (heart) characters -- treat that as profanity too.
                if "♥" in snippet:
                    continue
                if len(snippet.split()) >= 3:
                    return snippet
    return None
