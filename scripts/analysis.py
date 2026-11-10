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

STOPWORDS_EXTRA = {
    "game", "games", "rising", "v", "vrising", "play", "played", "playing",
    "player", "players", "just", "really", "like", "also", "even", "would",
    "get", "got", "still", "much", "many", "lot", "one", "time", "dont",
    "didnt", "im", "ive", "youre", "thats", "it", "https", "http", "com",
}


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
    weekly = (
        df.set_index("created_at")
        .resample("W")
        .agg(num_reviews=("recommended", "size"), pct_positive=("recommended", "mean"))
    )
    weekly["pct_positive"] = weekly["pct_positive"] * 100
    return weekly.reset_index()


def playtime_by_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("recommended")["playtime_at_review_hours"]
        .agg(["count", "median", "mean", "std"])
        .rename(index={True: "positive", False: "negative"})
    )


_token_re = re.compile(r"[a-zA-Z]{3,}")


def _tokenize(text: str) -> str:
    tokens = [t.lower() for t in _token_re.findall(text)]
    tokens = [t for t in tokens if t not in STOPWORDS_EXTRA]
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

        topics.append({"topic_id": k, "top_words": top_words, "example_phrases": phrases})
    return topics


def _short_phrase(text: str, keywords: list[str], max_words: int = 12) -> str | None:
    """Pull a short snippet (<= max_words) around the first matching keyword."""
    words = text.replace("\n", " ").split()
    lower = [w.lower().strip(".,!?;:\"'()") for w in words]
    for kw in keywords:
        kw_first = kw.split()[0]
        for i, w in enumerate(lower):
            if w == kw_first:
                start = max(0, i - 3)
                end = min(len(words), i + max_words - 3)
                snippet = " ".join(words[start:end]).strip()
                if len(snippet.split()) >= 3:
                    return snippet
    return None
