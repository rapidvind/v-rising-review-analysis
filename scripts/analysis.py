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


def fit_negative_topic_model(df: pd.DataFrame, n_topics: int = 6) -> dict:
    """Fit the TF-IDF + NMF model over negative reviews only. Kept as a
    single fitting function so every consumer (the one-pager's topic list,
    and the theme-trends-by-patch analysis) reuses the exact same
    vectorizer settings, NMF hyperparameters, and random_state -- never a
    second, differently-tuned fit.

    Returns a dict: {neg, vectorizer, feature_names, model, W, H}. `neg` is
    the filtered negative-review DataFrame (short reviews below the
    minimum token count are dropped and never enter the model); `W` is the
    document-topic matrix aligned row-for-row with `neg`.
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

    return {
        "neg": neg,
        "vectorizer": vectorizer,
        "feature_names": feature_names,
        "model": model,
        "W": W,
        "H": H,
    }


def build_topics(fit: dict, n_top_words: int = 8, n_examples: int = 3) -> list[dict]:
    """Build the per-topic summary (label, top words, example phrases)
    from an already-fitted model (see fit_negative_topic_model)."""
    neg, feature_names, W, H = fit["neg"], fit["feature_names"], fit["W"], fit["H"]
    n_topics = H.shape[0]

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


def topic_model_negative_reviews(
    df: pd.DataFrame, n_topics: int = 6, n_top_words: int = 8, n_examples: int = 3
):
    """TF-IDF + NMF topic model over negative reviews only.

    Returns a list of dicts: {topic_id, top_words, example_phrases}.
    example_phrases are short (<12 word) snippets pulled from the reviews
    that load most heavily on that topic -- not full review text.
    """
    fit = fit_negative_topic_model(df, n_topics=n_topics)
    return build_topics(fit, n_top_words=n_top_words, n_examples=n_examples)


THEME_ASSIGNMENT_THRESHOLD = 0.3


def assign_themes(
    df: pd.DataFrame,
    n_topics: int = 6,
    threshold: float = THEME_ASSIGNMENT_THRESHOLD,
    normalize: bool = True,
):
    """Assign zero or more of the six named themes to every negative
    review, reusing the exact same fitted model as topic_model_negative_reviews
    (same vectorizer settings, same NMF hyperparameters and random_state) so
    these themes are identical to the ones on the one-page summary.

    By default (normalize=True), each review's six NMF weights are
    row-normalized (divided by their sum) and a theme is assigned if its
    normalized weight is >= threshold -- this is the method used for the
    headline numbers. Row-normalizing introduces a compositional ("closure")
    side effect worth knowing about: if one theme's raw weight collapses for
    a group of reviews, the other five normalized weights mechanically rise
    for those same reviews even with no real change in them, since they're
    divided by a smaller sum. Passing normalize=False assigns themes from
    the *raw* NMF weights instead (no row-sum division), which doesn't have
    that side effect and is used as a robustness check against it; raw
    weights run much smaller than normalized ones, so `threshold` needs a
    much smaller value (e.g. ~0.01-0.02) to assign a comparable share of
    reviews at least one theme.

    A review can get 0, 1, or more themes either way.

    Returns (themed, labels, n_total_negative, n_no_theme):
      themed: DataFrame, one row per *modeled* negative review (reviews too
        short to enter the vectorizer's vocabulary at all are excluded, and
        counted in n_no_theme below) -- original columns plus one boolean
        column per theme label and an integer 'n_themes' column.
      labels: the six theme label strings, in the same order as their
        boolean columns in `themed`.
      n_total_negative: total negative reviews in df (modeled or not).
      n_no_theme: negative reviews with no theme assigned, including both
        reviews too short to model and modeled reviews whose weights never
        cross the threshold for any theme.
    """
    fit = fit_negative_topic_model(df, n_topics=n_topics)
    topics = build_topics(fit, n_top_words=8, n_examples=1)
    labels = [t["label"] for t in topics]

    W = fit["W"]
    if normalize:
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W_used = W / row_sums
    else:
        W_used = W

    themed = fit["neg"].copy()
    for k, label in enumerate(labels):
        themed[label] = W_used[:, k] >= threshold
    themed["n_themes"] = themed[labels].sum(axis=1)

    n_total_negative = int((~df["recommended"]).sum())
    n_modeled = len(themed)
    n_unmodeled = n_total_negative - n_modeled
    n_modeled_no_theme = int((themed["n_themes"] == 0).sum())
    n_no_theme = n_unmodeled + n_modeled_no_theme

    return themed, labels, n_total_negative, n_no_theme


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


_sentence_re = re.compile(r"(?<=[.!?])\s+")


def _is_clean(snippet_words: list[str]) -> bool:
    return not any(_has_profanity(w) for w in snippet_words) and "♥" not in " ".join(snippet_words)


def _short_phrase(text: str, keywords: list[str], max_words: int = 16) -> str | None:
    """Pull a short, clean, representative phrase mentioning one of the
    topic's keywords. Prefers a complete sentence (reads as a finished
    thought, quotable as-is); falls back to a word-window snippet around
    the keyword if no sentence of reasonable length qualifies. Rejects
    anything containing profanity, even if the keyword match itself is
    clean.
    """
    flat = text.replace("\n", " ")

    # 1) Prefer a whole sentence containing a keyword.
    for sentence in _sentence_re.split(flat):
        sentence = sentence.strip()
        n_words = len(sentence.split())
        if not (4 <= n_words <= max_words):
            continue
        lower_words = [w.lower().strip(".,!?;:\"'()") for w in sentence.split()]
        if not _is_clean(lower_words):
            continue
        for kw in keywords:
            if kw.split()[0] in lower_words:
                return sentence

    # 2) Fall back to a word-window snippet around the first keyword match.
    words = flat.split()
    lower = [w.lower().strip(".,!?;:\"'()") for w in words]
    for kw in keywords:
        kw_first = kw.split()[0]
        for i, w in enumerate(lower):
            if w == kw_first:
                start = max(0, i - 3)
                end = min(len(words), i + max_words - 3)
                snippet_words = lower[start:end]
                if not _is_clean(snippet_words):
                    continue
                snippet = " ".join(words[start:end]).strip()
                if len(snippet.split()) >= 3:
                    return snippet
    return None


# ---------------------------------------------------------------------------
# Theme trends by patch: before/after window analysis
# ---------------------------------------------------------------------------

MIN_WINDOW_N = 30


def load_patch_events(path: str) -> pd.DataFrame:
    events = pd.read_csv(path)
    events["date"] = pd.to_datetime(events["date"], utc=True)
    return events.sort_values("date").reset_index(drop=True)


def wilson_interval(count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (nan, nan)
    for n == 0 rather than dividing by zero."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = count / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _two_proportion_diff_ci(
    count_a: int, n_a: int, count_b: int, n_b: int, z: float = 1.96
) -> tuple[float, float, float]:
    """Normal-approximation CI for p_b - p_a (two independent samples).
    Returns (diff, lo, hi)."""
    p_a, p_b = count_a / n_a, count_b / n_b
    diff = p_b - p_a
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    return diff, diff - z * se, diff + z * se


def theme_shift_by_event(
    df: pd.DataFrame,
    themed: pd.DataFrame,
    labels: list[str],
    events: pd.DataFrame,
    before_days: int = 28,
    after_days: int = 28,
    skip_days: int = 7,
    min_n: int = MIN_WINDOW_N,
    denominator: str = "negative",
) -> pd.DataFrame:
    """Before/after comparison of each theme's share of reviews, around
    each patch event.

    For each event: "before" is the `before_days` immediately preceding the
    event date; the `skip_days` immediately following the event date are
    skipped entirely (the release week itself, when reaction is freshest
    and most mixed with launch-hype/sale noise); "after" is the
    `after_days` starting once that skip period ends.

    `denominator` controls what each theme's share is a fraction of:
      "negative" (default): all negative reviews in the window, including
        ones too short to have entered the topic model (they structurally
        can't be said to "mention" a theme, so they count against every
        theme). This is the headline metric: "share of negative reviews".
      "all": every review in the window, positive or negative. Since a
        theme can only ever come from a negative review, this reframes the
        same counts as "themed reviews per 100 total reviews" -- useful as
        a check against the "negative" denominator shifting for reasons
        unrelated to any theme (e.g. a change in overall % positive).

    A window with fewer than `min_n` reviews in that denominator is skipped
    entirely for that event (both themes and the event get marked
    skipped=True) -- there isn't enough data to say anything. A two-sided
    p-value for the before/after difference (normal approximation) is
    included in every non-skipped row so callers can apply their own
    multiple-testing correction (see benjamini_hochberg below).
    """
    if denominator == "negative":
        base_df = df[~df["recommended"]]
    elif denominator == "all":
        base_df = df
    else:
        raise ValueError(f"denominator must be 'negative' or 'all', got {denominator!r}")
    rows = []

    for _, ev in events.iterrows():
        event_date = ev["date"]
        before_start = event_date - pd.Timedelta(days=before_days)
        before_end = event_date
        after_start = event_date + pd.Timedelta(days=skip_days)
        after_end = after_start + pd.Timedelta(days=after_days)

        before_mask = (base_df["created_at"] >= before_start) & (base_df["created_at"] < before_end)
        after_mask = (base_df["created_at"] >= after_start) & (base_df["created_at"] < after_end)
        n_before = int(before_mask.sum())
        n_after = int(after_mask.sum())
        skipped = (n_before < min_n) or (n_after < min_n)

        before_idx = base_df.index[before_mask]
        after_idx = base_df.index[after_mask]

        for label in labels:
            if skipped:
                rows.append({
                    "event": ev["name"], "event_date": event_date, "event_type": ev["type"],
                    "theme": label, "n_before": n_before, "n_after": n_after,
                    "skipped": True, "skip_reason": f"fewer than {min_n} reviews in before and/or after window",
                    "count_before": np.nan, "count_after": np.nan,
                    "share_before": np.nan, "share_after": np.nan,
                    "before_lo": np.nan, "before_hi": np.nan,
                    "after_lo": np.nan, "after_hi": np.nan,
                    "diff": np.nan, "diff_lo": np.nan, "diff_hi": np.nan,
                    "p_value": np.nan, "significant": False,
                })
                continue

            count_before = int(themed[label].reindex(before_idx, fill_value=False).sum())
            count_after = int(themed[label].reindex(after_idx, fill_value=False).sum())
            share_before = count_before / n_before
            share_after = count_after / n_after
            before_lo, before_hi = wilson_interval(count_before, n_before)
            after_lo, after_hi = wilson_interval(count_after, n_after)
            diff, diff_lo, diff_hi = _two_proportion_diff_ci(count_before, n_before, count_after, n_after)
            significant = (diff_lo > 0) or (diff_hi < 0)
            p_value = _two_proportion_p_value(count_before, n_before, count_after, n_after)

            rows.append({
                "event": ev["name"], "event_date": event_date, "event_type": ev["type"],
                "theme": label, "n_before": n_before, "n_after": n_after,
                "skipped": False, "skip_reason": "",
                "count_before": count_before, "count_after": count_after,
                "share_before": share_before, "share_after": share_after,
                "before_lo": before_lo, "before_hi": before_hi,
                "after_lo": after_lo, "after_hi": after_hi,
                "diff": diff, "diff_lo": diff_lo, "diff_hi": diff_hi,
                "p_value": p_value, "significant": significant,
            })

    return pd.DataFrame(rows)


def _two_proportion_p_value(count_a: int, n_a: int, count_b: int, n_b: int) -> float:
    """Two-sided p-value for a two-proportion z-test (pooled SE under the
    null of no difference), used only for the multiple-testing correction
    below -- the significance call itself is based on the per-comparison
    CI in theme_shift_by_event, not this p-value."""
    p_a, p_b = count_a / n_a, count_b / n_b
    p_pool = (count_a + count_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = (p_b - p_a) / se
    from scipy.stats import norm
    return float(2 * (1 - norm.cdf(abs(z))))


def benjamini_hochberg(p_values, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns a boolean array, same
    order and length as `p_values`, marking which survive at the given
    false-discovery rate."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    passed = ranked <= thresholds
    cutoff = ranked[np.max(np.where(passed)[0])] if passed.any() else -1.0
    return p <= cutoff


def add_bh_correction(shift: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add a 'bh_significant' column to a theme_shift_by_event result,
    applying Benjamini-Hochberg correction across all non-skipped rows
    (i.e. across every event/theme comparison actually tested)."""
    shift = shift.copy()
    shift["bh_significant"] = False
    valid = ~shift["skipped"]
    if valid.any():
        shift.loc[valid, "bh_significant"] = benjamini_hochberg(shift.loc[valid, "p_value"], alpha=alpha)
    return shift


def weekly_theme_share(df: pd.DataFrame, themed: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Weekly share of negative reviews assigned to each theme, plus a
    4-week rolling version, masked the same way as weekly_volume_and_positivity:
    the trailing partial week is dropped, and weeks below MIN_REVIEWS_PER_WEEK
    negative reviews have their raw (non-rolling) share masked."""
    neg_all = df[~df["recommended"]].copy()
    for label in labels:
        neg_all[label] = themed[label].reindex(neg_all.index, fill_value=False)

    weekly = neg_all.set_index("created_at").resample("W").agg(
        num_negative=("recommended", "size"),
        **{f"count_{label}": (label, "sum") for label in labels},
    )
    weekly = weekly.reset_index()

    if len(weekly) and weekly["created_at"].iloc[-1] > neg_all["created_at"].max():
        weekly = weekly.iloc[:-1].copy()

    rolling_n_negative = weekly["num_negative"].rolling(4, min_periods=1).sum()
    for label in labels:
        raw_share = weekly[f"count_{label}"] / weekly["num_negative"] * 100
        weekly[f"share_{label}"] = raw_share.where(weekly["num_negative"] >= MIN_REVIEWS_PER_WEEK)
        rolling_share = (
            weekly[f"count_{label}"].rolling(4, min_periods=1).sum()
            / rolling_n_negative
            * 100
        )
        # Masked the same way as the main weekly chart: a rolling point is
        # only shown once its own trailing window has enough negative
        # reviews behind it (in practice the 4-week sum rarely dips below
        # this, but early/sparse weeks are protected against showing an
        # apparently-solid line built from a handful of reviews).
        weekly[f"share_rolling4_{label}"] = rolling_share.where(
            rolling_n_negative >= MIN_REVIEWS_PER_WEEK
        )
    return weekly
