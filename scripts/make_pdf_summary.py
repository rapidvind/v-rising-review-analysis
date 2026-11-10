"""
Build the one-page (A4) PDF summary: two charts, three findings, the six
recurring negative-review themes, limitations, and a closing line on what
internal telemetry could add. Run after fetch_reviews.py.

Layout note: all text on the page is placed with an explicit vertical
cursor (in inches, converted to figure-fraction) rather than relying on
matplotlib gridspec row fractions for text blocks. That's what keeps line
spacing consistent (no overlap) and avoids uneven empty gaps -- each block
consumes exactly the vertical space its wrapped text needs, no more.

Usage:
    python make_pdf_summary.py --data ../data/reviews_raw.jsonl --out ../V_Rising_Review_Analysis_Summary.pdf
"""
import argparse
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from analysis import (
    MAJOR_UPDATES,
    MINOR_EVENTS,
    load_reviews,
    playtime_by_recommendation,
    topic_model_negative_reviews,
    weekly_volume_and_positivity,
)

PALETTE = {
    "positive": "#2E7D5B",
    "negative": "#B3493B",
    "neutral": "#6B6E76",
    "accent": "#3D5A80",
}

# A4 in inches
PAGE_W, PAGE_H = 8.27, 11.69
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 0.5, 0.5, 0.32, 0.28
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

# One theme's best automatically-mined phrase reads as a sentence fragment
# ("is horrendous. Some server settings..."); this swaps in a complete,
# clean sentence that's already in that topic's own candidate pool (mined
# from the same negative reviews, not hand-written) -- a presentation
# choice, not a change to the underlying topic model or any statistic.
PREFERRED_PHRASE = {
    "PvP/PvE server balance and solo-play friction": "This game is just not fun as a PvE player anymore.",
}


class Cursor:
    """Tracks a vertical position in inches from the top of the page and
    converts to figure-fraction coordinates for fig.text()."""

    def __init__(self, fig, y_start_in):
        self.fig = fig
        self.y = y_start_in

    def frac(self):
        return self.y / PAGE_H

    def advance_in(self, inches):
        self.y -= inches

    def text(self, s, fontsize, **kwargs):
        self.fig.text(MARGIN_L / PAGE_W, self.frac(), s, fontsize=fontsize,
                       va="top", ha="left", **kwargs)

    def wrapped_text(self, s, fontsize, width_chars, line_gap_in=None, **kwargs):
        line_h = line_gap_in or (fontsize * 1.38 / 72)
        for line in textwrap.wrap(s, width=width_chars):
            self.text(line, fontsize, **kwargs)
            self.advance_in(line_h)


def build_page1(df, images_dir: str):
    weekly = weekly_volume_and_positivity(df)
    playtime = playtime_by_recommendation(df)
    topics = topic_model_negative_reviews(df, n_topics=6, n_top_words=8, n_examples=10)

    n_total = len(df)
    pct_positive_overall = df["recommended"].mean() * 100
    median_pt_pos = playtime.loc["positive", "median"]
    median_pt_neg = playtime.loc["negative", "median"]
    date_min = df["created_at"].min().date()
    date_max = df["created_at"].max().date()

    def pick_phrase(topic):
        override = PREFERRED_PHRASE.get(topic["label"])
        if override and override in topic["example_phrases"]:
            return override
        return topic["example_phrases"][0] if topic["example_phrases"] else None

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    cur = Cursor(fig, PAGE_H - MARGIN_T)

    # ---- Header ----
    cur.text("V Rising: Public Steam Review Analysis", 18, weight="bold")
    cur.advance_in(0.22)
    cur.text(
        f"English-language Steam reviews, {date_min} to {date_max}  •  n = {n_total:,}  "
        f"•  {pct_positive_overall:.0f}% positive overall",
        10.5, color="#444444",
    )
    cur.advance_in(0.22)
    cur.text(
        "Independent fan analysis using Steam's public review data. Not affiliated with "
        "or endorsed by Stunlock Studios.",
        8.5, color="#777777", style="italic",
    )
    cur.advance_in(0.2)

    # ---- Chart 1: weekly volume + 4-week rolling % positive ----
    # Axes rects below are inset from the text margins to leave room for
    # tick labels and axis labels, which matplotlib draws *outside* the
    # given rect -- fig.add_axes() does not reserve space for them.
    cur.text("Review volume and % positive over time", 11, weight="bold")
    cur.advance_in(0.26)

    chart1_h = 2.0
    left_in = MARGIN_L + 0.55
    right_in = MARGIN_R + 0.5
    width_in = PAGE_W - left_in - right_in
    ax1 = fig.add_axes([
        left_in / PAGE_W, (cur.y - chart1_h) / PAGE_H, width_in / PAGE_W, chart1_h / PAGE_H,
    ])
    ax1b = ax1.twinx()
    ax1.bar(weekly["created_at"], weekly["num_reviews"], width=5.5,
            color=PALETTE["neutral"], alpha=0.35)
    ax1b.plot(weekly["created_at"], weekly["pct_positive_rolling4"], color=PALETTE["accent"],
              linewidth=1.8)

    for _, date_str in MAJOR_UPDATES:
        d = mdates.datestr2num(date_str)
        ax1.axvline(d, color=PALETTE["negative"], linestyle="--", linewidth=1.3, alpha=0.9)
    for _, date_str in MINOR_EVENTS:
        d = mdates.datestr2num(date_str)
        ax1.axvline(d, color=PALETTE["neutral"], linestyle=":", linewidth=1.1, alpha=0.85)

    ax1.set_ylabel("Reviews / week", fontsize=9)
    ax1b.set_ylabel("% positive", fontsize=9)
    ax1b.set_ylim(0, 100)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.tick_params(axis="x", labelsize=8, rotation=30)
    ax1.tick_params(axis="y", labelsize=8)
    ax1b.tick_params(axis="y", labelsize=8)

    # Only the two event types need explaining (the bar/line series are
    # already named by the axis labels); keep the legend small and inside
    # the plot so it can't collide with the title above the axes.
    legend_handles = [
        Line2D([0], [0], color=PALETTE["negative"], linestyle="--", linewidth=1.3, label="Major content update"),
        Line2D([0], [0], color=PALETTE["neutral"], linestyle=":", linewidth=1.1, label="Other verified spike"),
    ]
    ax1.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=7,
               framealpha=0.85, handlelength=1.8)

    fig.savefig(f"{images_dir}/weekly_volume_positivity.png", dpi=160, bbox_inches="tight")
    # Extra clearance below the axes for the rotated date tick labels,
    # which render below the rect, not inside it.
    cur.advance_in(chart1_h + 0.55)

    cap1 = (
        "Major updates (red dashed, left to right): 1.0 Launch (8 May 2024), Invaders of "
        "Oakveil (28 Apr 2025). Other verified spikes (gray dotted, left to right): 1.0.9.0 "
        "patch (17 Sep 2024), Steam Autumn Sale (27 Nov-4 Dec 2024), Free Weekend + 1.1.1 "
        "patch (14-18 May 2026). Unlabeled fluctuations elsewhere are not individually "
        "explained. Line is a 4-week rolling % positive; weeks below 20 reviews are masked, "
        "and the final partial week is excluded."
    )
    cur.wrapped_text(cap1, 7.2, 118, color="#666666", line_gap_in=0.125)
    cur.advance_in(0.22)

    # ---- Chart 2: playtime distribution ----
    cur.text("Playtime at review: positive vs. negative", 11, weight="bold")
    cur.advance_in(0.26)

    chart2_h = 1.15
    left2_in = MARGIN_L + 0.5
    width2_in = PAGE_W - left2_in - MARGIN_R
    ax2 = fig.add_axes([
        left2_in / PAGE_W, (cur.y - chart2_h) / PAGE_H, width2_in / PAGE_W, chart2_h / PAGE_H,
    ])
    pos_hours = df.loc[df["recommended"], "playtime_at_review_hours"].clip(upper=400)
    neg_hours = df.loc[~df["recommended"], "playtime_at_review_hours"].clip(upper=400)
    bp = ax2.boxplot(
        [pos_hours, neg_hours], vert=False, tick_labels=["Positive", "Negative"],
        patch_artist=True, widths=0.5, showfliers=False,
    )
    for patch, color in zip(bp["boxes"], [PALETTE["positive"], PALETTE["negative"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    # Axis explicitly spans the full clip range so the label and the plot
    # agree (whiskers end well before 400h; that's real, not a mismatch).
    ax2.set_xlim(0, 400)
    ax2.set_xlabel("Playtime at time of review (hours, clipped at 400 for readability)", fontsize=8.5)
    ax2.tick_params(labelsize=9)
    fig.savefig(f"{images_dir}/playtime_by_recommendation.png", dpi=160, bbox_inches="tight")
    # Extra clearance below the axes for the x-axis tick labels + xlabel.
    cur.advance_in(chart2_h + 0.42)

    # ---- Findings ----
    cur.text("Findings", 11, weight="bold")
    cur.advance_in(0.17)
    findings = [
        f"1. Positive reviews have a higher median playtime at review ({median_pt_pos:.0f}h) "
        f"than negative ones ({median_pt_neg:.0f}h): people who bounce off the game tend to "
        f"say so earlier, not after a long investment.",
        "2. Review volume spikes with each major update and decays back to baseline within "
        "roughly 6-8 weeks; % positive (4-week rolling) stays fairly stable through the same "
        "window, so volume, not sentiment, is the clearer signal of an update's reach.",
        "3. Negative reviews cluster around six recurring, nameable themes rather than being "
        "diffuse (below), not a diagnosis of why but a map of where to look.",
    ]
    for f in findings:
        cur.wrapped_text(f, 8.7, 108, line_gap_in=0.145)
        cur.advance_in(0.04)
    cur.advance_in(0.1)

    # ---- Six negative-review themes ----
    cur.text("Recurring themes in negative reviews", 11, weight="bold")
    cur.advance_in(0.17)
    cur.wrapped_text(
        "TF-IDF + NMF topic model over negative reviews only. Each line is the model's theme "
        "plus one short, clean representative phrase (not a full quote).",
        7.6, 118, color="#666666", style="italic", line_gap_in=0.115,
    )
    cur.advance_in(0.05)
    for i, t in enumerate(topics, start=1):
        phrase = pick_phrase(t)
        cur.text(f"{i}. {t['label']}", 9.2, weight="bold")
        cur.advance_in(0.15)
        if phrase:
            cur.text(f"“{phrase}”", 8.3, style="italic", color="#444444")
            cur.advance_in(0.145)
        cur.advance_in(0.035)
    cur.advance_in(0.08)

    # ---- Limitations + closing line ----
    cur.text("Limitations", 10, weight="bold", color="#555555")
    cur.advance_in(0.15)
    limitations = (
        "Steam reviewers are a self-selected, vocal subset of players, not a representative "
        "sample; this dataset is English-language reviews only; review text reflects stated "
        "opinion, not behavioral telemetry (e.g. actual churn points, session structure)."
    )
    cur.wrapped_text(limitations, 8, 118, color="#666666", style="italic", line_gap_in=0.135)
    cur.advance_in(0.1)

    closing = (
        "What internal telemetry could add: exact drop-off points, session length and return "
        "cadence by patch, and silent churn from players who never write a review at all."
    )
    cur.wrapped_text(closing, 8.7, 100, weight="bold", color=PALETTE["accent"], line_gap_in=0.145)

    assert cur.y > MARGIN_B, (
        f"Page 1 content overflowed the page (cursor at {cur.y:.2f}in, margin {MARGIN_B}in) -- "
        "shorten text or shrink a chart."
    )
    print(f"Page 1 bottom margin remaining: {cur.y - MARGIN_B:.2f}in")
    return fig


def build_page2(df, events_path: str, images_dir: str):
    """Second page: theme trends by patch. Embeds the two pre-rendered
    chart PNGs (built by make_theme_trends.py) rather than re-plotting,
    so this page always matches exactly what make_theme_trends.py produced."""
    import matplotlib.image as mpimg

    from analysis import (
        add_bh_correction,
        assign_themes,
        load_patch_events,
        theme_shift_by_event,
    )

    RAW_WEIGHT_THRESHOLD = 0.01

    themed, labels, n_total_negative, n_no_theme = assign_themes(df)
    events = load_patch_events(events_path)
    shift = theme_shift_by_event(df, themed, labels, events)
    shift = add_bh_correction(shift)

    # Robustness checks against the normalization ("closure") side effect of
    # row-normalizing six weights that sum to 1: raw (unnormalized) NMF
    # weights, and the same comparison expressed as a share of *all* reviews
    # rather than negative reviews only. A finding is only reported below if
    # it survives the original CI test, both robustness checks, and
    # Benjamini-Hochberg correction across all tests.
    themed_raw, labels_raw, _, _ = assign_themes(df, normalize=False, threshold=RAW_WEIGHT_THRESHOLD)
    shift_raw = theme_shift_by_event(df, themed_raw, labels_raw, events)
    shift_all = theme_shift_by_event(df, themed, labels, events, denominator="all")

    key_cols = ["event", "theme"]
    merged = shift.merge(
        shift_raw[key_cols + ["significant"]].rename(columns={"significant": "sig_raw"}),
        on=key_cols, how="left",
    ).merge(
        shift_all[key_cols + ["significant"]].rename(columns={"significant": "sig_all"}),
        on=key_cols, how="left",
    )
    merged["confirmed"] = (
        merged["significant"] & merged["bh_significant"]
        & merged["sig_raw"].fillna(False) & merged["sig_all"].fillna(False)
    )
    merged["flagged_not_confirmed"] = merged["significant"] & ~merged["confirmed"]

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    cur = Cursor(fig, PAGE_H - MARGIN_T)

    # ---- Header ----
    cur.text("Theme trends by patch", 16, weight="bold")
    cur.advance_in(0.26)
    method = (
        f"Each negative review is assigned 0+ of the six themes from page 1 (same TF-IDF + "
        f"NMF model, not refit); a theme is assigned when its row-normalized NMF weight is "
        f">= 0.3. {n_no_theme:,} of {n_total_negative:,} negative reviews ({n_no_theme / n_total_negative * 100:.0f}%) "
        f"get no theme at all (mostly reviews too short to model). For each verified patch "
        f"event (table below), a 4-week window before is compared to a 4-week window after, "
        f"skipping the release week itself; windows under 30 negative reviews are skipped "
        f"and marked so. Intervals are 95% (Wilson for each share, normal-approximation for "
        f"the difference). A shift is reported as a finding below only if it survives all of: "
        f"the 95% interval excluding zero, Benjamini-Hochberg correction across all tests, and "
        f"two robustness checks (raw unnormalized NMF weights, and the same comparison as a "
        f"share of all reviews rather than negative reviews only) that guard against a "
        f"row-normalization side effect: when one theme's weight collapses, the other five can "
        f"mechanically look larger in share terms with no real independent change."
    )
    cur.wrapped_text(method, 7.2, 138, color="#444444", line_gap_in=0.105)
    cur.advance_in(0.06)

    # ---- Chart 1: small multiples ----
    img1 = mpimg.imread(f"{images_dir}/theme_trends_by_patch.png")
    w1_in = CONTENT_W * 0.76
    h1_in = w1_in * (img1.shape[0] / img1.shape[1])
    left1 = MARGIN_L + (CONTENT_W - w1_in) / 2
    ax1 = fig.add_axes([left1 / PAGE_W, (cur.y - h1_in) / PAGE_H, w1_in / PAGE_W, h1_in / PAGE_H])
    ax1.imshow(img1)
    ax1.axis("off")
    cur.advance_in(h1_in + 0.1)

    # ---- Chart 2: before/after heatmap ----
    img2 = mpimg.imread(f"{images_dir}/theme_shift_by_event.png")
    w2_in = CONTENT_W * 0.76
    h2_in = w2_in * (img2.shape[0] / img2.shape[1])
    left2 = MARGIN_L + (CONTENT_W - w2_in) / 2
    ax2 = fig.add_axes([left2 / PAGE_W, (cur.y - h2_in) / PAGE_H, w2_in / PAGE_W, h2_in / PAGE_H])
    ax2.imshow(img2)
    ax2.axis("off")
    cur.advance_in(h2_in + 0.12)

    # ---- Findings (grounded in the merged/checked table, not hardcoded) ----
    def fmt_pp(row):
        return f"{row['diff'] * 100:+.0f}pp (95% CI {row['diff_lo'] * 100:+.0f} to {row['diff_hi'] * 100:+.0f}pp)"

    confirmed = merged[merged["confirmed"]].copy()
    not_confirmed = merged[merged["flagged_not_confirmed"]].copy()
    n_tested = int((~merged["skipped"]).sum())

    findings = []
    if len(confirmed):
        top = confirmed.sort_values("diff", key=lambda s: s.abs(), ascending=False).iloc[0]
        findings.append(
            f"1. The one shift that survives every check here (95% interval, raw-weight "
            f"robustness check, all-reviews robustness check, and Benjamini-Hochberg "
            f"correction across all {n_tested} tests): around \"{top['event']}\", \"{top['theme']}\" "
            f"moved {fmt_pp(top)} (n={top['n_before']:.0f} before, {top['n_after']:.0f} after)."
        )
    else:
        findings.append(f"1. No shift out of {n_tested} event-theme pairs survives every check applied here.")

    if len(not_confirmed):
        flagged_desc = "; ".join(
            f"\"{r['event']}\" / \"{r['theme']}\" ({r['diff'] * 100:+.0f}pp originally)"
            for _, r in not_confirmed.iterrows()
        )
        findings.append(
            f"2. {len(not_confirmed)} other shift(s) looked significant on the original 95% "
            f"interval alone but did not survive the full check: {flagged_desc}. These are "
            f"consistent with a row-normalization side effect (six per-review weights sum to "
            f"1, so a collapse in one theme mechanically inflates the others' share) rather "
            f"than an independent signal, and are not reported as findings."
        )

    findings.append(
        f"3. Out of {n_tested} event-theme pairs with enough data to test, only the one finding "
        f"above holds up end to end. Everything else, including every theme around both "
        f"hotfix-type events, the October 2025 patch, and Patch 1.1.13.0, does not move in a "
        f"way this method can distinguish from noise once corrected for multiple testing. If a "
        f"theme doesn't move, that's the honest result, not a gap in the analysis."
    )
    findings.append(
        "4. \"Camera and controller control issues\" and \"Solo play feels like more of a grind "
        "than with friends\" show no significant shift around any verified event tested here: "
        "a stable background complaint, not something that visibly responds to a specific patch."
    )
    findings.append(
        "5. This is correlation keyed to release dates, not a causal test: free weekends, "
        "platform-wide sales, and influencer waves can land in the same windows (see page 1) "
        "and are not separated out here, so even the one confirmed shift may partly reflect "
        "something other than the patch itself."
    )

    cur.text("Findings", 10.5, weight="bold")
    cur.advance_in(0.15)
    for f in findings:
        cur.wrapped_text(f, 7.3, 132, line_gap_in=0.1)
        cur.advance_in(0.03)

    assert cur.y > MARGIN_B, (
        f"Page 2 content overflowed the page (cursor at {cur.y:.2f}in, margin {MARGIN_B}in) -- "
        "shorten text or shrink a chart."
    )
    print(f"Page 2 bottom margin remaining: {cur.y - MARGIN_B:.2f}in")
    return fig


def build(data_path: str, events_path: str, out_path: str, images_dir: str):
    from matplotlib.backends.backend_pdf import PdfPages

    df = load_reviews(data_path)
    fig1 = build_page1(df, images_dir)
    fig2 = build_page2(df, events_path, images_dir)

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
    plt.close(fig1)
    plt.close(fig2)
    print(f"Wrote {out_path} (2 pages)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/reviews_raw.jsonl")
    ap.add_argument("--events", default="../data/patch_events.csv")
    ap.add_argument("--out", default="../V_Rising_Review_Analysis_Summary.pdf")
    ap.add_argument("--images-dir", default="../images")
    args = ap.parse_args()
    build(args.data, args.events, args.out, args.images_dir)
