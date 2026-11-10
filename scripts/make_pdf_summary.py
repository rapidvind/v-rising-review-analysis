"""
Build the one-page PDF summary: two charts + three findings + a closing
line on what internal telemetry could add. Run after fetch_reviews.py.

Usage:
    python make_pdf_summary.py --data ../data/reviews_raw.jsonl --out ../v_rising_review_analysis_summary.pdf
"""
import argparse
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from analysis import (
    MAJOR_UPDATES,
    load_reviews,
    playtime_by_recommendation,
    weekly_volume_and_positivity,
)

PALETTE = {
    "positive": "#2E7D5B",
    "negative": "#B3493B",
    "neutral": "#6B6E76",
    "accent": "#3D5A80",
}


def build(data_path: str, out_path: str, images_dir: str):
    df = load_reviews(data_path)
    weekly = weekly_volume_and_positivity(df)
    playtime = playtime_by_recommendation(df)

    n_total = len(df)
    pct_positive_overall = df["recommended"].mean() * 100
    median_pt_pos = playtime.loc["positive", "median"]
    median_pt_neg = playtime.loc["negative", "median"]
    date_min = df["created_at"].min().date()
    date_max = df["created_at"].max().date()

    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.55, 1.5, 1.3, 1.1], hspace=0.55)

    # Header
    ax_header = fig.add_subplot(gs[0])
    ax_header.axis("off")
    ax_header.text(
        0, 1.0, "V Rising — Public Steam Review Analysis", fontsize=18, weight="bold",
        va="top",
    )
    ax_header.text(
        0, 0.55,
        f"English-language Steam reviews, {date_min} to {date_max}  •  n = {n_total:,}  "
        f"•  {pct_positive_overall:.0f}% positive overall",
        fontsize=10.5, color="#444444", va="top",
    )
    ax_header.text(
        0, 0.15,
        "Independent fan analysis using Steam's public review data. Not affiliated with "
        "or endorsed by Stunlock Studios.",
        fontsize=8.5, color="#777777", style="italic", va="top",
    )

    # Chart 1: weekly volume + % positive with update markers
    ax1 = fig.add_subplot(gs[1])
    ax1b = ax1.twinx()
    ax1.bar(weekly["created_at"], weekly["num_reviews"], width=5.5,
            color=PALETTE["neutral"], alpha=0.35, label="Reviews / week")
    ax1b.plot(weekly["created_at"], weekly["pct_positive"], color=PALETTE["accent"],
               linewidth=1.8, label="% positive")
    for label, date_str in MAJOR_UPDATES:
        d = mdates.datestr2num(date_str)
        ax1.axvline(d, color=PALETTE["negative"], linestyle="--", linewidth=1, alpha=0.8)
        ax1.text(d, ax1.get_ylim()[1] * 0.96, label, rotation=90, fontsize=7.5,
                  color=PALETTE["negative"], va="top", ha="right")
    ax1.set_ylabel("Reviews / week", fontsize=9)
    ax1b.set_ylabel("% positive", fontsize=9)
    ax1b.set_ylim(0, 100)
    ax1.set_title("Review volume and % positive over time", fontsize=11, weight="bold", loc="left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.tick_params(axis="x", labelsize=8, rotation=30)
    ax1.tick_params(axis="y", labelsize=8)
    ax1b.tick_params(axis="y", labelsize=8)
    fig.savefig(f"{images_dir}/weekly_volume_positivity.png", dpi=160, bbox_inches="tight")

    # Chart 2: playtime distribution by recommendation
    ax2 = fig.add_subplot(gs[2])
    pos_hours = df.loc[df["recommended"], "playtime_at_review_hours"].clip(upper=400)
    neg_hours = df.loc[~df["recommended"], "playtime_at_review_hours"].clip(upper=400)
    bp = ax2.boxplot(
        [pos_hours, neg_hours], vert=False, tick_labels=["Positive", "Negative"],
        patch_artist=True, widths=0.5, showfliers=False,
    )
    for patch, color in zip(bp["boxes"], [PALETTE["positive"], PALETTE["negative"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.set_xlabel("Playtime at time of review (hours, clipped at 400 for readability)", fontsize=9)
    ax2.set_title("Playtime at review: positive vs. negative", fontsize=11, weight="bold", loc="left")
    ax2.tick_params(labelsize=9)
    fig.savefig(f"{images_dir}/playtime_by_recommendation.png", dpi=160, bbox_inches="tight")

    # Findings + limitations text block
    ax_text = fig.add_subplot(gs[3])
    ax_text.axis("off")

    findings = [
        f"1. Positive reviews have a higher median playtime at review ({median_pt_pos:.0f}h) "
        f"than negative ones ({median_pt_neg:.0f}h) — people who bounce off the game tend to "
        f"say so earlier, not after a long investment.",
        "2. % positive moves visibly around major updates (see dashed lines above); volume "
        "spikes are a clearer signal of update impact than sentiment alone, which stays "
        "fairly stable.",
        "3. Negative reviews cluster around a small number of recurring, nameable gripes "
        "rather than being diffuse — see the notebook's topic model for the specific themes "
        "and representative phrasing.",
    ]
    y = 1.0
    ax_text.text(0, y, "Findings", fontsize=11, weight="bold", va="top")
    y -= 0.12
    for f in findings:
        for line in textwrap.wrap(f, width=100):
            ax_text.text(0, y, line, fontsize=8.7, va="top")
            y -= 0.075
        y -= 0.02

    y -= 0.03
    ax_text.text(0, y, "Limitations", fontsize=10, weight="bold", va="top", color="#555555")
    y -= 0.11
    limitations = (
        "Steam reviewers are a self-selected, vocal subset of players, not a representative "
        "sample; this dataset is English-language reviews only; review text reflects stated "
        "opinion, not behavioral telemetry (e.g. actual churn points, session structure)."
    )
    for line in textwrap.wrap(limitations, width=110):
        ax_text.text(0, y, line, fontsize=8, va="top", color="#666666", style="italic")
        y -= 0.075

    y -= 0.04
    ax_text.text(
        0, y,
        "What internal telemetry could add: exact drop-off points, session length and return "
        "cadence by patch, and silent churn from players who never write a review at all.",
        fontsize=8.7, va="top", weight="bold", color=PALETTE["accent"],
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/reviews_raw.jsonl")
    ap.add_argument("--out", default="../v_rising_review_analysis_summary.pdf")
    ap.add_argument("--images-dir", default="../images")
    args = ap.parse_args()
    build(args.data, args.out, args.images_dir)
