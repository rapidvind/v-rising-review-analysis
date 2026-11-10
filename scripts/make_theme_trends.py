"""
Theme trends by patch: builds the two supporting charts (small multiples
of each theme's share of negative reviews over time, and a before/after
heatmap per event) plus the numeric before/after table used in the README,
notebook, and PDF page 2.

Usage:
    python make_theme_trends.py --data ../data/reviews_raw.jsonl \
        --events ../data/patch_events.csv --images-dir ../images
"""
import argparse
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.lines import Line2D

from analysis import (
    MAJOR_UPDATES,
    add_bh_correction,
    assign_themes,
    load_patch_events,
    load_reviews,
    theme_shift_by_event,
    weekly_theme_share,
)

# Raw-weight threshold calibrated to assign roughly the same overall share
# of modeled reviews at least one theme as the normalized threshold does
# (see README/notebook for how this was chosen).
RAW_WEIGHT_THRESHOLD = 0.01

EVENT_STYLE = {
    "major": {"color": "#B3493B", "linestyle": "--", "linewidth": 1.3, "label": "Major update"},
    "patch": {"color": "#3D5A80", "linestyle": ":", "linewidth": 1.1, "label": "Patch"},
    "hotfix": {"color": "#6B6E76", "linestyle": ":", "linewidth": 0.9, "label": "Hotfix"},
}


def plot_theme_trends(wts, labels, events, out_path):
    n = len(labels)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, label in zip(axes, labels):
        ax.plot(wts["created_at"], wts[f"share_rolling4_{label}"], color="#3D5A80", linewidth=1.5)
        for _, ev in events.iterrows():
            style = EVENT_STYLE.get(ev["type"], EVENT_STYLE["patch"])
            ax.axvline(ev["date"], color=style["color"], linestyle=style["linestyle"],
                       linewidth=style["linewidth"], alpha=0.85)
        ax.set_title(label, fontsize=9.5, loc="left", wrap=True)
        ax.set_ylabel("% of neg.\nreviews", fontsize=7.5)
        ax.tick_params(labelsize=7.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_ylim(bottom=0)

    for ax in axes[n:]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color=s["color"], linestyle=s["linestyle"], linewidth=s["linewidth"] + 0.5, label=s["label"])
        for s in EVENT_STYLE.values()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "4-week rolling share of negative reviews mentioning each theme",
        fontsize=11, weight="bold", y=1.06,
    )
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_shift_heatmap(shift, labels, out_path):
    events_order = shift["event"].drop_duplicates().tolist()
    n_events, n_themes = len(events_order), len(labels)

    grid = np.full((n_themes, n_events), np.nan)
    sig = np.zeros((n_themes, n_events), dtype=bool)
    bh_sig = np.zeros((n_themes, n_events), dtype=bool)
    skip = np.zeros((n_themes, n_events), dtype=bool)

    for _, row in shift.iterrows():
        j = events_order.index(row["event"])
        i = labels.index(row["theme"])
        if row["skipped"]:
            skip[i, j] = True
            continue
        grid[i, j] = row["diff"] * 100
        sig[i, j] = row["significant"]
        bh_sig[i, j] = row.get("bh_significant", False)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(n_themes):
        for j in range(n_events):
            if skip[i, j]:
                ax.text(j, i, "skipped", ha="center", va="center", fontsize=7, color="#888888", style="italic")
                continue
            val = grid[i, j]
            weight = "bold" if bh_sig[i, j] else "normal"
            marker = "**" if bh_sig[i, j] else ("*" if sig[i, j] else "")
            ax.text(j, i, f"{val:+.0f}pp{marker}", ha="center", va="center", fontsize=7.5,
                    weight=weight, color="black" if abs(val) < vmax * 0.6 else "white")

    ax.set_xticks(range(n_events))
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(e, 16)) for e in events_order], fontsize=7, rotation=0,
    )
    ax.set_yticks(range(n_themes))
    ax.set_yticklabels(["\n".join(textwrap.wrap(l, 28)) for l in labels], fontsize=7.5)
    ax.set_title(
        "Theme share, after minus before (percentage points). * = 95% interval excludes zero,\n"
        "** = also survives Benjamini-Hochberg correction across all 36 tests (bold = reported as a finding)",
        fontsize=8.8, loc="left",
    )
    fig.colorbar(im, ax=ax, label="pp change", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build(data_path, events_path, images_dir):
    df = load_reviews(data_path)
    themed, labels, n_total_negative, n_no_theme = assign_themes(df)
    events = load_patch_events(events_path)
    wts = weekly_theme_share(df, themed, labels)
    shift = theme_shift_by_event(df, themed, labels, events)
    shift = add_bh_correction(shift)

    plot_theme_trends(wts, labels, events, f"{images_dir}/theme_trends_by_patch.png")
    plot_shift_heatmap(shift, labels, f"{images_dir}/theme_shift_by_event.png")

    print(f"n_total_negative={n_total_negative}, n_no_theme={n_no_theme} "
          f"({n_no_theme / n_total_negative * 100:.1f}%)")
    print(f"Wrote {images_dir}/theme_trends_by_patch.png")
    print(f"Wrote {images_dir}/theme_shift_by_event.png")
    print(f"BH-surviving (FDR 0.05): {int(shift['bh_significant'].sum())} of "
          f"{int((~shift['skipped']).sum())} tested")
    return df, themed, labels, n_total_negative, n_no_theme, events, wts, shift


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/reviews_raw.jsonl")
    ap.add_argument("--events", default="../data/patch_events.csv")
    ap.add_argument("--images-dir", default="../images")
    args = ap.parse_args()
    build(args.data, args.events, args.images_dir)
