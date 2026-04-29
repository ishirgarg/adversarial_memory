"""For each dataset, show how the error mix (storage / summary / retrieval / reasoning)
shifts as k grows. Produces one figure per dataset; within each, one stacked bar
per (memory_system, k)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_data import (
    CANONICAL_ERRORS,
    DATASETS,
    MEMORY_SYSTEMS,
    group_by_k,
    load_analysis,
    normalized_error,
)

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

DATASET_TITLES = {
    "coexisting": "Coexisting facts",
    "conditional": "Conditional",
    "conditional_hard": "Conditional (hard)",
    "persona_retrieval": "Persona retrieval",
}

ERROR_COLORS = {
    "storage": "#ff7f0e",
    "summary": "#9467bd",
    "retrieval": "#1f77b4",
    "reasoning": "#2ca02c",
}


def error_breakdown(records):
    """Among the errored records, return fraction per canonical error bucket
    and total errored count."""
    errs = [normalized_error(r) for r in records if normalized_error(r) is not None]
    total = len(errs)
    counts = Counter(errs)
    fracs = {et: (counts.get(et, 0) / total if total else 0.0) for et in CANONICAL_ERRORS}
    return fracs, total


def plot_dataset(dataset: str):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    bar_labels = []
    bar_data = {et: [] for et in CANONICAL_ERRORS}
    bar_totals = []
    group_starts = []  # x positions where each memory-system group begins

    x = 0
    xticks = []
    xtick_labels = []
    sep = 0.6  # gap between memory-system groups
    width = 0.8

    for mem in MEMORY_SYSTEMS:
        recs = load_analysis(dataset, mem)
        if not recs:
            continue
        grouped = group_by_k(recs)
        ks = sorted(grouped.keys())
        if not ks:
            continue
        group_starts.append((mem, x))
        for k in ks:
            fracs, total = error_breakdown(grouped[k])
            for et in CANONICAL_ERRORS:
                bar_data[et].append(fracs[et])
            bar_totals.append(total)
            bar_labels.append(f"k={k}")
            xticks.append(x)
            xtick_labels.append(f"k={k}")
            x += 1
        x += sep  # gap before next memory system

    if not xticks:
        plt.close(fig)
        return None

    xs = np.array(xticks, dtype=float)
    bottoms = np.zeros(len(xs))
    for et in CANONICAL_ERRORS:
        vals = np.array(bar_data[et])
        ax.bar(xs, vals, width=width, bottom=bottoms, label=et, color=ERROR_COLORS[et])
        bottoms += vals

    # Annotate total errored count above each bar
    for xi, total in zip(xs, bar_totals):
        ax.text(xi, 1.02, f"n={total}", ha="center", va="bottom", fontsize=7, color="#444")

    # Memory-system group labels under x ticks
    ymin = -0.07
    cursor = 0
    for mem in MEMORY_SYSTEMS:
        recs = load_analysis(dataset, mem)
        if not recs:
            continue
        ks = sorted({r["_k"] for r in recs})
        if not ks:
            continue
        span_xs = xticks[cursor:cursor + len(ks)]
        cursor += len(ks)
        center = (span_xs[0] + span_xs[-1]) / 2
        ax.text(center, ymin, mem, ha="center", va="top", fontsize=10, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("fraction of errored questions")
    ax.set_title(f"Error breakdown vs. k — {DATASET_TITLES[dataset]}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=9, ncol=4, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUT_DIR / f"errors_vs_k_{dataset}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)
    return out


def main():
    for dataset in DATASETS:
        plot_dataset(dataset)


if __name__ == "__main__":
    main()
