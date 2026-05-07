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
    DATASET_TITLES,
    MEMORY_DISPLAY,
    MEMORY_SYSTEMS,
    group_by_k,
    load_analysis,
    normalized_error,
)

OUT_DIR = Path(__file__).parent / "figures" / "error_breakdown"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


# Per-dataset output files. Persona-retrieval is split into two files
# (non-misleading and misleading) by filtering on question_type.
# Tuple: (dataset, file_suffix, title_override_or_None, rec_filter_or_None).
_DATASET_PANELS = [
    ("coexisting", "coexisting", None, None),
    ("conditional", "conditional", None, None),
    ("conditional_hard", "conditional_hard", None, None),
    ("persona_retrieval", "persona_retrieval_nonmisleading",
     "Persona-Retrieval (Non-misleading)",
     lambda r: r.get("question_type") == "base"),
    ("persona_retrieval", "persona_retrieval_misleading",
     "Persona-Retrieval (Misleading)",
     lambda r: r.get("question_type") == "misleading"),
    ("long_hop", "long_hop", None, None),
]


def plot_dataset(dataset: str, file_suffix: str = None,
                 title_override: str = None, rec_filter=None):
    fig, ax = plt.subplots(figsize=(16, 6.5))

    bar_data = {et: [] for et in CANONICAL_ERRORS}

    x = 0
    xticks = []
    xtick_labels = []
    step = 1.5     # spacing between adjacent k bars within a group
    sep = 1.0      # extra gap between memory-system groups
    width = 1.0

    def _filtered(mem):
        recs = load_analysis(dataset, mem)
        if rec_filter is not None:
            recs = [r for r in recs if rec_filter(r)]
        return recs

    for mem in MEMORY_SYSTEMS:
        recs = _filtered(mem)
        if not recs:
            continue
        grouped = group_by_k(recs)
        ks = sorted(grouped.keys())
        if not ks:
            continue
        for k in ks:
            fracs, _ = error_breakdown(grouped[k])
            for et in CANONICAL_ERRORS:
                bar_data[et].append(fracs[et])
            xticks.append(x)
            xtick_labels.append(f"k={k}")
            x += step
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

    # Memory-system group labels under x ticks
    ymin = -0.07
    cursor = 0
    for mem in MEMORY_SYSTEMS:
        recs = _filtered(mem)
        if not recs:
            continue
        ks = sorted({r["_k"] for r in recs})
        if not ks:
            continue
        span_xs = xticks[cursor:cursor + len(ks)]
        cursor += len(ks)
        center = (span_xs[0] + span_xs[-1]) / 2
        ax.text(center, ymin, MEMORY_DISPLAY[mem], ha="center", va="top", fontsize=13, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Fraction of Errored Questions")
    title = title_override if title_override is not None else DATASET_TITLES[dataset]
    ax.set_title(f"Error breakdown vs. k — {title}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(pad=0.4, rect=[0, 0.05, 1, 1])
    out = OUT_DIR / f"{file_suffix or dataset}.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    plt.close(fig)
    return out


def main():
    for dataset, suffix, title_override, rec_filter in _DATASET_PANELS:
        plot_dataset(dataset, file_suffix=suffix,
                     title_override=title_override, rec_filter=rec_filter)


if __name__ == "__main__":
    main()
