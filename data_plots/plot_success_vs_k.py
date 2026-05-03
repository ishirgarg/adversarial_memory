"""Success rate vs k (num memories retrieved), one subplot per dataset."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from load_data import (
    DATASETS,
    MEMORY_SYSTEMS,
    group_by_k,
    is_correct,
    load_analysis,
)

OUT_DIR = Path(__file__).parent / "figures" / "success_vs_k"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_TITLES = {
    "coexisting": "Coexisting facts",
    "conditional": "Conditional",
    "conditional_hard": "Conditional (hard)",
    "persona_retrieval": "Persona retrieval",
}

COLORS = {
    "mem0": "#1f77b4",
    "simplemem": "#2ca02c",
    "amem": "#d62728",
}


def success_rate(records):
    if not records:
        return None
    return sum(1 for r in records if is_correct(r)) / len(records)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes = axes.flatten()

    for ax, dataset in zip(axes, DATASETS):
        for mem in MEMORY_SYSTEMS:
            recs = load_analysis(dataset, mem)
            if not recs:
                continue
            grouped = group_by_k(recs)
            ks = sorted(grouped.keys())
            ys = [success_rate(grouped[k]) for k in ks]
            ax.plot(ks, ys, marker="o", label=mem, color=COLORS.get(mem))

        ax.set_title(DATASET_TITLES[dataset])
        ax.set_xlabel("k (num memories retrieved)")
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Success rate vs. k", fontsize=14)
    fig.tight_layout()
    out = OUT_DIR / "overview.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
