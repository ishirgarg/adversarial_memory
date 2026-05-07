"""Accuracy vs. avg tokens-per-memory.

For every k value, draw one figure with 4 side-by-side subplots (one per
test-taker model). In each subplot:
- y-axis: success rate (with Wilson 95% CI)
- x-axis: mean memory_tokens / k across graded turns (log scale, so the dense
  low-token cluster and long-tail outliers fit in one contiguous panel)
- one line per dataset (color encodes dataset)
- one marker per memory system (shape encodes memory system; legend below)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from load_data import (
    DATASETS,
    DATASET_TITLES,
    MEMORY_DISPLAY,
    MEMORY_SYSTEMS,
    REPORT_MODELS,
    load_analysis,
    mean_memory_tokens_per_memory,
    success_rate_with_ci,
)

OUT_DIR = Path(__file__).parent / "figures" / "acc_vs_tokens_per_memory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [3, 5, 10, 15, 20]

DATASET_COLORS = {
    "coexisting": "#1f77b4",
    "conditional": "#2ca02c",
    "conditional_hard": "#d62728",
    "persona_retrieval": "#ff7f0e",
    "long_hop": "#9467bd",
}

# Marker shape per memory system — replaces the per-point text annotations
# that were overlapping in the previous layout.
MEMORY_MARKERS = {
    "mem0": "o",
    "simplemem": "s",
    "amem": "^",
    "structmem": "D",
}


def _gather(model: str, k: int):
    """Per-dataset point series for a single (model, k)."""
    series = []
    for dataset in DATASETS:
        xs, ys, lo, hi, labels = [], [], [], [], []
        for mem in MEMORY_SYSTEMS:
            recs = load_analysis(dataset, mem, k=k, model=model)
            if not recs:
                continue
            res = success_rate_with_ci(recs)
            if res is None:
                continue
            tokens = mean_memory_tokens_per_memory(dataset, mem, k=k, model=model)
            if tokens is None or tokens <= 0:
                continue
            p, l, h = res
            xs.append(tokens)
            ys.append(p)
            lo.append(l)
            hi.append(h)
            labels.append(mem)
        if not xs:
            continue
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        series.append({
            "dataset": dataset,
            "xs": [xs[i] for i in order],
            "ys": [ys[i] for i in order],
            "lo": [lo[i] for i in order],
            "hi": [hi[i] for i in order],
            "labels": [labels[i] for i in order],
        })
    return series


def _plot_series(ax, series):
    for s in series:
        color = DATASET_COLORS.get(s["dataset"])
        # Connecting line drawn first so markers sit on top.
        ax.plot(s["xs"], s["ys"], color=color, linewidth=1.5, zorder=2)
        for x, y, l, h, lab in zip(s["xs"], s["ys"], s["lo"], s["hi"], s["labels"]):
            ax.errorbar(
                [x], [y], yerr=[[l], [h]],
                marker=MEMORY_MARKERS[lab], markersize=8,
                color=color, capsize=3, linestyle="none", zorder=3,
            )


def _plot_for_k(k: int) -> None:
    fig, axes = plt.subplots(
        1, len(REPORT_MODELS),
        figsize=(3.6 * len(REPORT_MODELS), 4.2),
        sharey=True, sharex=True,
    )
    axes = np.atleast_1d(axes).flatten()

    any_drawn = False
    all_xs: list[float] = []
    for ax, model in zip(axes, REPORT_MODELS):
        series = _gather(model, k)
        if series:
            any_drawn = True
            _plot_series(ax, series)
            for s in series:
                all_xs.extend(s["xs"])
        ax.set_title(model)
        ax.set_xscale("log")
        ax.set_xlabel("Avg tokens per retrieved memory")
        ax.set_ylim(0, 1)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Success Rate")

    if not any_drawn:
        plt.close(fig)
        return

    xmin, xmax = min(all_xs), max(all_xs)
    for ax in axes:
        ax.set_xlim(xmin / 1.15, xmax * 1.15)

    # Two-section legend: datasets (color) on the left, memory systems (marker)
    # on the right. Placed below the row of subplots so it never overlaps data.
    dataset_handles = [
        Line2D([0], [0], color=DATASET_COLORS[d], marker="o", linewidth=1.5,
               markersize=7, label=DATASET_TITLES[d])
        for d in DATASETS
    ]
    memory_handles = [
        Line2D([0], [0], color="black", marker=MEMORY_MARKERS[m],
               linestyle="none", markersize=8, label=MEMORY_DISPLAY[m])
        for m in MEMORY_SYSTEMS
    ]
    # Two stacked legends so datasets and memory systems stay grouped, and the
    # whole block stays narrower than the row of subplots.
    legend_kwargs = dict(
        loc="lower center", frameon=False,
        columnspacing=1.2, handletextpad=0.5,
    )
    leg_ds = fig.legend(
        handles=dataset_handles, ncol=len(dataset_handles),
        bbox_to_anchor=(0.5, 0.02), **legend_kwargs,
    )
    fig.add_artist(leg_ds)
    fig.legend(
        handles=memory_handles, ncol=len(memory_handles),
        bbox_to_anchor=(0.5, -0.03), **legend_kwargs,
    )

    fig.suptitle(f"Accuracy vs. Tokens/Memory — k={k}")
    fig.tight_layout(rect=[0, 0.11, 1, 0.97])
    out = OUT_DIR / f"k{k}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    for k in K_VALUES:
        _plot_for_k(k)


if __name__ == "__main__":
    main()
