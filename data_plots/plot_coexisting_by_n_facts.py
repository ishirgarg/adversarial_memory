"""Coexisting-facts: error breakdown conditioned on n_preferences (the number of
coexisting facts in the question), shown for each k."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_data import (
    CANONICAL_ERRORS,
    MEMORY_SYSTEMS,
    is_correct,
    load_analysis,
    n_preferences,
    normalized_error,
)

OUT_DIR = Path(__file__).parent / "figures" / "coexisting_by_nfacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_COLORS = {
    "storage": "#ff7f0e",
    "summary": "#9467bd",
    "retrieval": "#1f77b4",
    "reasoning": "#2ca02c",
}


def breakdown(records):
    errs = [normalized_error(r) for r in records if normalized_error(r) is not None]
    total = len(errs)
    counts = Counter(errs)
    return {et: (counts.get(et, 0) / total if total else 0.0) for et in CANONICAL_ERRORS}, total


def collect(memory: str):
    """Return {k: {n_pref: [records]}}."""
    recs = load_analysis("coexisting", memory)
    by_k_n: dict = defaultdict(lambda: defaultdict(list))
    for r in recs:
        n = n_preferences(r)
        if n is None:
            continue
        by_k_n[r["_k"]][n].append(r)
    return {k: dict(v) for k, v in sorted(by_k_n.items())}


def plot_for_memory(memory: str):
    by_k_n = collect(memory)
    if not by_k_n:
        return None

    ks = sorted(by_k_n.keys())
    all_ns = sorted({n for d in by_k_n.values() for n in d.keys()})

    fig, axes = plt.subplots(1, len(ks), figsize=(3.5 * len(ks), 5), sharey=True)
    if len(ks) == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        bottoms = np.zeros(len(all_ns))
        totals_err = []
        totals_all = []
        for et in CANONICAL_ERRORS:
            vals = []
            for n in all_ns:
                recs = by_k_n[k].get(n, [])
                fracs, _total_err = breakdown(recs)
                vals.append(fracs[et])
            ax.bar(all_ns, vals, bottom=bottoms, color=ERROR_COLORS[et], label=et if k == ks[0] else None)
            bottoms += np.array(vals)

        # totals annotations
        for n in all_ns:
            recs = by_k_n[k].get(n, [])
            n_all = len(recs)
            n_err = sum(1 for r in recs if not is_correct(r))
            if n_all:
                ax.text(n, 1.02, f"{n_err}/{n_all}", ha="center", va="bottom", fontsize=7, color="#444")

        ax.set_title(f"k={k}")
        ax.set_xlabel("# coexisting facts")
        ax.set_xticks(all_ns)
        ax.set_ylim(0, 1.10)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("fraction of errored questions")
    fig.suptitle(f"Coexisting facts — error breakdown by # facts and k ({memory})", fontsize=13)

    handles, labels = [], []
    for et in CANONICAL_ERRORS:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=ERROR_COLORS[et]))
        labels.append(et)
    fig.legend(handles, labels, loc="upper right", ncol=4, fontsize=9,
               bbox_to_anchor=(0.99, 0.97))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / f"errors_{memory}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)
    return out


def plot_success_by_n_facts():
    """Bonus: success rate vs n_preferences, one line per memory system, faceted by k."""
    per_mem = {m: collect(m) for m in MEMORY_SYSTEMS}
    per_mem = {m: v for m, v in per_mem.items() if v}
    if not per_mem:
        return None

    ks = sorted({k for v in per_mem.values() for k in v.keys()})
    fig, axes = plt.subplots(1, len(ks), figsize=(3.6 * len(ks), 4.5), sharey=True)
    if len(ks) == 1:
        axes = [axes]

    color = {"mem0": "#1f77b4", "simplemem": "#2ca02c", "amem": "#d62728"}
    for ax, k in zip(axes, ks):
        for mem, by_k_n in per_mem.items():
            if k not in by_k_n:
                continue
            ns = sorted(by_k_n[k].keys())
            ys = []
            for n in ns:
                recs = by_k_n[k][n]
                ys.append(sum(1 for r in recs if is_correct(r)) / len(recs) if recs else None)
            ax.plot(ns, ys, marker="o", label=mem, color=color.get(mem))
        ax.set_title(f"k={k}")
        ax.set_xlabel("# coexisting facts")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("success rate")
    fig.suptitle("Coexisting facts — success rate vs. # coexisting facts", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT_DIR / "success.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)
    return out


def main():
    for mem in MEMORY_SYSTEMS:
        plot_for_memory(mem)
    plot_success_by_n_facts()


if __name__ == "__main__":
    main()
