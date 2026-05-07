"""For each (model, dataset), show how every error type
(storage / summary / retrieval / reasoning) shifts as k grows.
One subplot per memory system.

Normalization: each line is the fraction of *all graded questions* at that k —
denominator is the total number of records (correct + every error type), so
the 4 lines need not sum to 1; the remainder is the correct rate. This is the
question-level companion to plot_memory_errors_vs_k.py, which divides by
memory errors only.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from load_data import (
    CANONICAL_ERRORS,
    DATASETS,
    DATASET_TITLES,
    MEMORY_DISPLAY,
    MEMORY_SYSTEMS,
    REPORT_MODELS,
    X_AXIS_LABEL,
    load_analysis,
    normalized_error,
    style_k_axis,
)

OUT_DIR = Path(__file__).parent / "figures" / "all_errors_vs_k"

# Match the legend ordering used by the memory-errors plot, and append reasoning.
ERROR_TYPES = ["storage", "summary", "retrieval", "reasoning"]
assert set(ERROR_TYPES) == set(CANONICAL_ERRORS), \
    "ERROR_TYPES must cover every canonical error bucket."

ERROR_COLORS = {
    "storage": "#ff7f0e",
    "summary": "#9467bd",
    "retrieval": "#1f77b4",
    "reasoning": "#2ca02c",
}


def all_error_fractions(records, z: float = 1.96):
    """For records at one (memory, k), return {error_type: (p, lower_err, upper_err)}
    and n_total, with n_total = number of graded questions (correct + all errors).
    Wilson score interval keeps CIs non-zero at p=0/1.

    Each fraction is (count of that error type) / (all questions), so the four
    lines do not sum to 1 — the missing mass is the correct rate. We use this
    framing instead of normalizing by error count so a system that simply gets
    fewer questions wrong shows up as a lower line, not the same shape.
    """
    buckets = {et: 0 for et in ERROR_TYPES}
    for r in records:
        et = normalized_error(r)
        if et in buckets:
            buckets[et] += 1
    n = len(records)
    out = {}
    for et in ERROR_TYPES:
        c = buckets[et]
        if n == 0:
            out[et] = (None, 0.0, 0.0)
        else:
            p = c / n
            denom = 1.0 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
            lower = max(0.0, center - half)
            upper = min(1.0, center + half)
            out[et] = (p, max(0.0, p - lower), max(0.0, upper - p))
    return out, n


def collect_per_k(dataset, memory, model, rec_filter=None):
    recs = load_analysis(dataset, memory, model=model)
    out = defaultdict(list)
    for r in recs:
        if rec_filter is not None and not rec_filter(r):
            continue
        out[r["_k"]].append(r)
    return dict(sorted(out.items()))


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


def plot_for_dataset_model(dataset: str, model: str, file_suffix: str = None,
                           title_override: str = None, rec_filter=None):
    n_mem = len(MEMORY_SYSTEMS)
    fig, axes = plt.subplots(1, n_mem, figsize=(3.4 * n_mem, 3.4), sharex=True, sharey=True)
    if n_mem == 1:
        axes = [axes]

    any_data = False
    for i, (ax, memory) in enumerate(zip(axes, MEMORY_SYSTEMS)):
        per_k = collect_per_k(dataset, memory, model, rec_filter=rec_filter)
        ks = sorted(per_k.keys())
        if not ks:
            ax.set_title(f"{MEMORY_DISPLAY[memory]} — no data")
            ax.set_xlabel(X_AXIS_LABEL)
            ax.set_ylim(0, 1)
            style_k_axis(ax)
            continue

        per_et = {et: ([], [], [], []) for et in ERROR_TYPES}
        ks_with_data = []
        for k in ks:
            fracs, n = all_error_fractions(per_k[k])
            if n == 0:
                continue
            ks_with_data.append(k)
            for et in ERROR_TYPES:
                p, lo, hi = fracs[et]
                per_et[et][0].append(k)
                per_et[et][1].append(p)
                per_et[et][2].append(lo)
                per_et[et][3].append(hi)
        ks = ks_with_data
        if not ks:
            ax.set_title(f"{MEMORY_DISPLAY[memory]} — no questions")
            ax.set_xlabel(X_AXIS_LABEL)
            ax.set_ylim(0, 1)
            style_k_axis(ax)
            continue

        for et in ERROR_TYPES:
            xs, ys, lo, hi = per_et[et]
            ax.errorbar(
                xs, ys, yerr=[lo, hi],
                marker="o", capsize=3, label=et,
                color=ERROR_COLORS[et],
            )

        ax.set_title(MEMORY_DISPLAY[memory])
        ax.set_xlabel(X_AXIS_LABEL)
        if i == 0:
            ax.set_ylabel("Fraction of Questions")
        ax.set_ylim(0, 1.10)
        style_k_axis(ax)
        ax.grid(True, alpha=0.3)
        ax.legend()
        any_data = True

    if not any_data:
        plt.close(fig)
        return None

    title = title_override if title_override is not None else DATASET_TITLES[dataset]
    fig.suptitle(f"All-error breakdown vs. k — {title} / {model}")
    fig.tight_layout(pad=0.4, w_pad=0.3, rect=[0.05, 0, 1, 0.94])

    out_dir = OUT_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{file_suffix or dataset}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
    plt.close(fig)
    return out


def main():
    for model in REPORT_MODELS:
        for dataset, suffix, title_override, rec_filter in _DATASET_PANELS:
            plot_for_dataset_model(dataset, model,
                                   file_suffix=suffix,
                                   title_override=title_override,
                                   rec_filter=rec_filter)


if __name__ == "__main__":
    main()
