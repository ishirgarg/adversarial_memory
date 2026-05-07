"""Success rate vs k (num memories retrieved), one subplot per dataset.

Persona-retrieval is split into two panels (non-misleading and misleading
questions), so the overall layout is 2 rows × 3 columns:

    Row 1: Coexisting | Conditional | Conditional (Hard)
    Row 2: Persona-Retrieval (Non-misleading) | Persona-Retrieval (Misleading) | Long-Hop
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from load_data import (
    MEMORY_COLORS,
    MEMORY_DISPLAY,
    MEMORY_SYSTEMS,
    X_AXIS_LABEL,
    group_by_k,
    load_analysis,
    style_k_axis,
    success_rate_with_ci,
)

OUT_DIR = Path(__file__).parent / "figures" / "success_vs_k"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Each panel: (dataset_key, title, optional record-filter, optional y-axis label).
# Persona-retrieval is split by question_type. Other datasets get no filter.
PANELS = [
    ("coexisting", "Coexisting-Facts", None),
    ("conditional", "Conditional-Facts", None),
    ("conditional_hard", "Conditional-Facts (Hard)", None),
    ("persona_retrieval", "Persona-Retrieval\n(Non-misleading)",
     lambda r: r.get("question_type") == "base"),
    ("persona_retrieval", "Persona-Retrieval\n(Misleading)",
     lambda r: r.get("question_type") == "misleading"),
    ("long_hop", "Long-Hop", None),
]

N_ROWS, N_COLS = 2, 3


def main():
    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(3.6 * N_COLS, 3.6 * N_ROWS),
        sharey=True,
    )
    axes = axes.flatten()

    for i, (ax, panel) in enumerate(zip(axes, PANELS)):
        dataset, title, rec_filter = panel
        for mem in MEMORY_SYSTEMS:
            recs = load_analysis(dataset, mem)
            if rec_filter is not None:
                recs = [r for r in recs if rec_filter(r)]
            if not recs:
                continue
            grouped = group_by_k(recs)
            ks = sorted(grouped.keys())
            ys, lo, hi = [], [], []
            for k in ks:
                res = success_rate_with_ci(grouped[k])
                p, l, h = (None, 0.0, 0.0) if res is None else res
                ys.append(p)
                lo.append(l)
                hi.append(h)
            ax.errorbar(
                ks, ys, yerr=[lo, hi],
                marker="o", capsize=3,
                label=MEMORY_DISPLAY[mem], color=MEMORY_COLORS[mem],
            )

        ax.set_title(title)
        ax.set_xlabel(X_AXIS_LABEL)
        if i % N_COLS == 0:
            ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        style_k_axis(ax)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[len(PANELS):]:
        ax.axis("off")

    fig.suptitle("Success rate vs. k")
    fig.tight_layout(pad=0.4, w_pad=0.3, rect=[0.03, 0, 1, 0.94])
    out = OUT_DIR / "overview.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
