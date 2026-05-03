"""Generate the full report bundle: summary tables + per-dataset / per-model plots.

Output layout under data_plots/figures/:

  tables/
    summary_scores.md
    summary_tokens.md
  success_vs_k/<model>/<dataset>.png
  perf_vs_model/<memory>.png
  errors_vs_k/<model>/<dataset>.png
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from load_data import (
    DATASETS,
    MEMORY_SYSTEMS,
    is_correct,
    load_analysis,
)

FIG_DIR = Path(__file__).parent / "figures"

DATASET_TITLES = {
    "coexisting": "Coexisting facts",
    "conditional": "Conditional",
    "conditional_hard": "Conditional (hard)",
    "persona_retrieval": "Persona retrieval",
}

MEMORY_COLORS = {
    "mem0": "#1f77b4",
    "simplemem": "#2ca02c",
    "amem": "#d62728",
}

MODEL_COLORS = {
    "gpt-4.1-mini": "#1f77b4",
    "haiku-4.5": "#d62728",
}

# Models we report on across all tables/plots.
REPORT_MODELS = ["gpt-4.1-mini", "haiku-4.5"]
REPORT_KS = [5, 10, 15]


# ----------------------------- aggregation helpers -----------------------------

def success_rate(records) -> Optional[float]:
    if not records:
        return None
    return sum(1 for r in records if is_correct(r)) / len(records)


def error_rate_with_se(records) -> Optional[tuple]:
    """Return (error_rate, standard_error) using the binomial-proportion SE.

    Standard error = sqrt(p*(1-p)/n) on the error proportion p = 1 - success.
    """
    if not records:
        return None
    n = len(records)
    correct = sum(1 for r in records if is_correct(r))
    p_err = 1.0 - correct / n
    se = math.sqrt(p_err * (1.0 - p_err) / n) if n > 0 else 0.0
    return p_err, se


def avg_tokens(records) -> Optional[float]:
    if not records:
        return None
    totals = [
        (r.get("eval_input_tokens") or 0) + (r.get("eval_output_tokens") or 0)
        for r in records
    ]
    return sum(totals) / len(totals)


def collect_records(dataset: str, memory: str, model: str) -> Dict[int, list]:
    recs = load_analysis(dataset, memory, model=model)
    out: Dict[int, list] = defaultdict(list)
    for r in recs:
        out[r["_k"]].append(r)
    return out


# ----------------------------------- tables -----------------------------------

def _fmt_score(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.3f}"


def _fmt_int(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.0f}"


def _build_table(metric: str) -> str:
    """metric: 'score' or 'tokens'."""
    cols: List[tuple] = [(model, k) for model in REPORT_MODELS for k in REPORT_KS]
    header_top = "| Dataset | Memory | " + " | ".join(
        f"{model} k={k}" for model, k in cols
    ) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(cols))) + "|"

    lines = [header_top, sep]
    for dataset in DATASETS:
        # Compute every cell first so we can find per-column maxima within the dataset.
        rows: Dict[str, Dict[tuple, Optional[float]]] = {}
        for memory in MEMORY_SYSTEMS:
            cells: Dict[tuple, Optional[float]] = {}
            for model, k in cols:
                bucket = collect_records(dataset, memory, model).get(k, [])
                if not bucket:
                    cells[(model, k)] = None
                elif metric == "score":
                    cells[(model, k)] = success_rate(bucket)
                else:
                    cells[(model, k)] = avg_tokens(bucket)
            rows[memory] = cells

        # Pick best per column (max score, min tokens — both still highlight the "winner").
        best: Dict[tuple, Optional[str]] = {}
        for col in cols:
            vals = [(m, rows[m][col]) for m in MEMORY_SYSTEMS if rows[m][col] is not None]
            if not vals:
                best[col] = None
                continue
            if metric == "score":
                best[col] = max(vals, key=lambda x: x[1])[0]
            else:
                best[col] = min(vals, key=lambda x: x[1])[0]

        for i, memory in enumerate(MEMORY_SYSTEMS):
            cells = rows[memory]
            label_dataset = DATASET_TITLES[dataset] if i == 0 else ""
            cell_strs = []
            for col in cols:
                v = cells[col]
                s = _fmt_score(v) if metric == "score" else _fmt_int(v)
                if best[col] == memory and v is not None:
                    s = f"**{s}**"
                cell_strs.append(s)
            lines.append(
                f"| {label_dataset} | {memory} | " + " | ".join(cell_strs) + " |"
            )
    return "\n".join(lines) + "\n"


def write_tables() -> None:
    out_dir = FIG_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    score_md = "# Mean success rate per memory system\n\n"
    score_md += "Bold = best score in each (dataset, column).\n\n"
    score_md += _build_table("score")
    (out_dir / "summary_scores.md").write_text(score_md, encoding="utf-8")
    print(f"wrote {out_dir / 'summary_scores.md'}")

    tok_md = "# Average tokens per graded question\n\n"
    tok_md += "Sum of `eval_input_tokens + eval_output_tokens`. Bold = lowest in each (dataset, column).\n\n"
    tok_md += _build_table("tokens")
    (out_dir / "summary_tokens.md").write_text(tok_md, encoding="utf-8")
    print(f"wrote {out_dir / 'summary_tokens.md'}")


# ------------------------- success-vs-k (per test-taker) -----------------------

def plot_success_vs_k_per_model() -> None:
    base = FIG_DIR / "success_vs_k"
    for model in REPORT_MODELS:
        out_dir = base / model
        out_dir.mkdir(parents=True, exist_ok=True)
        for dataset in DATASETS:
            fig, ax = plt.subplots(figsize=(6.2, 4.4))
            any_data = False
            for memory in MEMORY_SYSTEMS:
                grouped = collect_records(dataset, memory, model)
                ks = sorted(grouped.keys())
                if not ks:
                    continue
                ys = [success_rate(grouped[k]) for k in ks]
                ax.plot(ks, ys, marker="o", label=memory, color=MEMORY_COLORS[memory])
                any_data = True
            if not any_data:
                plt.close(fig)
                continue
            ax.set_xlabel("k (num memories retrieved)")
            ax.set_ylabel("success rate")
            ax.set_ylim(0, 1)
            ax.set_title(f"{DATASET_TITLES[dataset]} — {model}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            out = out_dir / f"{dataset}.png"
            fig.savefig(out, dpi=150)
            print(f"wrote {out}")
            plt.close(fig)


# ----------------------- performance vs test-taker model ----------------------

def plot_perf_vs_model() -> None:
    out_dir = FIG_DIR / "perf_vs_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ds = len(DATASETS)
    n_cols = 2
    n_rows = math.ceil(n_ds / n_cols)
    for memory in MEMORY_SYSTEMS:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4.0 * n_rows), sharey=True)
        axes = np.array(axes).flatten()
        any_curve = False
        for ax, dataset in zip(axes, DATASETS):
            ds_any = False
            for model in REPORT_MODELS:
                grouped = collect_records(dataset, memory, model)
                ks = sorted(grouped.keys())
                if not ks:
                    continue
                ys = [success_rate(grouped[k]) for k in ks]
                ax.plot(ks, ys, marker="o", label=model, color=MODEL_COLORS.get(model))
                ds_any = True
                any_curve = True
            ax.set_title(DATASET_TITLES[dataset])
            ax.set_xlabel("k")
            ax.set_ylabel("success rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if ds_any:
                ax.legend(fontsize=8)
        for ax in axes[n_ds:]:
            ax.axis("off")
        fig.suptitle(f"Success vs. k by test-taker model — {memory}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = out_dir / f"{memory}.png"
        if any_curve:
            fig.savefig(out, dpi=150)
            print(f"wrote {out}")
        plt.close(fig)


# ------------------------ fraction-of-errors per dataset ----------------------

def plot_errors_vs_k_per_model() -> None:
    base = FIG_DIR / "errors_vs_k"
    for model in REPORT_MODELS:
        out_dir = base / model
        out_dir.mkdir(parents=True, exist_ok=True)
        for dataset in DATASETS:
            fig, ax = plt.subplots(figsize=(6.2, 4.4))
            any_data = False
            for memory in MEMORY_SYSTEMS:
                grouped = collect_records(dataset, memory, model)
                ks = sorted(grouped.keys())
                if not ks:
                    continue
                xs, ys, errs = [], [], []
                for k in ks:
                    res = error_rate_with_se(grouped[k])
                    if res is None:
                        continue
                    p, se = res
                    xs.append(k)
                    ys.append(p)
                    errs.append(se)
                if not xs:
                    continue
                ax.errorbar(
                    xs, ys, yerr=errs,
                    marker="o", capsize=3, label=memory,
                    color=MEMORY_COLORS[memory],
                )
                any_data = True
            if not any_data:
                plt.close(fig)
                continue
            ax.set_xlabel("k (num memories retrieved)")
            ax.set_ylabel("fraction of errored questions")
            ax.set_ylim(0, 1)
            ax.set_title(f"{DATASET_TITLES[dataset]} — {model}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            out = out_dir / f"{dataset}.png"
            fig.savefig(out, dpi=150)
            print(f"wrote {out}")
            plt.close(fig)


# ------------------------------------ main ------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_tables()
    plot_success_vs_k_per_model()
    plot_perf_vs_model()
    plot_errors_vs_k_per_model()


if __name__ == "__main__":
    main()
