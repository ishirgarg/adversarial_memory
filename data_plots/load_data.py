"""Shared loader for analysis JSONs across datasets/memory-systems/k values."""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as _mpl
from matplotlib.ticker import MultipleLocator, NullLocator

# Global matplotlib style — applied at import so every plotting module inherits
# the same axis/legend/label sizes.
_mpl.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
})


def style_k_axis(ax) -> None:
    """Force integer-only x-axis ticks at multiples of 4 (0, 4, 8, ...)."""
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)

X_AXIS_LABEL = "k"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYGROUND = PROJECT_ROOT / "playground"

DATASETS = {
    "coexisting": PLAYGROUND / "coexisting_facts" / "results",
    "conditional": PLAYGROUND / "conditional_facts" / "results",
    "conditional_hard": PLAYGROUND / "conditional_facts" / "results",
    "persona_retrieval": PLAYGROUND / "custom_persona_retrieval" / "results",
    "long_hop": PLAYGROUND / "long_hop" / "results",
}

# Canonical memory-system order (used as legend order everywhere).
MEMORY_SYSTEMS = ["mem0", "simplemem", "amem", "structmem"]

MEMORY_DISPLAY = {
    "mem0": "Mem0",
    "simplemem": "SimpleMem",
    "amem": "A-MEM",
    "structmem": "StructMem",
}

MEMORY_COLORS = {
    "mem0": "#1f77b4",
    "simplemem": "#2ca02c",
    "amem": "#d62728",
    "structmem": "#9467bd",
}

# Per-memory-system metadata field that holds the internal LLM model.
MEMORY_MODEL_FIELD = {
    "mem0": "mem0_llm_model",
    "amem": "amem_llm_model",
    "simplemem": "simplemem_model",
    "structmem": "structmem_model",
}

# Canonical test-taker model order (used as legend order everywhere).
REPORT_MODELS = ["gpt-4.1-mini", "haiku-4.5", "gpt-5.4-mini", "gemini-3.1"]

MODEL_COLORS = {
    "gpt-4.1-mini": "#1f77b4",
    "haiku-4.5": "#d62728",
    "gpt-5.4-mini": "#2ca02c",
    "gemini-3.1": "#ff7f0e",
}

# Normalize raw model strings to short display names used everywhere.
MODEL_DISPLAY = {
    "gpt-4.1-mini": "gpt-4.1-mini",
    "vertex_ai/claude-haiku-4-5": "haiku-4.5",
    "claude-haiku-4-5": "haiku-4.5",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5-mini": "gpt-5.4-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "gemini/gemini-3.1-pro-preview": "gemini-3.1",
    "gemini-3.1-pro-preview": "gemini-3.1",
}

DATASET_TITLES = {
    "coexisting": "Coexisting-Facts",
    "conditional": "Conditional-Facts",
    "conditional_hard": "Conditional-Facts (Hard)",
    "persona_retrieval": "Persona-Retrieval",
    "long_hop": "Long-Hop",
}


def normalize_model(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    return MODEL_DISPLAY.get(raw, raw)


def memory_model(meta: Dict[str, Any], memory: str) -> Optional[str]:
    """Pull the internal LLM model used by `memory` from a graded_traces metadata blob."""
    cli = meta.get("all_cli_args", {}) or {}
    field = MEMORY_MODEL_FIELD.get(memory)
    if not field:
        return None
    return normalize_model(cli.get(field))

CANONICAL_ERRORS = ["storage", "summary", "retrieval", "reasoning"]

ERROR_NORMALIZE = {
    "storage_error": "storage",
    "not_stored": "storage",
    "summary_error": "summary",
    "retrieval_error": "retrieval",
    "not_retrieved": "retrieval",
    "reasoning_error": "reasoning",
}


def _is_hard_run(run_dir: str) -> bool:
    name = os.path.basename(run_dir)
    # New naming carries HARD as a task token: `run_<task>_HARD__...`.
    # Old naming used a `run_hard_` prefix; keep that as a fallback.
    return "_HARD__" in name or name.startswith("run_hard_")


def _list_runs(dataset: str) -> List[str]:
    base = DATASETS[dataset]
    runs = sorted(glob.glob(str(base / "run_*")))
    if dataset == "conditional":
        return [r for r in runs if not _is_hard_run(r)]
    if dataset == "conditional_hard":
        return [r for r in runs if _is_hard_run(r)]
    return runs


def _latest_analysis(subdir: Path) -> Optional[Path]:
    files = sorted(subdir.glob("analysis_*.json"))
    return files[-1] if files else None


def load_analysis(
    dataset: str,
    memory: str,
    k: Optional[int] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return concatenated analysis records (per-question dicts) for the given
    dataset/memory combo. If k is given, filter to that num_memories. If model
    is given, filter to runs where the memory system's internal LLM matches
    (matched against the normalized display name, e.g. "gpt-4.1-mini" or
    "haiku-4.5").

    Each returned record is augmented with ``_k`` (the num_memories of its run),
    ``_dataset``, ``_memory`` and ``_model`` for convenience.

    When multiple completed sister runs share the same (k, model) — e.g. the
    same combo was evaluated twice on different days — only the latest one
    contributes, to avoid double-counting questions in pooled metrics.
    """
    # Pick the latest matching run per (run_k, run_model). _list_runs returns
    # paths sorted ascending by name (timestamps are part of the dir name), so
    # later writes overwrite earlier ones.
    latest: Dict[Tuple[Optional[int], Optional[str]], Tuple[Path, Optional[int], Optional[str]]] = {}
    for run_dir in _list_runs(dataset):
        sub = Path(run_dir) / memory
        if not sub.is_dir():
            continue
        analysis = _latest_analysis(sub)
        if analysis is None:
            continue
        gt_files = sorted(sub.glob("graded_traces_*.json"))
        if not gt_files:
            continue
        with open(gt_files[-1], "r", encoding="utf-8") as f:
            meta = json.load(f).get("run_metadata", {})
        run_k = meta.get("num_memories")
        if k is not None and run_k != k:
            continue
        run_model = memory_model(meta, memory)
        if model is not None and run_model != model:
            continue
        latest[(run_k, run_model)] = (analysis, run_k, run_model)

    out: List[Dict[str, Any]] = []
    for (analysis, run_k, run_model) in latest.values():
        with open(analysis, "r", encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            r["_k"] = run_k
            r["_dataset"] = dataset
            r["_memory"] = memory
            r["_model"] = run_model
        out.extend(records)
    return out


def available_models(dataset: str, memory: str) -> List[str]:
    return sorted({r["_model"] for r in load_analysis(dataset, memory) if r.get("_model")})


def mean_memory_tokens_per_memory(
    dataset: str,
    memory: str,
    k: int,
    model: Optional[str] = None,
) -> Optional[float]:
    """Mean (memory_tokens / k) over graded turns matching the given
    dataset/memory/k(/model) filter. Returns None if no graded turns are found.

    When multiple completed sister runs share the same (k, model), only the
    latest one is used (matches load_analysis behavior — avoids double-counting).
    """
    # Pick the latest matching run per model. _list_runs is sorted ascending by
    # name, so later writes overwrite earlier ones.
    latest: Dict[Optional[str], Path] = {}
    for run_dir in _list_runs(dataset):
        sub = Path(run_dir) / memory
        if not sub.is_dir():
            continue
        gt_files = sorted(sub.glob("graded_traces_*.json"))
        if not gt_files:
            continue
        with open(gt_files[-1], "r", encoding="utf-8") as f:
            meta = json.load(f).get("run_metadata", {})
        run_k = meta.get("num_memories")
        if run_k != k:
            continue
        run_model = memory_model(meta, memory)
        if model is not None and run_model != model:
            continue
        traces_files = sorted(sub.glob("traces_*.json"))
        if not traces_files:
            continue
        latest[run_model] = traces_files[-1]

    total = 0.0
    n = 0
    for traces_path in latest.values():
        with open(traces_path, "r", encoding="utf-8") as f:
            traces_blob = json.load(f)
        for res in traces_blob.get("evaluation_summary", {}).get("results", []):
            for t in res.get("traces", []):
                if t.get("should_grade"):
                    total += t.get("memory_tokens", 0)
                    n += 1
    if n == 0:
        return None
    return (total / n) / k


def all_records(dataset: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mem in MEMORY_SYSTEMS:
        out.extend(load_analysis(dataset, mem))
    return out


def is_correct(record: Dict[str, Any]) -> bool:
    """Treat judge_result == 'correct' as success."""
    return record.get("judge_result") == "correct"


def success_rate_with_ci(records, z: float = 1.96) -> Optional[Tuple[float, float, float]]:
    """Binomial success rate with Wilson 95% CI.

    Returns (p, lower_err, upper_err) where lower_err = p - lower_bound and
    upper_err = upper_bound - p (i.e. the offsets to use for matplotlib's
    asymmetric yerr). Wilson keeps a non-zero interval at p=0 and p=1, where
    the Wald formula sqrt(p(1-p)/n) collapses. Returns None for empty input.
    """
    if not records:
        return None
    n = len(records)
    correct = sum(1 for r in records if is_correct(r))
    p = correct / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    # max(0, ...) guards against numerical jitter where lower > p or upper < p.
    return p, max(0.0, p - lower), max(0.0, upper - p)


def normalized_error(record: Dict[str, Any]) -> Optional[str]:
    """Map the per-record error_type into one of the canonical buckets,
    or None if the record was correct (no error)."""
    et = record.get("error_type")
    if et in (None, "correct"):
        return None
    return ERROR_NORMALIZE.get(et)


def n_preferences(record: Dict[str, Any]) -> Optional[int]:
    prefs = record.get("preferences")
    if isinstance(prefs, list):
        return len(prefs)
    return None


def group_by_k(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        out[r["_k"]].append(r)
    return dict(sorted(out.items()))


def available_ks(dataset: str, memory: str) -> List[int]:
    return sorted({r["_k"] for r in load_analysis(dataset, memory)})
