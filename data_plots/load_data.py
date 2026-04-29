"""Shared loader for analysis JSONs across datasets/memory-systems/k values."""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYGROUND = PROJECT_ROOT / "playground"

DATASETS = {
    "coexisting": PLAYGROUND / "coexisting_facts" / "results",
    "conditional": PLAYGROUND / "conditional_facts" / "results",
    "conditional_hard": PLAYGROUND / "conditional_facts" / "results",
    "persona_retrieval": PLAYGROUND / "custom_persona_retrieval" / "results",
}

MEMORY_SYSTEMS = ["mem0", "simplemem", "amem"]

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
    return "run_hard_" in os.path.basename(run_dir)


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


def load_analysis(dataset: str, memory: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return concatenated analysis records (per-question dicts) for the given
    dataset/memory combo. If k is given, filter to that num_memories.

    Each returned record is augmented with ``_k`` (the num_memories of its run),
    ``_dataset`` and ``_memory`` for convenience.
    """
    out: List[Dict[str, Any]] = []
    for run_dir in _list_runs(dataset):
        sub = Path(run_dir) / memory
        if not sub.is_dir():
            continue
        analysis = _latest_analysis(sub)
        if analysis is None:
            continue
        # Need num_memories from the corresponding graded_traces metadata.
        gt_files = sorted(sub.glob("graded_traces_*.json"))
        if not gt_files:
            continue
        with open(gt_files[-1], "r", encoding="utf-8") as f:
            meta = json.load(f).get("run_metadata", {})
        run_k = meta.get("num_memories")
        if k is not None and run_k != k:
            continue
        with open(analysis, "r", encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            r["_k"] = run_k
            r["_dataset"] = dataset
            r["_memory"] = memory
        out.extend(records)
    return out


def all_records(dataset: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mem in MEMORY_SYSTEMS:
        out.extend(load_analysis(dataset, mem))
    return out


def is_correct(record: Dict[str, Any]) -> bool:
    """Treat judge_result == 'correct' as success."""
    return record.get("judge_result") == "correct"


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
