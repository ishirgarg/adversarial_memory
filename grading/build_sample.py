"""
build_sample.py — assemble a deterministic manual-grading sample for the
coexisting-facts LLM-judge.

Scans `playground/coexisting_facts/results/` for paired
`analysis_*.json` + `graded_traces_*.json` files (one analysis per
memory-system folder), flattens every (run, memory_system, trace_index)
into a candidate pool, then deterministically picks `--n` items using
`--seed`.

Each sample is fully self-contained (memories + retrieved + response +
judge verdicts inlined) so the grading UI does not need to re-read the
source files. The manifest written here is also the canonical source of
truth for downstream summary statistics.

Usage
-----
  uv run python grading/build_sample.py            # 50 samples, seed=42
  uv run python grading/build_sample.py --seed 7 --n 100
  uv run python grading/build_sample.py --force    # rebuild even if exists
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "playground" / "coexisting_facts" / "results"
MANIFEST_PATH = SCRIPT_DIR / "sample_manifest.json"

DEFAULT_SEED = 42
DEFAULT_N = 50


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_pairs(results_root: Path) -> List[Dict[str, Any]]:
    """Find every (analysis, graded_traces) pair under results_root.

    Each memory-system subdirectory under each run is expected to contain
    at most one analysis_*.json and one graded_traces_*.json. If multiple
    of either exist we keep the most recent.
    """
    pairs: List[Dict[str, Any]] = []
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for sys_dir in sorted(run_dir.iterdir()):
            if not sys_dir.is_dir():
                continue
            analyses = sorted(sys_dir.glob("analysis_*.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
            graded = sorted(sys_dir.glob("graded_traces_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
            if not analyses or not graded:
                continue
            pairs.append({
                "run": run_dir.name,
                "memory_system": sys_dir.name,
                "analysis_path": str(analyses[0].relative_to(PROJECT_ROOT)),
                "graded_traces_path": str(graded[0].relative_to(PROJECT_ROOT)),
            })
    return pairs


# ---------------------------------------------------------------------------
# Inlining
# ---------------------------------------------------------------------------

def _index_analysis_by_conv_id(analysis: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        a["conversation_id"]: a
        for a in analysis
        if isinstance(a, dict) and a.get("conversation_id")
    }


def build_candidate_pool(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten every trace into a fully-inlined sample dict.

    Skips traces where the judge analysis errored out (analysis_error set)
    so manual grading does not get blocked on missing verdicts.
    """
    pool: List[Dict[str, Any]] = []
    for pair in pairs:
        analysis_path = PROJECT_ROOT / pair["analysis_path"]
        graded_path = PROJECT_ROOT / pair["graded_traces_path"]
        with open(analysis_path, encoding="utf-8") as f:
            analysis = json.load(f)
        with open(graded_path, encoding="utf-8") as f:
            graded = json.load(f)

        analysis_by_id = _index_analysis_by_conv_id(analysis)
        all_memories = graded.get("all_memories_at_time_of_questions", [])

        for trace_idx, trace in enumerate(graded.get("graded_traces", [])):
            conv_id = trace.get("conversation_id")
            judge = analysis_by_id.get(conv_id)
            if judge is None:
                continue
            if judge.get("analysis_error"):
                continue

            sample_id = f"{pair['memory_system']}__{pair['run']}__{trace_idx}"
            pool.append({
                "id": sample_id,
                "memory_system": pair["memory_system"],
                "run": pair["run"],
                "conversation_id": conv_id,
                "trace_index": trace_idx,
                "analysis_path": pair["analysis_path"],
                "graded_traces_path": pair["graded_traces_path"],
                "all_memories": all_memories,
                "preference_category": trace.get("preference_category", ""),
                "preferences": trace.get("preferences", []),
                "preference_facts": trace.get("preference_facts", []),
                "ground_truth_answer": trace.get("ground_truth_answer", ""),
                "question": trace.get("question", ""),
                "retrieved_memories": trace.get("retrieved_memories", ""),
                "llm_response": trace.get("llm_response", ""),
                "judge": {
                    "per_fact_results": judge.get("per_fact_results", []),
                    "error_type": judge.get("error_type"),
                    "judge_result": judge.get("judge_result"),
                    "correctly_invoked": judge.get("correctly_invoked"),
                    "invocation_reasoning": judge.get("invocation_reasoning"),
                },
            })
    return pool


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def deterministic_sample(pool: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n > len(pool):
        raise ValueError(
            f"Requested {n} samples but pool only has {len(pool)} items."
        )
    pool_sorted = sorted(pool, key=lambda s: s["id"])
    rng = random.Random(seed)
    indices = list(range(len(pool_sorted)))
    rng.shuffle(indices)
    return [pool_sorted[i] for i in indices[:n]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Sampling seed (default: {DEFAULT_SEED}).")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of samples (default: {DEFAULT_N}).")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT,
                        help="Path to coexisting-facts results directory.")
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH,
                        help="Output manifest path.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the manifest already exists.")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Manifest already exists at {args.output}. "
              f"Pass --force to rebuild.")
        sys.exit(0)

    pairs = discover_pairs(args.results_root)
    if not pairs:
        sys.exit(f"No (analysis, graded_traces) pairs found under {args.results_root}.")
    print(f"Found {len(pairs)} (analysis, graded_traces) pairs:")
    for p in pairs:
        print(f"  - {p['memory_system']:>10}  {p['run']}")

    pool = build_candidate_pool(pairs)
    print(f"Candidate pool: {len(pool)} traces (after dropping analysis_error rows).")

    samples = deterministic_sample(pool, args.n, args.seed)
    print(f"Selected {len(samples)} samples with seed={args.seed}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "n_samples": len(samples),
        "pool_size": len(pool),
        "source_pairs": pairs,
        "samples": samples,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {args.output}")

    by_system: Dict[str, int] = {}
    for s in samples:
        by_system[s["memory_system"]] = by_system.get(s["memory_system"], 0) + 1
    print("Sample distribution by memory system:")
    for ms, c in sorted(by_system.items()):
        print(f"  {ms:>10}  {c}")


if __name__ == "__main__":
    main()
