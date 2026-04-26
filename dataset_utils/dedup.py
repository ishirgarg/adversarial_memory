"""
MinHash LSH deduplication for dataset generation pipelines.
"""

from typing import Callable, Dict, List

from datasketch import MinHash, MinHashLSH

NUM_PERM = 128
SHINGLE_SIZE = 3  # word trigrams


def _shingles(text: str) -> set:
    tokens = text.lower().split()
    if len(tokens) < SHINGLE_SIZE:
        return {text.lower()}
    return {" ".join(tokens[i : i + SHINGLE_SIZE]) for i in range(len(tokens) - SHINGLE_SIZE + 1)}


def _minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for s in _shingles(text):
        m.update(s.encode("utf-8"))
    return m


def deduplicate(
    rows: List[Dict],
    key: Callable[[Dict], str],
    threshold: float = 0.8,
) -> List[Dict]:
    """Remove near-duplicate rows using MinHash LSH. Keeps the first occurrence.

    Args:
        rows:      list of row dicts to deduplicate
        key:       function mapping a row to the text to compare
        threshold: Jaccard similarity threshold above which rows are considered duplicates

    Returns:
        Deduplicated list (subset of rows, preserving original order).
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    kept: List[Dict] = []
    for i, row in enumerate(rows):
        text = key(row)
        m = _minhash(text)
        if not lsh.query(m):
            lsh.insert(str(i), m)
            kept.append(row)
    return kept
