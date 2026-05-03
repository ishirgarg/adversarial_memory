"""
grading_ui.py — Streamlit UI for manually grading the coexisting-facts judge.

Each sample shows the question, ground-truth answer, retrieved memories,
LLM response, and (per-fact) the judge's verdict alongside an editable
"your verdict" control. The full memory store is searchable and ranked
by relevance to whichever fact you are currently focusing on.

Per-sample grades are written incrementally to `grading/grades.json`
so closing the tab never loses progress. A sidebar button writes a
summary of agreement statistics to `grading/summary.json`.

Usage
-----
  uv run python grading/build_sample.py            # one-time, builds manifest
  uv run streamlit run grading/grading_ui.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_PATH = SCRIPT_DIR / "sample_manifest.json"
GRADES_PATH = SCRIPT_DIR / "grades.json"
SUMMARY_PATH = SCRIPT_DIR / "summary.json"

FACT_CATEGORIES = ["not_stored", "summary_error", "not_retrieved", "reasoning_error", "correct"]
ERROR_TYPE_PRIORITY = ["not_stored", "summary_error", "not_retrieved", "reasoning_error"]


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_manifest() -> Optional[dict]:
    if not MANIFEST_PATH.exists():
        return None
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_grades() -> Dict[str, Any]:
    if not GRADES_PATH.exists():
        return {}
    with open(GRADES_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_grades(grades: Dict[str, Any]) -> None:
    with open(GRADES_PATH, "w", encoding="utf-8") as f:
        json.dump(grades, f, indent=2)


# ---------------------------------------------------------------------------
# Memory utilities
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


def _normalize_memory(mem: Any) -> str:
    if isinstance(mem, str):
        return mem
    if isinstance(mem, dict):
        return mem.get("memory") or mem.get("text") or json.dumps(mem)
    return str(mem)


def relevance_score(memory: str, query_terms: List[str], boost_terms: List[str]) -> float:
    """Substring + token-frequency hybrid.

    Boost terms (the preference name itself) are weighted more so memories
    that mention the specific preference float to the top even if they
    omit other tokens from the original fact sentence.
    """
    if not memory:
        return 0.0
    mem_lower = memory.lower()
    score = 0.0
    for term in boost_terms:
        t = term.lower().strip()
        if t and t in mem_lower:
            score += 5.0 * mem_lower.count(t)
    for term in query_terms:
        t = term.lower().strip()
        if not t or len(t) < 2:
            continue
        score += mem_lower.count(t)
    return score


def rank_memories(
    memories: List[Any],
    preference: str,
    original_fact: str,
    keyword_filter: str = "",
) -> List[Tuple[int, str, float]]:
    """Return [(orig_index, memory_text, score), ...] sorted by score desc.

    If `keyword_filter` is non-empty, results are filtered to memories
    containing that substring (case-insensitive) before ranking.
    """
    boost_terms = [preference] if preference else []
    query_terms = _tokens(f"{preference} {original_fact}")

    out: List[Tuple[int, str, float]] = []
    kw = keyword_filter.lower().strip()
    for i, raw in enumerate(memories):
        text = _normalize_memory(raw)
        if kw and kw not in text.lower():
            continue
        score = relevance_score(text, query_terms, boost_terms)
        out.append((i, text, score))
    out.sort(key=lambda t: (-t[2], t[0]))
    return out


# ---------------------------------------------------------------------------
# Verdict derivation
# ---------------------------------------------------------------------------

def collapse_error_type(per_fact_categories: List[str]) -> str:
    cats = set(per_fact_categories)
    for p in ERROR_TYPE_PRIORITY:
        if p in cats:
            return p
    return "correct"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(manifest: dict, grades: Dict[str, Any]) -> dict:
    samples = manifest["samples"]
    n_total = len(samples)

    fact_total = 0
    fact_agree = 0
    fact_confusion: Dict[str, Dict[str, int]] = {
        c: {c2: 0 for c2 in FACT_CATEGORIES} for c in FACT_CATEGORIES
    }

    trace_total = 0
    trace_agree_error_type = 0
    trace_agree_judge_result = 0
    error_type_confusion: Dict[str, Dict[str, int]] = {}

    by_system: Dict[str, Dict[str, int]] = {}

    per_sample_log: List[Dict[str, Any]] = []

    for sample in samples:
        sid = sample["id"]
        if sid not in grades:
            continue
        g = grades[sid]
        if not g.get("complete"):
            continue

        judge = sample["judge"]
        judge_per_fact = [f.get("category") for f in judge["per_fact_results"]]
        user_per_fact = g["user_per_fact_categories"]

        sample_fact_agree = 0
        for i, judge_cat in enumerate(judge_per_fact):
            user_cat = user_per_fact.get(str(i)) or user_per_fact.get(i)
            if user_cat is None:
                continue
            fact_total += 1
            if user_cat == judge_cat:
                fact_agree += 1
                sample_fact_agree += 1
            if judge_cat in fact_confusion and user_cat in fact_confusion[judge_cat]:
                fact_confusion[judge_cat][user_cat] += 1

        trace_total += 1
        judge_error_type = judge["error_type"]
        user_error_type = g["user_error_type"]
        if user_error_type == judge_error_type:
            trace_agree_error_type += 1
        error_type_confusion.setdefault(judge_error_type or "?", {})
        error_type_confusion[judge_error_type or "?"][user_error_type or "?"] = (
            error_type_confusion[judge_error_type or "?"].get(user_error_type or "?", 0) + 1
        )

        judge_result = judge["judge_result"]
        user_result = "correct" if user_error_type == "correct" else "incorrect"
        if judge_result == user_result:
            trace_agree_judge_result += 1

        ms = sample["memory_system"]
        bs = by_system.setdefault(ms, {"trace_total": 0, "trace_agree_error_type": 0,
                                       "fact_total": 0, "fact_agree": 0})
        bs["trace_total"] += 1
        bs["trace_agree_error_type"] += int(user_error_type == judge_error_type)
        bs["fact_total"] += len(judge_per_fact)
        bs["fact_agree"] += sample_fact_agree

        per_sample_log.append({
            "id": sid,
            "memory_system": ms,
            "run": sample["run"],
            "preferences": sample["preferences"],
            "judge_per_fact_categories": judge_per_fact,
            "user_per_fact_categories": [user_per_fact.get(str(i)) for i in range(len(judge_per_fact))],
            "judge_error_type": judge_error_type,
            "user_error_type": user_error_type,
            "judge_correctly_invoked": judge["correctly_invoked"],
            "user_notes": g.get("user_notes", ""),
            "fact_agreement": f"{sample_fact_agree}/{len(judge_per_fact)}",
            "trace_agreement": user_error_type == judge_error_type,
        })

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_seed": manifest["seed"],
        "manifest_n": n_total,
        "n_graded": trace_total,
        "fact_level": {
            "total": fact_total,
            "agreement": fact_agree,
            "agreement_rate": (fact_agree / fact_total) if fact_total else None,
            "confusion_matrix_judge_to_user": fact_confusion,
        },
        "trace_level": {
            "total": trace_total,
            "error_type_agreement": trace_agree_error_type,
            "error_type_agreement_rate": (trace_agree_error_type / trace_total) if trace_total else None,
            "judge_result_agreement": trace_agree_judge_result,
            "judge_result_agreement_rate": (trace_agree_judge_result / trace_total) if trace_total else None,
            "error_type_confusion_matrix_judge_to_user": error_type_confusion,
        },
        "by_memory_system": {
            ms: {
                **v,
                "trace_agreement_rate": (v["trace_agree_error_type"] / v["trace_total"]) if v["trace_total"] else None,
                "fact_agreement_rate": (v["fact_agree"] / v["fact_total"]) if v["fact_total"] else None,
            }
            for ms, v in by_system.items()
        },
        "per_sample_log": per_sample_log,
    }


def write_summary(manifest: dict, grades: Dict[str, Any]) -> dict:
    summary = compute_summary(manifest, grades)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def init_state(manifest: dict) -> None:
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "grades" not in st.session_state:
        st.session_state.grades = load_grades()
    if "focus_fact_idx" not in st.session_state:
        st.session_state.focus_fact_idx = 0
    if "memory_filter" not in st.session_state:
        st.session_state.memory_filter = ""
    if "show_all_memories" not in st.session_state:
        st.session_state.show_all_memories = False


def sample_grade_state(sample: dict) -> dict:
    """Return the grade dict for a sample, creating defaults from the judge.

    Pre-fill per-fact = judge's per-fact category, but if the judge's overall
    verdict is `reasoning_error` (meaning all judge per-facts are `correct`
    yet the model failed to invoke them), promote those `correct` defaults to
    `reasoning_error` so the user's default trace verdict aligns with the
    judge's. The user can flip individual facts back to `correct` if they
    think a specific preference *was* invoked.
    """
    sid = sample["id"]
    grades = st.session_state.grades
    if sid not in grades:
        judge_per_fact = sample["judge"]["per_fact_results"]
        judge_error_type = sample["judge"]["error_type"]
        prefill = {}
        for i, f in enumerate(judge_per_fact):
            cat = f.get("category")
            if judge_error_type == "reasoning_error" and cat == "correct":
                cat = "reasoning_error"
            prefill[str(i)] = cat
        grades[sid] = {
            "user_per_fact_categories": prefill,
            "user_error_type": collapse_error_type(list(prefill.values())),
            "user_notes": "",
            "complete": False,
            "graded_at": None,
        }
    return grades[sid]


def render_sidebar(manifest: dict) -> None:
    st.sidebar.header("Manual grading")
    samples = manifest["samples"]
    n_total = len(samples)
    grades = st.session_state.grades
    n_complete = sum(1 for s in samples if grades.get(s["id"], {}).get("complete"))
    st.sidebar.metric("Progress", f"{n_complete} / {n_total}")
    st.sidebar.progress(n_complete / n_total if n_total else 0.0)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Manifest seed: **{manifest['seed']}**  ·  "
        f"Pool: {manifest.get('pool_size', '?')} traces"
    )

    jump = st.sidebar.number_input(
        "Jump to sample #", min_value=1, max_value=n_total,
        value=st.session_state.current_idx + 1, step=1,
    )
    if jump - 1 != st.session_state.current_idx:
        st.session_state.current_idx = int(jump) - 1
        st.session_state.focus_fact_idx = 0
        st.session_state.memory_filter = ""
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick nav")
    filter_choice = st.sidebar.radio(
        "Show",
        ["All", "Ungraded", "Disagreements only"],
        horizontal=False,
        key="quick_nav_filter",
    )
    nav_indices = list(range(n_total))
    if filter_choice == "Ungraded":
        nav_indices = [i for i in nav_indices if not grades.get(samples[i]["id"], {}).get("complete")]
    elif filter_choice == "Disagreements only":
        nav_indices = [
            i for i in nav_indices
            if grades.get(samples[i]["id"], {}).get("complete")
            and grades[samples[i]["id"]].get("user_error_type")
                != samples[i]["judge"]["error_type"]
        ]

    if nav_indices:
        labels = []
        for i in nav_indices:
            s = samples[i]
            g = grades.get(s["id"], {})
            mark = "✓" if g.get("complete") else " "
            disagree = ""
            if g.get("complete") and g.get("user_error_type") != s["judge"]["error_type"]:
                disagree = "⚠"
            labels.append(f"{mark}{disagree} #{i+1} {s['memory_system']} — {s['preference_category'][:25]}")
        try:
            current_label_idx = nav_indices.index(st.session_state.current_idx)
        except ValueError:
            current_label_idx = 0
        chosen = st.sidebar.radio(
            "Sample list", labels, index=current_label_idx, label_visibility="collapsed",
        )
        chosen_i = nav_indices[labels.index(chosen)]
        if chosen_i != st.session_state.current_idx:
            st.session_state.current_idx = chosen_i
            st.session_state.focus_fact_idx = 0
            st.session_state.memory_filter = ""
            st.rerun()
    else:
        st.sidebar.caption("(no samples match this filter)")

    st.sidebar.markdown("---")
    if st.sidebar.button("Compute summary", use_container_width=True):
        summary = write_summary(manifest, grades)
        st.session_state.last_summary = summary
        st.sidebar.success(f"Wrote {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")

    if "last_summary" in st.session_state:
        s = st.session_state.last_summary
        st.sidebar.markdown("**Latest summary**")
        if s["fact_level"]["agreement_rate"] is not None:
            st.sidebar.write(
                f"Fact agreement: {s['fact_level']['agreement_rate']:.1%} "
                f"({s['fact_level']['agreement']}/{s['fact_level']['total']})"
            )
        if s["trace_level"]["error_type_agreement_rate"] is not None:
            st.sidebar.write(
                f"Trace error-type agreement: "
                f"{s['trace_level']['error_type_agreement_rate']:.1%} "
                f"({s['trace_level']['error_type_agreement']}/{s['trace_level']['total']})"
            )


def render_memory_panel(sample: dict) -> None:
    st.subheader("Memory store")
    preferences = sample["preferences"]
    facts = sample["preference_facts"]
    n = len(preferences)

    cols = st.columns([2, 3])
    with cols[0]:
        focus_options = [f"#{i+1}: {preferences[i]}" for i in range(n)]
        focus_options.append("(no focus — original order)")
        focus_idx = st.session_state.focus_fact_idx
        if focus_idx > n:
            focus_idx = 0
        choice = st.radio(
            "Rank memories by relevance to:",
            focus_options,
            index=focus_idx if focus_idx < len(focus_options) else 0,
            key=f"focus_radio_{sample['id']}",
        )
        new_focus = focus_options.index(choice)
        if new_focus != st.session_state.focus_fact_idx:
            st.session_state.focus_fact_idx = new_focus
    with cols[1]:
        st.session_state.memory_filter = st.text_input(
            "Keyword filter (substring, case-insensitive)",
            value=st.session_state.memory_filter,
            key=f"mem_filter_{sample['id']}",
            placeholder="e.g. 'Hatha' or 'sushi'",
        )
        st.session_state.show_all_memories = st.checkbox(
            "Show all memories (default top-25)",
            value=st.session_state.show_all_memories,
            key=f"show_all_{sample['id']}",
        )

    focus_idx = st.session_state.focus_fact_idx
    if focus_idx < n:
        ranked = rank_memories(
            sample["all_memories"],
            preferences[focus_idx],
            facts[focus_idx] if focus_idx < len(facts) else "",
            st.session_state.memory_filter,
        )
    else:
        ranked = rank_memories(
            sample["all_memories"], "", "", st.session_state.memory_filter,
        )
        ranked.sort(key=lambda t: t[0])

    total = len(sample["all_memories"])
    matched = len(ranked)
    st.caption(
        f"{matched} of {total} memories "
        f"{'after filter' if st.session_state.memory_filter else 'in store'}"
    )

    cap = matched if st.session_state.show_all_memories else min(25, matched)
    for orig_idx, text, score in ranked[:cap]:
        score_str = f"score={score:.0f}" if score > 0 else "score=0"
        st.markdown(f"**[{orig_idx}]** ({score_str})  \n{text}")
    if matched > cap:
        st.caption(f"… {matched - cap} more hidden. Tick 'Show all memories' to expand.")


def render_per_fact_grading(sample: dict, grade: dict) -> None:
    st.subheader("Per-fact grading")
    judge_per_fact = sample["judge"]["per_fact_results"]
    preferences = sample["preferences"]
    facts = sample["preference_facts"]

    for i, jf in enumerate(judge_per_fact):
        pref = preferences[i] if i < len(preferences) else ""
        fact = facts[i] if i < len(facts) else ""
        judge_cat = jf.get("category")
        with st.container(border=True):
            head_cols = st.columns([3, 2])
            with head_cols[0]:
                st.markdown(f"**Fact #{i+1} — `{pref}`**")
                st.caption(fact)
            with head_cols[1]:
                if st.button(f"Focus memory ranking on this fact",
                             key=f"focus_btn_{sample['id']}_{i}",
                             use_container_width=True):
                    st.session_state.focus_fact_idx = i
                    st.rerun()

            judge_col, user_col = st.columns(2)
            with judge_col:
                st.markdown(f"**Judge:** `{judge_cat}`")
                if sample["judge"]["error_type"] == "reasoning_error":
                    st.caption(
                        "ℹ Judge classified this fact as `correct` per-fact "
                        "but the trace overall as `reasoning_error` "
                        "(model failed to invoke retrieved preferences)."
                    )
                with st.expander("Judge reasoning"):
                    st.write(f"**Storage check** "
                             f"(`fact_in_store`={jf.get('fact_in_store')}):")
                    st.caption(jf.get("storage_reasoning") or "—")
                    st.write(f"**Summary check** "
                             f"(`summary_passed`={jf.get('summary_passed')}):")
                    st.caption(jf.get("summary_reasoning") or "—")
                    st.write(f"**Retrieval check** "
                             f"(`fact_in_retrieved`={jf.get('fact_in_retrieved')}):")
                    st.caption(jf.get("retrieval_reasoning") or "—")
                    inv_reasoning = sample["judge"].get("invocation_reasoning")
                    if inv_reasoning:
                        st.write(
                            f"**Invocation check (trace-level)** "
                            f"(`correctly_invoked={sample['judge']['correctly_invoked']}`):"
                        )
                        st.caption(inv_reasoning)
            with user_col:
                current = grade["user_per_fact_categories"].get(str(i), judge_cat)
                if current not in FACT_CATEGORIES:
                    current = judge_cat or FACT_CATEGORIES[0]
                user_choice = st.radio(
                    "Your verdict",
                    FACT_CATEGORIES,
                    index=FACT_CATEGORIES.index(current),
                    key=f"user_cat_{sample['id']}_{i}",
                    horizontal=False,
                )
                grade["user_per_fact_categories"][str(i)] = user_choice
                if user_choice == judge_cat:
                    st.success("agree", icon="✅")
                else:
                    st.warning(f"disagree (judge: `{judge_cat}`)", icon="⚠️")


def render_trace_verdict(sample: dict, grade: dict) -> None:
    user_cats = list(grade["user_per_fact_categories"].values())
    grade["user_error_type"] = collapse_error_type(user_cats)

    judge_et = sample["judge"]["error_type"]
    user_et = grade["user_error_type"]
    if user_et == judge_et:
        st.success(f"Overall verdict: `{user_et}` — matches judge", icon="✅")
    else:
        st.warning(
            f"Overall verdict: yours=`{user_et}` vs judge=`{judge_et}`",
            icon="⚠️",
        )


def render_main_pane(manifest: dict) -> None:
    samples = manifest["samples"]
    idx = st.session_state.current_idx
    sample = samples[idx]
    grade = sample_grade_state(sample)

    top = st.columns([3, 2, 2])
    with top[0]:
        st.markdown(f"### Sample {idx + 1} / {len(samples)}")
        st.caption(
            f"`{sample['memory_system']}` · `{sample['run']}` · "
            f"category=*{sample['preference_category']}* · "
            f"id=`{sample['id']}`"
        )
    with top[1]:
        st.markdown("**Judge verdict**")
        st.code(
            f"error_type    = {sample['judge']['error_type']}\n"
            f"judge_result  = {sample['judge']['judge_result']}\n"
            f"correctly_inv = {sample['judge']['correctly_invoked']}",
            language=None,
        )
    with top[2]:
        nav = st.columns(2)
        if nav[0].button("◀ Prev", use_container_width=True, disabled=idx == 0):
            st.session_state.current_idx = idx - 1
            st.session_state.focus_fact_idx = 0
            st.session_state.memory_filter = ""
            st.rerun()
        if nav[1].button("Next ▶", use_container_width=True,
                         disabled=idx == len(samples) - 1):
            st.session_state.current_idx = idx + 1
            st.session_state.focus_fact_idx = 0
            st.session_state.memory_filter = ""
            st.rerun()

    qa, gt = st.columns([3, 2])
    with qa:
        st.markdown("**Question**")
        st.info(sample["question"])
    with gt:
        st.markdown("**Ground truth answer**")
        st.success(sample["ground_truth_answer"])

    pref_str = "\n".join(
        f"{i+1}. **{p}** — *{sample['preference_facts'][i] if i < len(sample['preference_facts']) else ''}*"
        for i, p in enumerate(sample["preferences"])
    )
    st.markdown("**Coexisting preferences (ground truth):**")
    st.markdown(pref_str)

    body_cols = st.columns(2, gap="large")
    with body_cols[0]:
        st.subheader("Retrieved memories (shown to model)")
        retrieved = sample["retrieved_memories"] or "(none)"
        st.code(retrieved if isinstance(retrieved, str) else str(retrieved), language=None)

        st.subheader("LLM response")
        st.markdown(
            f"<div style='background:#f3f4f6;padding:12px;border-radius:6px;"
            f"white-space:pre-wrap;color:#111;'>{sample['llm_response']}</div>",
            unsafe_allow_html=True,
        )

    with body_cols[1]:
        render_memory_panel(sample)

    st.markdown("---")
    render_per_fact_grading(sample, grade)
    render_trace_verdict(sample, grade)

    grade["user_notes"] = st.text_area(
        "Notes (optional)",
        value=grade.get("user_notes", ""),
        key=f"notes_{sample['id']}",
        placeholder="Anything noteworthy about this sample…",
    )

    save_cols = st.columns([1, 1, 3])
    if save_cols[0].button(
        "✅ Mark complete & save", use_container_width=True, type="primary",
        key=f"save_{sample['id']}",
    ):
        grade["complete"] = True
        grade["graded_at"] = datetime.now().isoformat(timespec="seconds")
        save_grades(st.session_state.grades)
        st.toast("Saved.", icon="💾")
        if idx < len(samples) - 1:
            st.session_state.current_idx = idx + 1
            st.session_state.focus_fact_idx = 0
            st.session_state.memory_filter = ""
            st.rerun()
    if save_cols[1].button(
        "💾 Save (without marking complete)", use_container_width=True,
        key=f"save_partial_{sample['id']}",
    ):
        save_grades(st.session_state.grades)
        st.toast("Saved.", icon="💾")
    save_cols[2].caption(
        f"Status: {'complete' if grade.get('complete') else 'in progress'}"
        + (f" · last graded {grade['graded_at']}" if grade.get("graded_at") else "")
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Coexisting-facts grading", layout="wide",
        initial_sidebar_state="expanded",
    )
    manifest = load_manifest()
    if manifest is None:
        st.error(
            "No sample manifest found. Run "
            "`uv run python grading/build_sample.py` first."
        )
        st.stop()

    init_state(manifest)
    render_sidebar(manifest)
    render_main_pane(manifest)


if __name__ == "__main__":
    main()
