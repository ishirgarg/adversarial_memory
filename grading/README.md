# Manual grading for the coexisting-facts judge

A small Streamlit app for spot-checking the LLM-judge in
`playground/coexisting_facts/analyze_errors.py`. You manually classify
each preference fact in a sample of traces, the app records what you
chose alongside what the judge chose, and a summary report tells you
how aligned you and the judge are.

## Workflow

```bash
# 1. Build a deterministic 50-sample manifest (seed=42 by default).
uv run python grading/build_sample.py

# 2. Grade interactively in the browser. Progress is auto-saved.
uv run streamlit run grading/grading_ui.py

# 3. Re-run the summary anytime (UI also has a "Compute summary" button).
uv run python grading/summary.py
```

## Files

- `build_sample.py` — scans `playground/coexisting_facts/results/` for
  `analysis_*.json` + `graded_traces_*.json` pairs, flattens every trace
  into a candidate pool, and writes a seeded `n=50` sample to
  `sample_manifest.json`. Each sample inlines all data needed to
  grade it (memories, retrieved, response, judge verdicts).
- `grading_ui.py` — Streamlit UI. For each sample shows:
  - the question, ground-truth answer, retrieved memories, LLM response;
  - the full memory store, ranked by relevance to whichever fact you are
    focusing on (toggle via the radio next to each fact), with a keyword
    filter and a "show all" expander;
  - per-fact judge verdicts side-by-side with editable "your verdict"
    radios — defaults to the judge's choice so you only flip the ones
    you disagree with;
  - an invocation question shown only when every fact is graded
    `correct` (so a `reasoning_error` is the only path left).
  Grades are written to `grades.json` after every save.
- `summary.py` — recomputes alignment metrics and writes
  `summary.json`. Same logic as the UI's "Compute summary" button.

## Output (`summary.json`)

- `fact_level.agreement_rate` — fraction of all individual fact verdicts
  where you and the judge agreed.
- `trace_level.error_type_agreement_rate` — fraction of traces where the
  collapsed error-type matched.
- `trace_level.judge_result_agreement_rate` — fraction agreeing on
  `correct` vs `incorrect` (less strict than error-type agreement).
- Confusion matrices for both fact-level and trace-level (rows=judge
  verdict, cols=your verdict).
- Per-memory-system breakdown.
- `per_sample_log` — the raw judge-vs-user table.

## Reproducibility

Sampling is seeded. `sample_manifest.json` is the canonical record
of which 50 traces were chosen — keep it under version control if you
care about reproducing the exact grading set. `grades.json` is your
work product; commit it once you're done. `summary.json` is
derived and can always be regenerated from those two.
