# Decisions

> Rule: Every meaningful AI-assisted decision must be appended to this file
> before or immediately after implementation.
> Last updated: 2026-08-11

## Use dense embedding search (`dense_search`) as the production retrieval method

Date: 2026-08-11 (per timestamps in `README_module1.md` and session
history)
Status: accepted

Context:
The project needed a retrieval method for a document-QA pipeline over 5
SEC 10-K filings. Multiple retrieval strategies were built and evaluated:
pure dense embedding search, BM25 sparse search, RRF hybrid fusion of the
two, and cross-encoder reranking on top of the hybrid results, both with
and without a query-preprocessing (stopword removal) step.

Decision:
`dense_search` (from `stage2_embed.py`, using `BAAI/bge-small-en-v1.5`
embeddings over a persistent ChromaDB collection) is the retrieval function
used for Module 2 onward.

Alternatives:
- `hybrid_search` (BM25 + dense via Reciprocal Rank Fusion, k=60)
- `hybrid_search` with query preprocessing (`hybrid_preprocessed`)
- `hybrid_rerank_search` (hybrid candidates reranked by a cross-encoder)
- `hybrid_rerank_search` with query preprocessing
  (`hybrid_rerank_preprocessed`)

Reasoning:
Across a 28-query hand-written eval set (`eval_set.json`), dense-only
retrieval achieved hit_rate@5 = 1.000 and precision@1 = 0.893, outperforming
every hybrid variant on both metrics (hybrid: 0.964 / 0.679; hybrid
preprocessed: 0.964 / 0.679; hybrid+rerank: 0.964 / 0.821; hybrid+rerank
preprocessed: 0.964 / 0.786). The result was confirmed stable via 3
back-to-back reruns of the evaluation with zero variance in the numbers.
A follow-up experiment testing whether BM25-side query preprocessing
(stopword removal) closed the gap found no improvement. The likely
root cause, per `README_module1.md`, is that 10-K filings share heavy
boilerplate language across companies (SEC cover-page structure, standard
risk-factor phrasing), which makes keyword-based BM25 matching less
discriminative than semantic dense matching for this specific corpus.

Consequences:
- Simpler production path: no BM25 index build or cross-encoder inference
  pass required at query time, which is also faster.
- `stage3_hybrid.py` and `stage4_rerank.py` remain in the repository as
  reference/comparison code rather than being deleted, per explicit
  instruction recorded in `README_module1.md`. Future contributors must not
  assume they are dead code to remove — see `CONSTRAINTS.md`.
- Any future corpus with less boilerplate overlap (e.g., diverse document
  types rather than same-template 10-Ks) would need this decision
  re-evaluated rather than assumed to generalize.

Evidence:
- `README_module1.md` (root) — full pipeline description, final evaluation
  table, Findings section, and explicit Decision section.
- `evaluate_retrieval.py` output across multiple session runs (28-query
  eval set, 5 methods compared).
- `stage2_embed.py`, `stage3_hybrid.py`, `stage4_rerank.py` — the
  implementations of the compared methods.

---

## Retain `stage3_hybrid.py` and `stage4_rerank.py` in the repo despite not being production code

Date: 2026-08-11
Status: accepted

Context:
Once dense-only retrieval was chosen as production (see above decision),
the hybrid and reranking scripts were no longer part of the active
pipeline.

Decision:
Keep both files in the repository, with a top-of-file comment noting they
are retained for reference/comparison and are not used in the production
pipeline going forward.

Alternatives:
- Delete the files entirely.
- Move them to an `archive/` or `experiments/` subfolder.

Reasoning:
The comparison data they produced (hybrid vs. dense vs. reranked) is the
evidentiary basis for the dense-only decision itself. Removing the code
would make that evidence harder to reproduce or re-verify later, and both
files are still imported by `evaluate_retrieval.py` and
`debug_zoom_currency.py` for ongoing comparison/debugging purposes.

Consequences:
- The repo carries two retrieval implementations that are not on the
  "hot path" of any future application code — a maintenance/clarity cost
  that must be weighed if the project grows.
- Anyone extending `evaluate_retrieval.py` needs to know these imports are
  intentional, not leftover cruft (see `FLOW.md`'s File-to-File Call Map
  and Risks section).

Evidence:
- Top-of-file comments in `stage3_hybrid.py` and `stage4_rerank.py`.
- `README_module1.md` Decision section.
- `evaluate_retrieval.py` and `debug_zoom_currency.py` both still import
  from these files as of this session.

---

## Retrieval-method reversal, then revert: dense_search remains production

Date: 2026-08-12
Status: accepted (net result: no change from the original decision above,
but arrived at via a real excursion — see Reasoning)

Context:
An LLM-as-judge re-evaluation of retrieval quality (grading retrieved
*content* relevance rather than source-file match) showed
`hybrid_rerank_search` outperforming `dense_search`, even after fixing a
cross-company leniency bug in the judge. This looked like strong evidence
to reverse the original dense-only decision above.

Decision:
`stage2_self_correct.py` was switched to `hybrid_rerank_search` as
production retrieval. The full 28-query calibration dataset was rebuilt
and conformal calibration re-run on top of it as an end-to-end check
before finalizing anything in `README_module1.md`.

Reasoning:
The end-to-end check failed the switch: `llm_judge_correct` dropped from
19/28 (dense) to 18/28 (hybrid_rerank), and conformal empirical coverage
collapsed from 87.5% to 44.4% against an 80% target — the red-team audit's
`passed` label became far less correlated with actual correctness once
retrieval changed, even though the retrieved *chunks* were individually
more content-relevant in isolation. The switch was reverted:
`stage2_self_correct.py` and `stage3_redteam.py` went back to
`dense_search`, and the calibration dataset / conformal calibration were
rebuilt again to confirm recovery (~20/28 correct, 83.3% coverage).

Consequences:
- **Net production state is unchanged** — `dense_search` remains the
  retrieval method for Module 2 onward, same as the original decision.
- **Methodological lesson, now documented in README.md's "Known
  Limitations & Lessons Learned":** a component-level metric (chunk
  relevance) is not a reliable predictor of end-to-end system quality.
  Any future retrieval change must be validated with a full pipeline run
  (final-answer correctness + calibration coverage), not just a retrieval
  eval, before being adopted.
- `stage4_conformal.py`'s `get_retrieval_chunks` always re-runs
  `dense_search` for its similarity/spread signal regardless of which
  method actually answered — this asymmetry was a pre-existing, deliberate
  simplification, not something introduced or fixed by this episode. It is
  flagged as a known gap in `README_module4.md`'s Future Work, since it
  would silently go stale if production retrieval changes again in the
  future.

Evidence:
- `README_module1.md`'s "Decision Reversal Investigation (and Revert)"
  section — full table of lenient/strict judge results and the end-to-end
  before/after numbers.
- `README_module4.md`'s Final Results and Future Work sections — updated
  post-revert numbers (threshold 0.9796, 83.3% coverage) and the 44.4%
  coverage collapse observed mid-episode.

---

## Repository reorganization into src/eval/reports/experiments

Date: 2026-08-12
Status: accepted

Context:
All Python scripts and data files lived flat at the repo root. Ahead of a
GitHub push, the repo needed a conventional, browsable layout.

Decision:
Moved production pipeline scripts into `src/`, eval/calibration data into
`eval/`, sample generated reports into `reports/`, and retired/debug
scripts (previously `debug_zoom_currency.py`, `spot_check_grading.py`,
`stage1_ingest_v2.py`, `stage2_embed_v2.py`, `evaluate_v2.py`, and all
`scratch_*.py` files) into `experiments/` — kept, not deleted, since they
are evidentiary record of the investigation process. `app.py` stays at
root and inserts `src/` onto `sys.path` before importing pipeline modules.
Every `Path(__file__).parent`-based constant and cross-module import was
updated to match the new layout; a smoke test (`src/stage_final_report.py`
generating all 3 example reports) confirmed everything still works
end-to-end post-move.

Alternatives:
- Leave the flat layout — rejected, doesn't read as a portfolio-ready repo.
- Delete the `scratch_*.py`/`*_v2.py` experiment files instead of moving
  them — rejected per explicit instruction; they document real
  investigation steps (contamination bug discovery, chunking-parameter
  experiments) referenced by README module writeups.

Consequences:
- See `ARCHITECTURE.md`'s Folder Map for the new layout.
- Anyone extending an `experiments/` script needs a `sys.path.insert(...,
  "../src")` shim to import production modules — already added to each
  file that needed it.
- `chunks.json`/`chunks_v2.json` remain gitignored (large, regenerable
  from tracked `documents/` PDFs); `documents/` itself is now tracked
  (previously gitignored) for reproducibility.

## Report a bootstrap confidence interval alongside the point-estimate coverage

Date: 2026-08-20
Status: accepted

Context:
Module 4's headline number — empirical coverage of the high-confidence
label — is computed on a small held-out test set (18 high-confidence cases
out of 24). Reporting it as a single point estimate ("83.3%") overstates
its precision: at this sample size the true coverage could plausibly be
anywhere in a wide band, and several of the project's own "Known
Limitations" notes already flag small-sample fragility as the #1 caveat.

Decision:
`stage4_conformal.py` now also prints a percentile bootstrap confidence
interval on empirical coverage (`bootstrap_coverage_ci`, default 90% CI,
10,000 resamples, seeded with `SPLIT_RANDOM_SEED` for reproducibility). The
point estimate is retained and unchanged; the CI is additive context. On
the current data this prints `83.3%  90% bootstrap CI [66.7%, 94.4%]`.

Alternatives:
- Report only the point estimate (status quo) — rejected: implies a
  precision the sample size does not support.
- Use an analytic (Wilson/Clopper-Pearson) binomial interval instead of a
  bootstrap — a reasonable alternative; bootstrap was chosen for its
  transparency (no distributional assumption, trivially re-derivable) and
  because `numpy` was already a dependency, requiring no new install.

Reasoning:
The whole thesis of the project is producing a *trustworthy* confidence
signal. A coverage number without an uncertainty band is exactly the kind
of overconfident point estimate the project exists to critique. The CI
makes the small-sample caveat quantitative and visible in the tool's own
output rather than only in prose in the README.

Consequences:
- No new dependency (numpy already present); no change to the point
  estimate, threshold, or any labeling logic — purely additive reporting.
- The interval will tighten as the eval/calibration set grows, giving a
  concrete, measurable payoff for the "grow the eval set" next step.

Evidence:
- `src/stage4_conformal.py` — `bootstrap_coverage_ci()` and its call in the
  Empirical Coverage section; verified running (deterministic, reproduces
  the 0.9820 threshold and 15/18 point estimate).

## Add a query-routing layer that scopes retrieval to the relevant document(s)

Date: 2026-08-21
Status: accepted

Context:
`answer_with_self_correction` inferred a single company via `detect_company`
and, when it found none, retrieved across ALL five filings — the
cross-company contamination hole flagged in README.md's Known Limitations
(a single-company question could silently pull chunks from other companies'
filings). There was also no mechanism to deliberately read across documents
for a genuinely comparative question; those were simply declined.

Decision:
Added `detect_companies` (all named companies), `is_comparative`, and
`route_query` to `stage2_self_correct.py`. `route_query` returns a scope:
- `single`      -> retrieval HARD-filtered to one filing; others never read.
- `comparative` -> fan out across the named filings (or all five if none
                   named) via `routed_dense_search`, `COMPARATIVE_PER_SOURCE_K`
                   chunks each, merged by similarity.
- `ambiguous`   -> no company signal; search all (unchanged legacy behavior).
`run_dense_retrieval_flow` now takes a `route` instead of a `filter_source`;
an explicit UI `filter_source` maps to a forced `single` route.

Alternatives:
- Keep declining all comparative queries (status quo) — rejected: the user
  explicitly asked for scoped cross-document lookup when, and only when, the
  question requires it.
- Use an LLM to classify scope — rejected as the primary path: the
  deterministic heuristic is testable without a model and covers the corpus's
  fixed five-company vocabulary; an LLM classifier can be added later for
  ambiguous cases if needed.

Reasoning:
The deterministic router makes the contamination guard verifiable (a `single`
route provably returns chunks from only its filing — covered by a live-index
test) while enabling multi-document retrieval strictly for comparative
queries. Single/ambiguous retrieval is behavior-preserving by construction, so
the validated Module 2-4 pipeline and its calibration numbers are unaffected.

Consequences:
- **Intended behavior change on comparative queries only.** The four
  `graceful_decline`-tagged comparative queries in `eval_set.json` now trigger
  multi-document *retrieval* rather than an automatic decline. This has NOT
  been re-validated end-to-end for answer quality — comparative *synthesis*
  (generation over multi-filing context) remains the known-hard, previously
  scoped-out problem; only the retrieval routing changed. Their expected
  behavior should be revisited before those entries are re-scored.
- New harder eval queries staged in `eval/eval_set_additions.json` exercise
  the router and other robustness dimensions (false premise, out-of-corpus,
  attribution traps, numeric extraction, injection resistance).
- `tests/test_pipeline.py` gains routing + retrieval-guard tests (26 tests
  total, all passing), including a live-index assertion that `single` scope
  never reads another company's document.

Evidence:
- `src/stage2_self_correct.py` (`route_query`, `routed_dense_search`) and
  `src/stage3_redteam.py` (updated caller).
- `tests/test_pipeline.py` — `TestQueryRouting`, `TestRetrievalGuard`.

## Ground comparative metadata questions on exact values, not retrieved text

Date: 2026-08-21
Status: accepted

Context:
End-to-end testing of the new comparative routing (above) showed the weakest
answers were on comparative questions about structured cover-page facts (e.g.
"which of the five companies does NOT end its fiscal year on December 31?").
Retrieval fan-out pulled noisy chunks — one answer conflated a "$42.8M as of
December 31" forward-contract line with a fiscal-year-end date — producing a
confidently muddled result.

Decision:
Added `comparative_metadata_answer` (+ `_metadata_fields_in_query`) to
`stage2_self_correct.py`. For a comparative query about a known metadata field
(fiscal year end, ticker, exchange, incorporation, address, EIN), the model is
grounded on the EXACT values for the in-scope companies from
`company_metadata.json` instead of retrieved chunk text. `answer_with_self_
correction` was reordered so comparative scope is handled by the routing layer
(grounded metadata path, else retrieval fan-out) and never by the
single-company metadata fast-path, whose downstream `metadata_sanity_check` is
single-company and had been wrongly rejecting correct comparative answers.
Single/ambiguous scope behavior is unchanged.

Reasoning / iteration (methods tried, measured against ground truth):
1. Retrieval fan-out only -> hallucinated (forward-contract confusion).
2. + Grounding on exact metadata -> hallucination eliminated; HubSpot correctly
   identified as the only Dec-31 filer, but the 8B model dropped one company
   (Salesforce) from the answer.
3. + "consider EVERY company one at a time" prompt -> completeness fixed (all
   five enumerated with correct per-company reasoning).
   RESIDUAL: on the negation phrasing, `llama3.1:8b` still flips its own
   correct analysis in the concluding sentence ("...does NOT end on Dec 31 is
   HubSpot"). This is a small-model negation-reasoning ceiling, not a
   retrieval/grounding defect. Looping was stopped here rather than overfitting
   prompts to one query.

Consequences:
- Direct factual comparatives (e.g. "compare the fiscal year-end dates of
  Atlassian and Salesforce") are now answered correctly and grounded.
- The residual negation error is not silently emitted as trustworthy: the
  red-team audit `flagged` it, so it flows into a low-confidence / review
  label — the intended behavior of the trust stack.
- `source_method="comparative_metadata"` is a new value; `finalize_with_audit`
  routes it through the red-team audit (not the single-company sanity check).
- `tests/test_pipeline.py` gains `TestComparativeMetadata` (field detection +
  narrative-query fall-through); 28 tests total, all passing.

Evidence:
- `src/stage2_self_correct.py` — `comparative_metadata_answer`, reordered
  `answer_with_self_correction`.
- Ground-truth end-to-end runs recorded during this session (Q5 correct; Q7
  reasoning correct, conclusion flipped and audit-flagged).

## Unverified Decisions

The following appear to be decisions but could not be confidently traced
to explicit reasoning in the repository. Do not assume these are settled;
confirm with the user before treating them as fixed:

- UNVERIFIED: Chunking parameters (`chunk_size=500`, `chunk_overlap=50`,
  character-based `RecursiveCharacterTextSplitter`) — these are hardcoded
  constants in `stage1_ingest.py` with no comment or doc explaining why
  these specific values were chosen over alternatives.
- UNVERIFIED: Embedding model choice (`BAAI/bge-small-en-v1.5`) and
  cross-encoder model choice (`cross-encoder/ms-marco-MiniLM-L-6-v2`) —
  used throughout the pipeline but no comparison against alternative
  models (e.g., larger BGE variants, OpenAI embeddings, other
  cross-encoders) was found in the repo.
- UNVERIFIED: RRF constant `k=60` in `stage3_hybrid.py` — this is a common
  default value in RRF literature, but no repo evidence confirms it was
  deliberately tuned versus just adopted as a standard default.
- UNVERIFIED: Candidate pool sizes (`CANDIDATE_POOL_SIZE = 15` in
  `stage3_hybrid.py`, `HYBRID_CANDIDATE_POOL_SIZE = 10` in
  `stage4_rerank.py`) — no documented reasoning for these specific numbers.
- UNVERIFIED: Decision to use Ollama + `llama3.1:8b` specifically (as
  opposed to a hosted LLM API) for the planned generation step — stated as
  a requirement in a prior session's task description, but no reasoning
  for local-vs-hosted was recorded in the repo itself.
- UNVERIFIED: Decision to have no formal test suite / CI — could be
  deliberate (early-stage research project) or simply not yet done; no
  statement either way was found.
