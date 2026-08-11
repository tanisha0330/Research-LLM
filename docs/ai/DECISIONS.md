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
