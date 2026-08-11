# Module 1: Retrieval Pipeline

## Pipeline Overview

1. **Ingestion & chunking** (`stage1_ingest.py`) — Loads all PDFs from `documents/`
   with `pypdf`, extracts text page by page, strips likely repeated
   headers/footers (short lines that recur near-identically across pages of
   the same document), and splits the cleaned text into chunks with
   LangChain's `RecursiveCharacterTextSplitter` (chunk_size=500,
   chunk_overlap=50, character-based). Each chunk is stored with a
   `chunk_id`, `source_file`, best-effort `page_number`, and `text`, written
   to `chunks.json`.
2. **Dense embedding** (`stage2_embed.py`) — Embeds every chunk with the
   `BAAI/bge-small-en-v1.5` sentence-transformers model and loads the
   vectors into a persistent local ChromaDB collection (`chroma_db/`,
   collection `saas_10k_filings`). Exposes `dense_search(query, k)` for
   cosine-similarity retrieval.
3. **Sparse retrieval** (`stage3_hybrid.py`) — Builds a `BM25Okapi` index
   over all chunk texts (whitespace/lowercase tokenization) and exposes
   `sparse_search(query, k, preprocess=False)`. When `preprocess=True`, the
   query is run through `clean_for_bm25()` (lowercase + hardcoded stopword
   removal) before BM25 tokenization; the dense query is never altered.
4. **Hybrid fusion** (`stage3_hybrid.py`) — `hybrid_search(query, k, preprocess=False)`
   pulls the top 15 candidates from both dense and sparse search and fuses
   the two ranked lists with Reciprocal Rank Fusion (RRF, constant k=60):
   `score = sum(1 / (60 + rank))` across both lists. The `preprocess` flag
   passes through to the internal `sparse_search` call only.
5. **Cross-encoder reranking** (`stage4_rerank.py`) — `hybrid_rerank_search(query, k, preprocess=False)`
   takes the top 10 hybrid candidates and reranks them with the
   `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder, returning the
   top `k` by cross-encoder score.
6. **Evaluation** (`evaluate_retrieval.py`) — Runs all five retrieval
   methods (`dense`, `hybrid`, `hybrid_preprocessed`, `hybrid_rerank`,
   `hybrid_rerank_preprocessed`), all at k=5, against a hand-written eval
   set (`eval_set.json`) and reports `hit_rate@5` and `precision@1` per
   method, plus a per-query breakdown.

## Final Evaluation Results (28 queries, 5 methods)

| Method                      | hit_rate@5 | precision@1 |
|------------------------------|-----------:|------------:|
| dense                        |      1.000 |       0.893 |
| hybrid                       |      0.964 |       0.679 |
| hybrid_preprocessed          |      0.964 |       0.679 |
| hybrid_rerank                |      0.964 |       0.821 |
| hybrid_rerank_preprocessed   |      0.964 |       0.786 |

## Findings

**Dense-only retrieval outperformed all hybrid variants** on both
hit_rate@5 and precision@1, across every configuration tested (plain
hybrid, preprocessed hybrid, reranked hybrid, and reranked preprocessed
hybrid). No hybrid variant matched dense on either metric.

**Stability was confirmed via 3 repeated eval runs.** `evaluate_retrieval.py`
was run three times back-to-back against the same 28-query eval set with no
code changes. All three runs produced byte-identical results — same
per-query hit@5/precision@1 outcomes, same aggregate table (dense 1.000/0.893,
hybrid 0.964/0.679, hybrid_rerank 0.964/0.821) — a min/max range of zero
across all methods and both metrics. The evaluation pipeline is fully
deterministic; the results are not an artifact of run-to-run noise.

**Query preprocessing (stopword removal) was tested and showed no
improvement.** `clean_for_bm25()` strips a hardcoded stopword list
(what, does, is, the, a, how, of, etc.) from the query before BM25
tokenization. Comparing `hybrid` vs `hybrid_preprocessed` showed **identical**
results (0.964/0.679 for both), and `hybrid_rerank_preprocessed` was
actually *worse* than `hybrid_rerank` (0.786 vs 0.821 precision@1). This is
expected: BM25 already downweights common terms via IDF, so stripping them
manually is largely redundant. More importantly, the specific failure this
experiment targeted (the Zoom foreign-currency query) still missed hit@5
under every preprocessed variant — the root cause is a **candidate-pool
limitation**, not query phrasing. If the correct chunk never enters BM25's
top-15 candidates in the first place, no amount of query cleanup changes
what gets fused or reranked downstream.

**Root cause: 10-K boilerplate language makes BM25 less discriminative for
this corpus.** SEC 10-K filings share substantial boilerplate — cover page
structure, standard risk-factor phrasing ("Foreign Currency Exchange Risk",
"Item 7A. Qualitative and Quantitative Disclosures About Market Risk"), and
repeated legal/financial terminology — across all five companies. This
creates high keyword overlap between documents regardless of which
company's filing is actually being asked about, so BM25 scores often favor
the wrong document's boilerplate chunk over the right document's specific
answer. Dense embeddings, by contrast, capture semantic/topical relevance
rather than surface term overlap and are far less susceptible to this
cross-document boilerplate collision.

## Decision

**`dense_search` (from `stage2_embed.py`) will be used as the retrieval
function for Module 2 onward.** It matched or beat every hybrid variant on
every metric tested, held up under repeated-run stability checks, and
remained the best performer even after a targeted preprocessing experiment
aimed at closing the gap. It is also simpler and faster than the hybrid
alternatives (no BM25 index or cross-encoder pass required).
`stage3_hybrid.py` and `stage4_rerank.py` are retained in the repo for
reference/comparison but are not part of the production pipeline going
forward.

## Package Versions

Key packages (see `requirements.txt` for the full pinned dependency list):

- `pypdf==6.15.0`
- `langchain==1.3.14`
- `langchain-text-splitters==1.1.2`
- `sentence-transformers==5.7.0`
- `chromadb==1.5.9`
- `rank-bm25==0.2.2`
- `torch==2.13.0`
- `transformers==5.15.0`
- `numpy==2.5.2`

## Dataset & Evaluation Stats

- **5** SEC 10-K filings (Atlassian, DocuSign, HubSpot, Salesforce, Zoom —
  fiscal year 2026)
- **~4,993** chunks after cleaning and splitting
- **28** hand-written evaluation queries
- **3** stability-check reruns of the full eval set (zero variance observed)
- **1** query-preprocessing experiment (stopword removal; no improvement)
