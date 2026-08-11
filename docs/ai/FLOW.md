# Flow

> Last updated: 2026-08-11

## Entry Points

All entry points in this repository are **CLI script invocations** — there
are no HTTP routes, background jobs, event handlers, scheduled tasks, or UI
actions observed anywhere in the codebase.

- `python stage1_ingest.py` — ingestion + chunking
- `python stage2_embed.py` — dense embedding + ChromaDB indexing + demo
  queries
- `python stage3_hybrid.py` — BM25 index build + hybrid search demo
  (reference/comparison only, not production)
- `python stage4_rerank.py` — cross-encoder rerank demo
  (reference/comparison only, not production)
- `python evaluate_retrieval.py` — full retrieval evaluation over
  `eval_set.json`
- `python debug_zoom_currency.py` — one-off diagnostic script for a
  specific query
- UNVERIFIED: `stage_generate.py` — planned entry point for LLM answer
  generation; does not exist yet (blocked, see `HANDOVER.md`)

Each script also has an `if __name__ == "__main__":` block, meaning its
top-level functions can also be imported by other scripts without
triggering the demo/CLI behavior (this is exploited by
`evaluate_retrieval.py` and `stage4_rerank.py` — see File-to-File Call Map).

## Main Execution Flows

### Flow: Ingestion

- **Trigger:** manual run of `python stage1_ingest.py`
- **Entry file/function:** `stage1_ingest.py` → `main()`
- **Intermediate functions:** `extract_pages()`, `find_repeated_lines()`,
  `strip_headers_footers()`, `build_full_text_with_offsets()`,
  `process_pdf()` (per file), `RecursiveCharacterTextSplitter.split_text()`
- **Data accessed:** reads all `*.pdf` files under `documents/`
- **Side effects:** writes `chunks.json` to the repo root (overwrites any
  existing file)
- **Output:** prints a summary (`Total documents processed`,
  `Total chunks created`, avg/min/max chunk length) to stdout

### Flow: Dense embedding + indexing

- **Trigger:** manual run of `python stage2_embed.py`
- **Entry file/function:** `stage2_embed.py` → `main()`
- **Intermediate functions:** `load_chunks()`, `embed_texts()` (calls
  `SentenceTransformer.encode()`), `build_collection()` (calls
  `chromadb.PersistentClient`, deletes+recreates the collection, then
  `collection.add()` in batches of 500)
- **Data accessed:** reads `chunks.json`; downloads/loads the
  `BAAI/bge-small-en-v1.5` model (from Hugging Face Hub on first run)
- **Side effects:** deletes and recreates the `saas_10k_filings` ChromaDB
  collection under `chroma_db/` — **destructive to any prior embedded
  state for that collection name**
- **Output:** prints an embed summary, then (in the `__main__` block) runs
  3 hardcoded test queries through `dense_search()` and prints results

### Flow: Hybrid search (reference only)

- **Trigger:** manual run of `python stage3_hybrid.py`, or import of
  `hybrid_search()` / `sparse_search()` by another script
- **Entry file/function:** `stage3_hybrid.py` → `main()` (when run
  directly) or direct import of `hybrid_search()`
- **Intermediate functions:** `build_bm25_index()`, `sparse_search()`
  (calls `clean_for_bm25()` when `preprocess=True`), `dense_search()`
  (imported from `stage2_embed.py`), RRF fusion logic inline in
  `hybrid_search()`
- **Data accessed:** reads `chunks.json` (in-memory BM25 index, rebuilt on
  every process start — **not persisted**); reads the existing
  `chroma_db/` collection (does not rebuild it)
- **Side effects:** none persisted; connects to the existing Chroma
  collection read-only via `dense_search()`
- **Output:** when run directly, prints side-by-side comparison of dense
  vs. sparse vs. hybrid for 3 hardcoded queries

### Flow: Cross-encoder reranking (reference only)

- **Trigger:** manual run of `python stage4_rerank.py`, or import of
  `hybrid_rerank_search()` / `rerank()` by another script
- **Entry file/function:** `stage4_rerank.py` — note module-level code
  (outside any function) eagerly loads `_chunks`, `_bm25`,
  `_embedding_model`, `_collection`, `_cross_encoder` **at import time**,
  not inside `main()`
- **Intermediate functions:** `hybrid_search()` (imported from
  `stage3_hybrid.py`), `rerank()` (calls
  `CrossEncoder.predict()`)
- **Data accessed:** same as hybrid search flow, plus the
  `cross-encoder/ms-marco-MiniLM-L-6-v2` model
- **Side effects:** none persisted
- **Output:** when run directly, prints a single demo query's top-5
  reranked results

### Flow: Evaluation

- **Trigger:** manual run of `python evaluate_retrieval.py`
- **Entry file/function:** `evaluate_retrieval.py` → `main()`
- **Intermediate functions:** `load_eval_set()`, `run_methods()` (calls
  `dense_search()`, `hybrid_search()` x2 with/without `preprocess`,
  `hybrid_rerank_search()` x2 with/without `preprocess`), `hit_at_k()`,
  `precision_at_1()`
- **Data accessed:** reads `eval_set.json`; reuses the module-level
  singletons imported from `stage4_rerank.py` (`_bm25`, `_chunks`,
  `_collection`, `_embedding_model`) rather than re-instantiating them
- **Side effects:** none persisted — this script only prints to stdout
- **Output:** per-query breakdown printed for all 5 methods, followed by a
  `Final Summary` table (hit_rate@5 / precision@1 averaged over 28 queries)

### Flow: Debug script

- **Trigger:** manual run of `python debug_zoom_currency.py`
- **Entry file/function:** `debug_zoom_currency.py` → `main()`
- **Intermediate functions:** `load_chunks()`, `find_matching_chunks()`
  (local keyword search over `chunks.json`), then calls `dense_search()`,
  `hybrid_search()`, `hybrid_rerank_search()` for one hardcoded query
- **Data accessed:** `chunks.json`; reuses singletons imported from
  `stage4_rerank.py`
- **Side effects:** none
- **Output:** printed comparison of whether a keyword-matched chunk
  appears in each method's top-k results

### Flow: LLM answer generation (planned, UNVERIFIED)

UNVERIFIED: this flow does not exist yet. Per `HANDOVER.md`, the intended
shape was `stage_generate.py` calling `dense_search()` then a local Ollama
`llama3.1:8b` model for `generate_answer()` and `critique_sufficiency()`,
but the script was never created because Ollama could not be verified as
installed/running on this machine.

## File-to-File Call Map

```
stage1_ingest.py        (standalone; writes chunks.json)

stage2_embed.py          (standalone; reads chunks.json, writes chroma_db/)
    ^
    | imports dense_search, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME
    |
stage3_hybrid.py          (imports from stage2_embed.py)
    ^
    | imports hybrid_search, build_bm25_index, load_chunks
    |
stage4_rerank.py          (imports from stage3_hybrid.py, and
                            CHROMA_DIR/COLLECTION_NAME/EMBEDDING_MODEL_NAME
                            from stage2_embed.py directly)
    ^
    | imports _bm25, _chunks, _collection, _embedding_model,
    | hybrid_rerank_search
    |
evaluate_retrieval.py     (imports dense_search from stage2_embed.py,
                            hybrid_search from stage3_hybrid.py, and the
                            above from stage4_rerank.py)

debug_zoom_currency.py    (imports the same set as evaluate_retrieval.py:
                            dense_search from stage2_embed.py,
                            hybrid_search from stage3_hybrid.py,
                            singletons + hybrid_rerank_search from
                            stage4_rerank.py)
```

Note the layering: `stage4_rerank.py` is a **hard dependency** for both
`evaluate_retrieval.py` and `debug_zoom_currency.py`, because they both
import its module-level singletons (`_bm25`, `_chunks`, `_collection`,
`_embedding_model`) to avoid reloading models repeatedly. This means
**importing `evaluate_retrieval.py` or `debug_zoom_currency.py` triggers
model loading (SentenceTransformer, CrossEncoder) and a ChromaDB connection
as a side effect of the import itself**, not lazily on first use.

## Current Change Area

_(Leave blank between sessions; fill in when actively modifying a flow.)_

- UNVERIFIED: none — no flow is currently being modified. Last activity was
  documentation-only (creating `docs/ai/`).

## Risks

- **Import-time side effects (shared mutable state):** `stage4_rerank.py`
  loads two ML models and opens a ChromaDB connection at module import
  time via module-level variables (`_bm25`, `_chunks`, `_embedding_model`,
  `_collection`, `_cross_encoder`). Any script that imports from it
  (`evaluate_retrieval.py`, `debug_zoom_currency.py`) pays this cost
  immediately and implicitly, even if it never calls
  `hybrid_rerank_search()`. This is fragile: a future refactor that adds an
  import of `stage4_rerank.py` anywhere will silently trigger model loads.
- **No test coverage:** there are no automated tests (see
  `TEST_CHECKLIST.md`), so regressions in any of the flows above would only
  be caught by manually re-running scripts and eyeballing output.
- **Destructive re-embedding:** `stage2_embed.py`'s `build_collection()`
  unconditionally deletes and recreates the `saas_10k_filings` ChromaDB
  collection every time it runs. There is no confirmation prompt or
  "skip if already populated" guard — re-running `stage2_embed.py`
  silently discards the existing index.
  Rerunning `stage1_ingest.py` similarly overwrites `chunks.json`
  unconditionally.
- **External-service dependency at import/runtime:** `dense_search()` and
  the cross-encoder path depend on Hugging Face Hub being reachable on
  first model load (observed "unauthenticated requests" warning), and any
  future `stage_generate.py` will depend on a local Ollama server. Neither
  dependency is currently verified/guarded in code — failures would surface
  as raw exceptions, not a clear error message.
- **Implicit ordering assumption:** `evaluate_retrieval.py` and
  `debug_zoom_currency.py` assume `chroma_db/` already contains a populated
  `saas_10k_filings` collection (i.e., that `stage2_embed.py` was run
  first). There is no check for this — calling `client.get_collection()`
  on a missing collection would raise an unhandled exception.
- **Eval set is hand-maintained, not generated:** `eval_set.json` has been
  directly edited by the user across sessions (28 queries as of this
  writing). Any tooling that regenerates or reformats this file risks
  destroying manually curated content — see `CONSTRAINTS.md`.
