# Architecture

> Last updated: 2026-08-11

## System Overview

This project is a **retrieval-augmented document-QA pipeline** over a fixed
set of 5 SEC 10-K PDF filings (Atlassian, DocuSign, HubSpot, Salesforce,
Zoom — fiscal year 2026, per `documents/`). It ingests the PDFs, chunks and
embeds their text, indexes the chunks for retrieval, and evaluates several
retrieval strategies (dense embedding search, BM25 sparse search, RRF hybrid
fusion, cross-encoder reranking) against a hand-written eval set to decide
which retrieval method to standardize on. A planned-but-not-yet-built next
step (blocked, see `HANDOVER.md`) is LLM answer generation over retrieved
chunks using a local Ollama model.

There is no web server, API, or UI layer observed — the system currently
runs as a sequence of standalone Python scripts executed manually from the
command line.

## Tech Stack

- **Language:** Python (version UNVERIFIED — no `pyproject.toml` or
  `.python-version` file found; venv exists at `venv/`)
- **Package manager:** pip, with dependencies pinned in `requirements.txt`
  (no `pyproject.toml`, `Pipfile`, or `poetry.lock` found)
- **Key libraries** (from `requirements.txt`):
  - `pypdf` — PDF text extraction
  - `langchain` / `langchain-text-splitters` — text chunking
    (`RecursiveCharacterTextSplitter`)
  - `sentence-transformers` — dense embeddings (`BAAI/bge-small-en-v1.5`)
    and cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  - `chromadb` — persistent local vector database
  - `rank-bm25` — BM25 sparse retrieval
  - `torch`, `transformers` — ML backend for the above
- **Database / storage:** ChromaDB, persisted to local disk at `chroma_db/`
  (no external/hosted database observed)
- **Build tools:** none found (no Makefile, Dockerfile, or build scripts)
- **Test tools:** none found at the project level (no pytest config, no
  `tests/` directory containing project test files — confirmed by search)
- **CI/CD:** none found (no `.github/workflows/`, no other CI config)
- **Planned addition (blocked):** `ollama` Python client for local LLM
  calls via a local Ollama server running `llama3.1:8b` — not yet installed
  or verified working on this machine (see `HANDOVER.md`)

## Folder Map

- `documents/` — source PDF files (5 SEC 10-K filings), read-only input data
- `chroma_db/` — generated ChromaDB persistent vector store (output
  artifact, not hand-authored)
- `venv/` — Python virtual environment (not source-controlled logic)
- `docs/ai/` — this AI-collaboration documentation system (new as of this
  session)
- `docs/ai/bugs/` — placeholder for per-bug documentation (empty,
  `.gitkeep` only)
- `docs/ai/features/` — placeholder for per-feature documentation (empty,
  `.gitkeep` only)
- Repository root — all pipeline scripts live flat in the root (no `src/`
  layout observed): `stage1_ingest.py`, `stage2_embed.py`,
  `stage3_hybrid.py`, `stage4_rerank.py`, `evaluate_retrieval.py`,
  `debug_zoom_currency.py`
- `chunks.json`, `eval_set.json` — generated/hand-written data files at root
- `README_module1.md` — narrative writeup of the Module 1 pipeline and
  findings (root)
- `requirements.txt` — pinned Python dependencies (root)

## Main Modules / Services

- **Ingestion (`stage1_ingest.py`)** — Loads PDFs from `documents/`,
  extracts text per page via `pypdf`, strips likely repeated
  headers/footers, splits into ~500-char chunks with 50-char overlap, and
  writes `chunks.json` (chunk_id, source_file, page_number, text).
- **Dense retrieval (`stage2_embed.py`)** — Embeds chunks with
  `BAAI/bge-small-en-v1.5` and loads them into a ChromaDB collection
  (`saas_10k_filings`). Exposes `dense_search(query, k)`. **This is the
  production retrieval module per the recorded decision.**
- **Sparse / hybrid retrieval (`stage3_hybrid.py`)** — Builds a BM25 index
  over chunk texts and combines it with dense search via Reciprocal Rank
  Fusion. Retained for reference/comparison only, not production.
- **Reranking (`stage4_rerank.py`)** — Cross-encoder reranking
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over hybrid candidates. Retained
  for reference/comparison only, not production.
- **Evaluation (`evaluate_retrieval.py`)** — Runs all retrieval methods
  against `eval_set.json` (28 queries) and reports hit_rate@5 /
  precision@1 per method, plus a per-query breakdown.
- **Debugging (`debug_zoom_currency.py`)** — One-off diagnostic script for a
  specific eval query miss; not a reusable module.
- **Answer generation (planned, not built)** — `stage_generate.py` would
  add an LLM-generation layer (Ollama, `llama3.1:8b`) on top of
  `dense_search`, with a sufficiency-critique step. Currently blocked.

## Data Flow

1. `documents/*.pdf` → `stage1_ingest.py` → `chunks.json`
2. `chunks.json` → `stage2_embed.py` → embeddings written into
   `chroma_db/` (persistent ChromaDB collection `saas_10k_filings`)
3. `chunks.json` (+ `chroma_db/` via `dense_search`) → `stage3_hybrid.py` →
   in-memory BM25 index (not persisted) + hybrid fusion logic, callable as
   `hybrid_search()`
4. Hybrid candidates → `stage4_rerank.py` → cross-encoder reranked results,
   callable as `hybrid_rerank_search()`
5. `eval_set.json` + all of the above retrieval functions →
   `evaluate_retrieval.py` → console-printed per-query breakdown and
   summary metrics table (not currently written to a file)
6. UNVERIFIED / planned: retrieved chunks → `stage_generate.py` → local
   Ollama LLM call → generated answer + sufficiency critique (not yet
   implemented)

There is no persistent application state beyond `chunks.json` and
`chroma_db/`; everything else is recomputed in-memory on each script run.

## External Dependencies

- **Hugging Face Hub** — model weights for `BAAI/bge-small-en-v1.5` and
  `cross-encoder/ms-marco-MiniLM-L-6-v2` are downloaded from the Hugging
  Face Hub at first run (observed "unauthenticated requests to the HF Hub"
  warning in prior run output); no `HF_TOKEN` currently configured.
- **Ollama (local)** — planned dependency for local LLM inference
  (`llama3.1:8b`), served over `localhost:11434`. UNVERIFIED / not
  currently installed or reachable on this machine.
- **ChromaDB** — embedded/local, not a hosted external service (persisted
  to `chroma_db/` on disk).
- No cloud APIs, payment processors, auth providers, or message queues were
  observed anywhere in the codebase.

## Boundaries

- The **retrieval-method decision boundary**: `README_module1.md` and
  `DECISIONS.md` document a specific, evidence-based choice
  (`dense_search` as production). Changes to which method is "production"
  should not be made silently — see `CONSTRAINTS.md`.
- The **generated-artifact boundary**: `chunks.json` and `chroma_db/` are
  derived from `documents/` + the embedding model. They should be
  regenerated via the stage scripts, not hand-edited.
- The **external-service boundary**: any code path that calls out to
  Hugging Face Hub or a local Ollama server should fail gracefully/loudly
  when those are unavailable rather than silently fabricating results (see
  the Ollama-unavailable handling already established in this session).

## Uncertainties

- UNVERIFIED: exact Python version in use (venv exists but no
  `.python-version` / `pyproject.toml` pin found).
- UNVERIFIED: whether this repo is meant to grow into a larger
  application (API, UI) or remain a script-based research pipeline —
  no product/architecture doc beyond `README_module1.md` (which covers
  only Module 1) was found.
- UNVERIFIED: intended scope of "Module 2" — referenced in
  `README_module1.md` and this session's conversation as the next phase
  (LLM generation), but no module-2-level design doc exists yet.
- UNVERIFIED: deployment target, if any — no Dockerfile, cloud config, or
  infra-as-code found; this appears to be a local/dev-only project so far.
