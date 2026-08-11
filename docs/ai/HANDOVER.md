# Project Handover

> Living document. Update this at the end of every AI-assisted session so the
> next session (human or AI) can pick up without re-deriving context.
> Last updated: 2026-08-11

## Session Handover — 2026-08-11

### Done
Created the full `docs/ai/` documentation system: `HANDOVER.md`,
`ARCHITECTURE.md`, `CONSTRAINTS.md`, `TEST_CHECKLIST.md`, `FLOW.md`,
`DECISIONS.md`, `ROLLBACK.md`, `CONTEXT_LOG.md`, and empty
`bugs/.gitkeep` / `features/.gitkeep` placeholders. All content was
derived from reading actual repo files (`README_module1.md`, the stage
scripts, `requirements.txt`, `git status`/`git log`) — nothing was
invented; gaps are marked UNVERIFIED or TODO throughout.

### In Progress
Nothing code-level is in progress. Several follow-up documentation
requests arrived mid-session as templates with unfilled placeholders
(inline-comment target, bug filename, feature filename, "describe the
change") — these are queued but cannot be started without the user
supplying the missing specifics. See the end of this session's chat
response for the exact list.

### Blocked / Risky
- Module 2 (LLM answer generation via local Ollama `llama3.1:8b`) remains
  blocked — Ollama was not detected on this machine in the prior session
  (not in PATH, no known install dir, no running process, port 11434
  unreachable). Unchanged this session; not re-checked.
- No risky or destructive changes were made this session — docs-only, no
  source code, dependencies, or data files touched. See `ROLLBACK.md`'s
  Risky Changes Log.

### Next Step
Ask the user for the missing specifics on the four queued template
requests (inline-doc target; bug name + description; feature name +
description; "describe the change" for the plan-first request) before
acting on any of them — do not guess a bug/feature name or a code target.

### Files to Read First
- `docs/ai/HANDOVER.md` (this file)
- `docs/ai/CONSTRAINTS.md`
- `docs/ai/DECISIONS.md`
- `README_module1.md` (root)

## Current State

- A document-QA retrieval pipeline over 5 SEC 10-K PDFs (Atlassian,
  DocuSign, HubSpot, Salesforce, Zoom), implemented as a sequence of
  numbered stage scripts run manually from the command line (no
  orchestration framework, no web server, no CLI entry point observed).
- `stage1_ingest.py` — loads PDFs from `documents/` with `pypdf`, strips
  likely repeated headers/footers, chunks text with LangChain's
  `RecursiveCharacterTextSplitter` (chunk_size=500, chunk_overlap=50), and
  writes `chunks.json` (~4,993 chunks per `README_module1.md`).
- `stage2_embed.py` — embeds chunks with `BAAI/bge-small-en-v1.5`
  (sentence-transformers) into a persistent local ChromaDB collection
  (`chroma_db/`, collection name `saas_10k_filings`). Exposes
  `dense_search(query, k)`.
- `evaluate_retrieval.py` — runs a 28-query hand-written eval set
  (`eval_set.json`) against multiple retrieval methods and reports
  hit_rate@5 / precision@1. Per `README_module1.md`, this was run
  successfully and repeated 3x with zero variance in results.
- **Decision recorded in `README_module1.md`:** `dense_search` (from
  `stage2_embed.py`) is the chosen retrieval method going forward, based on
  it outperforming BM25/hybrid/reranked variants on the eval set.
- A Python virtual environment (`venv/`) exists with dependencies installed
  and pinned in `requirements.txt`.

## In Progress

- `stage3_hybrid.py` (BM25 sparse search + RRF hybrid fusion) and
  `stage4_rerank.py` (cross-encoder reranking) are implemented and were
  evaluated, but per `README_module1.md` are explicitly **retained for
  reference/comparison only** — not the production path.
- `stage_generate.py` (LLM answer generation over retrieved chunks via a
  local Ollama `llama3.1:8b` model, plus a sufficiency-critique function)
  was requested but **not created**. Work stopped because Ollama could not
  be detected on this machine (not in PATH, no known install directory, no
  running process, API port 11434 unreachable). This is the first unfinished
  piece of work in the repo.
- `docs/ai/` documentation system (this file and siblings) is being created
  in the current session and was not previously present.

## Known Issues

- UNVERIFIED: whether `stage1_ingest.py` through `evaluate_retrieval.py`
  still run successfully right now — this session did not re-execute them.
  Prior session transcripts (reflected in `README_module1.md`) report
  successful runs as of 2026-08-11, but no automated test suite exists to
  reconfirm this on demand.
- UNVERIFIED: no lint, type-check, or automated test tooling was found in
  the repo (no `pytest`, `tox.ini`, `pyproject.toml`, `.flake8`, or CI
  config observed). `TEST_CHECKLIST.md` should be treated as best-effort
  until such tooling exists.
- The git repository (`.git/` present) has **no commits yet** — all files
  are currently untracked. There is no commit history to fall back on for
  rollback purposes yet.
- `chroma_db/` and `chunks.json` are generated artifacts checked into the
  working tree (not yet committed); if `documents/` changes, these will go
  stale and need regeneration by re-running `stage1_ingest.py` and
  `stage2_embed.py`.

## Do Not Touch

- Do not silently re-promote `stage3_hybrid.py` / `stage4_rerank.py` to be
  the production retrieval path — this reverses a documented, evidence-based
  decision in `README_module1.md`. See `DECISIONS.md`.
- Do not fabricate or assume Ollama/LLM output — this environment does not
  currently have a verified working Ollama installation. Any code depending
  on it must first verify with `ollama list`.
- Do not delete/regenerate `chunks.json` or `chroma_db/` without
  understanding they represent ~5 minutes of embedding compute over
  ~4,993 chunks (per prior session timing) — regenerating is not free.
- See `CONSTRAINTS.md` for the full list of hard rules.

## Next Best Step

1. Ask the user to confirm Ollama is installed and `llama3.1:8b` is pulled
   (verify via `ollama list`).
2. If confirmed: install the `ollama` Python client, add it to
   `requirements.txt`, and write `stage_generate.py` implementing
   `generate_answer()` and `critique_sufficiency()` as originally specified.
3. If not confirmed: do not proceed with generation work — this is a hard
   external blocker, not something to route around.
4. Independent of the above: consider re-running `stage1_ingest.py` through
   `evaluate_retrieval.py` once to reconfirm current pipeline state before
   building anything new on top of it, since no automated verification
   exists.

## Evidence

- `README_module1.md` (root) — full Module 1 pipeline description, eval
  results table (5 methods x 28 queries), findings, and the dense-only
  decision with reasoning.
- `stage1_ingest.py`, `stage2_embed.py`, `stage3_hybrid.py`,
  `stage4_rerank.py`, `evaluate_retrieval.py`, `debug_zoom_currency.py` —
  present in repo root; read directly for function signatures and behavior.
- `requirements.txt` (root) — pinned dependency list confirming
  `sentence-transformers`, `chromadb`, `rank-bm25`, `pypdf`, `langchain`,
  `langchain-text-splitters`, `torch`, `transformers` are installed; no
  `ollama` package present yet.
- `eval_set.json` (root) — 28 query/expected_source/expected_keywords
  objects used by `evaluate_retrieval.py`.
- `chunks.json`, `chroma_db/`, `documents/` (root) — generated
  data/artifacts confirming the pipeline has been run at least once.
- `git status` / `git log` output — confirms a git repo exists locally with
  no commits and all files untracked as of this session.
- Absence of `package.json`, `pyproject.toml`, `Dockerfile`, CI config
  (`.github/workflows/`), or test directories — used to infer no
  build/lint/CI tooling currently exists (UNVERIFIED as an exhaustive
  check; based on directory listing only).
