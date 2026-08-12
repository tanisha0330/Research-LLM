# Rollback Plan

> Last updated: 2026-08-11

## Current Safe Reference Point

- **Stale as of 2026-08-11 — commits now exist.** The repo has a real
  commit history (`git log`) including the Module 1 pipeline, Modules
  2–4, and a reorg into `src/`/`eval/`/`reports/`/`experiments/`. Run
  `git log --oneline` for the current baseline before relying on the
  "no commits" framing below.

## General Rollback Procedure (once commits exist)

1. Identify the last known safe commit (to be recorded below as history
   accumulates).
2. Confirm current working tree state: `git status`.
3. If uncommitted changes need to be preserved before rolling back:
   `git stash push -u -m "<description>"`.
4. To revert to the last safe commit without losing history:
   `git revert <bad-commit-hash>` (creates a new commit undoing the
   change — preferred for shared/pushed history).
5. To hard-reset a local-only branch to a known-safe commit (destructive,
   local work only, requires explicit user approval per `CONSTRAINTS.md`):
   `git reset --hard <safe-commit-hash>`.
6. Re-run the verification steps in `docs/ai/TEST_CHECKLIST.md` after any
   rollback to confirm the repo is actually back to a working state.

## Files/Artifacts Outside Git History

These are not (yet) tracked by git and have their own regeneration paths
rather than a git-revert path:

- `src/chunks.json` — regenerate by re-running `python src/stage1_ingest.py`
  against the current contents of `documents/` (still tracked in git — the
  PDFs are included in this repo, unlike `chunks.json` itself). This is
  fast (single-pass PDF parsing).
- `src/chroma_db/` — regenerate by re-running `python src/stage2_embed.py`,
  which deletes and recreates the `saas_10k_filings` collection from
  `src/chunks.json`. Per `FLOW.md`'s Risks section, this is a multi-minute
  operation (embedding ~4,993 chunks) and is destructive to whatever
  collection state existed before — there is no "undo," only "regenerate
  from `chunks.json`."
- `venv/` — not source-controlled; if corrupted, delete and recreate with
  `python -m venv venv` followed by
  `.\venv\Scripts\python.exe -m pip install -r requirements.txt`
  (per `requirements.txt` pinned versions).
- `eval_set.json` — **hand-curated by the user across sessions.** There is
  no automated regeneration path for this file's content. If it is
  accidentally overwritten, it must be manually reconstructed or restored
  from a prior git commit/backup — treat with extra care (see
  `CONSTRAINTS.md`).

## Verification Steps After Any Rollback

1. Confirm `documents/` still contains the expected 5 PDF files.
2. Re-run `python stage1_ingest.py` if `chunks.json` was affected; confirm
   the printed summary matches expectations (~4,993 chunks from 5
   documents, per `README_module1.md`'s last recorded baseline).
3. Re-run `python stage2_embed.py` if `chroma_db/` was affected; confirm
   embedding dimension = 384 and chunk count matches `chunks.json`.
4. Re-run `python evaluate_retrieval.py` and compare the `Final Summary`
   table against the baseline in `README_module1.md` (dense: 1.000
   hit_rate@5 / 0.893 precision@1, as of the last documented run) to
   confirm the rollback did not silently change retrieval behavior.
5. Update `docs/ai/HANDOVER.md` with the outcome of the rollback.

## Risky Changes Log

_(Record entries here whenever a risky change is made, so future sessions
know what was recently touched and how to undo it.)_

- 2026-08-11 — No risky/destructive changes made. This session only added
  new documentation files under `docs/ai/`; no existing source files,
  data files (`chunks.json`, `eval_set.json`), or the `chroma_db/`
  collection were modified.
