# Test Checklist

> Last updated: 2026-08-11

No `package.json`, `Makefile`, `pyproject.toml`, `setup.py`, `Cargo.toml`,
`go.mod`, `composer.json`, `build.gradle`, `pom.xml`, `Dockerfile`,
`docker-compose.yml`, or CI config files were found anywhere in this
repository (checked at repo root and searched for common CI paths such as
`.github/workflows/`). This project has **no formal build/lint/test
tooling configured**. The commands below are inferred from actual script
usage observed in prior sessions (see `README_module1.md` and script
contents), not from a manifest.

## Before Starting

- [ ] Working tree clean? Run `git status` — UNVERIFIED baseline: as of
      this session, the repo has **no commits yet** and all files are
      untracked, so "clean" currently means "matches the untracked state
      you expect."
- [ ] Current branch correct? Run `git branch --show-current` — repo is on
      `main` as of this session (no other branches observed).
- [ ] Necessary environment variables present? None are required by the
      current scripts (no `.env` file or `os.environ` reads observed in
      `stage1_ingest.py`–`stage4_rerank.py` or `evaluate_retrieval.py`).
      TODO: confirm once `stage_generate.py` (Ollama integration) exists,
      as it may need `OLLAMA_HOST` or similar — UNVERIFIED.
- [ ] Local services running? For anything beyond `stage1_ingest.py` /
      `stage2_embed.py`, Hugging Face Hub must be reachable (model
      downloads observed on first run). For any future Ollama-dependent
      script, a local Ollama server must be running and reachable at
      `http://localhost:11434` — verify with `ollama list` first.

## Automated Checks

There is no `pip install -r requirements.txt` step to run automatically as
part of this checklist per current task constraints (installing
dependencies is out of scope unless explicitly requested) — listed here
for completeness only:

- **Install command:** `.\venv\Scripts\python.exe -m pip install -r requirements.txt`
  - Expected output: `Successfully installed ...` or
    `Requirement already satisfied` for every pinned package, exit code 0.
  - If it fails: check for a version conflict in `requirements.txt`; do not
    edit the file to "fix" it without approval (see `CONSTRAINTS.md`).

- **Typecheck command:** TODO: discover command for typecheck — no
  `mypy.ini`, `pyrightconfig.json`, or type-checking config found in the
  repo.

- **Lint command:** TODO: discover command for lint — no `.flake8`,
  `.pylintrc`, `ruff.toml`, or similar config found in the repo.

- **Unit test command:** TODO: discover command for unit tests — no
  `tests/` directory or `test_*.py` / `*_test.py` files exist at the
  project level (confirmed by search; only third-party test files exist
  inside `venv/Lib/site-packages/`, which are not this project's tests).

- **Integration/pipeline "smoke test" command** (the closest thing this
  repo has to an automated check, inferred from actual prior usage):
  ```
  .\venv\Scripts\python.exe stage1_ingest.py
  .\venv\Scripts\python.exe stage2_embed.py
  .\venv\Scripts\python.exe evaluate_retrieval.py
  ```
  - Expected output: `stage1_ingest.py` prints a summary block ending in
    total documents/chunks/avg/min/max chunk length; `stage2_embed.py`
    prints an embed summary (chunk count, embedding dimension) followed by
    3 test-query result blocks; `evaluate_retrieval.py` prints a per-query
    breakdown followed by a `Final Summary` table with `hit_rate@5` and
    `precision@1` columns. Exit code 0 for all three.
  - If it fails: read the traceback — most likely causes based on prior
    sessions are a missing/unreachable Hugging Face Hub connection (model
    download) or a stale/missing `chroma_db/` collection if
    `stage2_embed.py` was skipped.
  - Note: this is **not a true regression test** — it re-runs the full
    pipeline (including a multi-minute embedding pass over ~4,993 chunks
    per prior timing) and only checks that it completes and prints
    plausible output, not that specific numeric results match a golden
    baseline. TODO: consider adding an actual `tests/` suite with
    assertions if this project grows.

- **Build command:** TODO: discover command for build — no build step
  exists; this is a script-based project with no packaging/bundling
  observed.

## Manual Checks

- [ ] Re-run `stage1_ingest.py` and visually confirm the printed summary
      (`Total documents processed`, `Total chunks created`, etc.) is
      plausible for the 5 PDFs in `documents/`.
- [ ] Re-run `stage2_embed.py` and check the 3 printed test-query result
      blocks for topical relevance (e.g. "revenue growth drivers" should
      surface MD&A-style text).
- [ ] Re-run `evaluate_retrieval.py` and compare the `Final Summary` table
      against the baseline recorded in `README_module1.md` (dense
      1.000/0.893 hit_rate@5/precision@1 as of the last documented run) —
      a large unexplained deviation is a signal something changed.
- [ ] If touching `stage3_hybrid.py` or `stage4_rerank.py`, confirm they
      still import cleanly and their respective `if __name__ == "__main__"`
      demo blocks still run, since `evaluate_retrieval.py` and
      `stage4_rerank.py` import functions from each other at module load
      time (see `FLOW.md` for the call map).
- [ ] Check console output for unhandled exceptions/tracebacks — there is
      no logging framework in use; all diagnostics are `print()` statements
      to stdout.

## Definition of Done

A change is only done when:

- [ ] Relevant automated checks above pass (or are explicitly marked
      TODO/not-applicable for this change).
- [ ] The pipeline smoke test (`stage1_ingest.py` → `stage2_embed.py` →
      `evaluate_retrieval.py`, or the relevant subset) completes without
      error.
- [ ] Manual checks relevant to the change pass.
- [ ] `docs/ai/HANDOVER.md` is updated.
- [ ] `docs/ai/DECISIONS.md` is updated if a decision was made or changed.
- [ ] `docs/ai/FLOW.md` is updated if execution flow changed.
