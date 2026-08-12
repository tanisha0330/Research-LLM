# Constraints

> Explicit boundaries for AI assistants working in this repository.
> Last updated: 2026-08-11

## Hard Rules

- Do not install new dependencies without asking first.
- Do not delete existing functionality unless explicitly instructed.
- Do not refactor unrelated code.
- Do not change authentication, authorization, payments, billing, secrets,
  or database migrations without explicit approval.
- Do not change CI/CD pipelines without approval.
- Do not change public APIs unless asked.
- Do not make broad formatting or linting changes across the repo unless
  asked.
- Do not claim a task is complete without running the relevant checklist in
  `docs/ai/TEST_CHECKLIST.md`.

## Change Rules

- One logical change per request.
- Before implementing, explain the plan.
- Show affected files before editing.
- Preserve existing patterns unless there is a strong reason not to.
- If a change requires a migration, new dependency, schema change, or
  security-sensitive change, stop and ask.

## Sensitive Areas

Based on inspection of this repository as of 2026-08-11:

- `README_module1.md` and the retrieval-method decision it documents
  (`dense_search` as production, `stage3_hybrid.py` /
  `stage4_rerank.py` retained for reference only) — do not silently
  reverse this without new evidence; see `DECISIONS.md`.
- `chunks.json` and `chroma_db/` — generated artifacts derived from
  `documents/`. Regenerating them is not free (embedding ~4,993 chunks
  took real compute time in a prior session) and overwriting them destroys
  reproducibility of past eval results unless intentional.
- `documents/` — source PDF input data; treat as read-only.
- `eval_set.json` — hand-written evaluation queries; the user has directly
  authored and edited this file across sessions. Do not regenerate or
  overwrite its contents programmatically.
- `requirements.txt` — pinned dependency versions; do not `pip freeze`
  over this file casually, since it reflects deliberately installed
  packages across multiple sessions.
- No secrets, credentials, `.env` files, auth code, payment code, or
  database migration files were found in this repository as of this
  inspection — if any are added later, they fall under the Hard Rules
  above by default.

## Proposed Constraints

The following are inferred from repository inspection and are **not**
active rules — they require user confirmation before being promoted to
Hard Rules:

- **RESOLVED (2026-08-12):** `.gitignore` now exists at repo root, covering
  `venv/`, `__pycache__/`, `*.pyc`, `chroma_db/`, `.env`, `*.log`,
  `.DS_Store`, and the regenerable `src/chunks.json` /
  `experiments/chunks_v2.json`. Note `documents/` is deliberately
  **tracked** (not gitignored), a change from the original proposal below,
  for reproducibility — the PDFs are small enough to check in and the repo
  is meant to be clonable and runnable end-to-end.
- PROPOSED: Do not run any code path that depends on a local Ollama server
  without first verifying it is installed and reachable (`ollama list`,
  or a request to `http://localhost:11434`) — Ollama was confirmed
  unavailable on this machine in a prior session (see `HANDOVER.md`).
- PROPOSED: Do not treat `stage3_hybrid.py` / `stage4_rerank.py` as dead
  code to delete — they are intentionally retained for
  reference/comparison per `README_module1.md`, even though unused in
  production.
- PROPOSED: Do not overwrite `requirements.txt` via a blanket
  `pip freeze > requirements.txt` without reviewing the diff first — the
  venv may accumulate transitive dependencies from packages installed for
  one-off experiments (e.g. `rank-bm25`) that should still be reviewed
  before locking in.
