# AI Context Log

> A log of AI model/version context across sessions. Append a new entry at
> the end of every AI-assisted session — do not edit past entries except to
> fix factual errors.

## Format

```
## [Date]

- Model/version: [e.g. Claude Sonnet 5 / claude-sonnet-5]
- Task: [one-line description]
- Decision: [what was decided, if anything — cross-reference DECISIONS.md]
- Files touched: [list]
- Confidence: [high / medium / low, with a short reason]
```

---

## 2026-08-11

- Model/version: Claude Sonnet 5 (`claude-sonnet-5`), via Claude Code CLI
- Task: Set up the `docs/ai/` AI-collaboration documentation system
  (`HANDOVER.md`, `DECISIONS.md`, `FLOW.md`, `ARCHITECTURE.md`,
  `CONSTRAINTS.md`, `TEST_CHECKLIST.md`, `ROLLBACK.md`, `CONTEXT_LOG.md`,
  plus `bugs/.gitkeep` and `features/.gitkeep`). No application source code
  was modified; no dependencies were installed.
- Decision: No new technical decisions were made in this session. Existing
  decisions from prior sessions (dense-only retrieval as production; retain
  `stage3_hybrid.py`/`stage4_rerank.py` for reference) were **documented**
  in `DECISIONS.md`, not newly decided here.
- Files touched:
  - `docs/ai/HANDOVER.md` (created, then revised per updated template)
  - `docs/ai/ARCHITECTURE.md` (created)
  - `docs/ai/CONSTRAINTS.md` (created)
  - `docs/ai/TEST_CHECKLIST.md` (created)
  - `docs/ai/FLOW.md` (created)
  - `docs/ai/DECISIONS.md` (created)
  - `docs/ai/ROLLBACK.md` (created)
  - `docs/ai/CONTEXT_LOG.md` (created — this file)
  - `docs/ai/bugs/.gitkeep` (created)
  - `docs/ai/features/.gitkeep` (created)
- Confidence: Medium-high for factual claims directly backed by reading
  repo files (`README_module1.md`, the stage scripts, `requirements.txt`,
  `git status`/`git log` output). Low/UNVERIFIED for anything about
  intended future scope, tooling choices without recorded reasoning, or
  the state of Ollama on this machine beyond what was checked in a prior
  session — all such items are explicitly marked UNVERIFIED or TODO in the
  relevant files rather than asserted as fact.

---

## 2026-08-12

- Model/version: Claude Sonnet 5 (`claude-sonnet-5`), via Claude Code CLI
- Task: (1) A retrieval-method reversal-and-revert: switched production
  retrieval to `hybrid_rerank_search` based on LLM-judge content-relevance
  evidence, found it regressed end-to-end correctness and conformal
  coverage, and reverted to `dense_search`. (2) Portfolio polish: repo
  reorganization (`src/`/`eval/`/`reports/`/`experiments/`), `.gitignore`,
  `LICENSE` (MIT, Tanisha Jaiswal), top-level `README.md` rewrite with a
  "Results at a Glance" table and a "Known Limitations & Lessons Learned"
  section, module README consistency pass, and a live smoke test.
- Decision: See `DECISIONS.md`'s two newest entries (reversal-and-revert;
  repository reorganization).
- Files touched: `src/stage2_self_correct.py`, `src/stage3_redteam.py`,
  `src/report_schema.py` (retrieval revert); every moved script (import/
  path fixes); `README.md`, `README_module1.md`, `README_module4.md`
  (numbers + new sections); `.gitignore`, `LICENSE` (new);
  `docs/ai/ARCHITECTURE.md`, `docs/ai/DECISIONS.md`, `docs/ai/HANDOVER.md`,
  `docs/ai/ROLLBACK.md` (this reorg/decision documented).
- Confidence: High — the revert was confirmed by re-running the full
  28-query calibration build and conformal calibration (not just read
  from prior output), and the reorg was confirmed by a live smoke test
  (`src/stage_final_report.py`, 3 queries, all succeeded, output inspected
  directly) rather than assumed from the edits alone.
