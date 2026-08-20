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

---

## 2026-08-21

- Model/version: Claude Opus 4.8 (`claude-opus-4-8`) and Claude Sonnet 5
  (`claude-sonnet-5`), via Claude Code CLI. Ollama (`llama3.1:8b`) was
  UNAVAILABLE at session start (per prior HANDOVER) but the user installed
  it mid-session and it was confirmed reachable at `localhost:11434` — so,
  unlike prior sessions, generation-dependent paths were actually run here.
- Task: Four threads, in order — (1) assessed whether a ColPali visual
  retrieval layer belongs in this project (recommendation: NO — sampled the
  real table chunks in `chunks.json`, financial tables extract cleanly via
  pypdf, and the eval set is ~90% prose/cover-page facts, so ColPali would
  solve a problem this corpus doesn't have; it's also a pipeline replacement,
  not a layer, and fights the "local & free" constraint). (2) Fixed genuinely
  fixable items. (3) Added harder adversarial eval queries. (4) Built a query
  routing / "action" layer so single-company questions never read other
  filings, while comparative questions deliberately fan out — then looped two
  methods (grounding on exact metadata, then per-company enumeration) to push
  comparative accuracy, hitting an honest 8B-model negation-reasoning ceiling.
- Decision: See `DECISIONS.md`'s three newest entries — bootstrap coverage
  CI; the query-routing layer; and grounding comparative metadata questions on
  exact values. Key honest finding recorded there: grounding fixed
  hallucination and completeness on comparative-metadata questions, but
  `llama3.1:8b` still flips its own correct reasoning on the hardest negation
  case — this is caught by the red-team audit (`flagged`), not emitted as
  trustworthy. "Loop until fully accurate" is not reachable with an 8B local
  model on negation/subjective reasoning; the honest lever is model
  capability, not more prompt tweaks. NOT changed: the scoped-out comparative
  *synthesis* quality problem, and single/ambiguous retrieval behavior
  (preserved byte-for-byte so Module 2-4 calibration numbers are unaffected).
- Work committed & pushed: branch `comparative-routing-and-tests`, commit
  `78ea70d`, pushed to `origin` (github.com/tanisha0330/Research-LLM). NOT
  merged to `main` — left as a branch for review/PR. The two untracked
  graph-extraction experiment files (`src/stage_graph_extract.py`,
  `eval/graph_triples.json`) are someone else's in-progress work and were
  deliberately excluded from the commit; they remain untracked.
- Files touched:
  - `src/stage2_self_correct.py` — `route_query`, `detect_companies`,
    `is_comparative`, `routed_dense_search`, `comparative_metadata_answer`,
    `_metadata_fields_in_query`; reordered `answer_with_self_correction` so
    comparative scope is handled by routing (grounded metadata else fan-out)
    and never by the single-company metadata fast-path. `run_dense_retrieval_
    flow` now takes a `route` instead of `filter_source`.
  - `src/stage3_redteam.py` — updated the one `run_dense_retrieval_flow`
    caller to build a route; added `route_query` import.
  - `src/stage4_conformal.py` — `bootstrap_coverage_ci` (90% CI on coverage);
    fixed stale hardcoded "28 cases" / "14 test cases" output counts.
  - `tests/test_pipeline.py` (new) — 28 stdlib-unittest contract tests, two
    tiers (Tier 2 skips cleanly without a built `chroma_db/`).
  - `.github/workflows/test.yml` (new) — runs the suite on push/PR.
  - `eval/eval_set_additions.json` — replaced duplicate content with 11 new
    adversarial queries (false-premise, out-of-corpus, attribution-trap,
    numeric extraction, comparative routing, negation, injection-resistance).
  - `docs/ai/DECISIONS.md`, `docs/ai/ARCHITECTURE.md` (test-tools line),
    `docs/ai/CONTEXT_LOG.md` (this entry).
- Verified (not assumed): 28/28 unit tests pass; the single-scope retrieval
  guard was checked against the LIVE ChromaDB index (returns only the target
  filing's chunks); comparative routing + grounded metadata were run
  end-to-end through `finalize_with_audit` against ground truth (Q5 correct;
  Q7 reasoning correct, conclusion negation-flipped and audit-flagged); Module
  4 numbers reproduce exactly (threshold 0.9820, 15/18 = 83.3%,
  CI [66.7%, 94.4%]).
- Open / next steps for a future session:
  - The 4 `graceful_decline`-tagged comparative queries in `eval_set.json`
    now trigger multi-document retrieval instead of an auto-decline — their
    `expected_behavior` should be revisited before they are re-scored end to
    end. The new comparative queries in `eval_set_additions.json` are staged,
    not yet merged into `eval_set.json` or scored.
  - Highest-leverage open item remains growing the hand-labeled eval set
    (n=24 test) — the bootstrap CI now makes that uncertainty visible.
  - If reliable comparative/negation reasoning is needed, evaluate a larger
    model for the comparative generation step (cost vs. "local & free"
    tradeoff) rather than more prompt tuning.
- Confidence: High for everything run and inspected directly this session
  (tests, live-index guard, end-to-end comparative runs, the push). Medium
  for the generalization of the single-example comparative findings — they
  are illustrative of the mechanisms, on a handful of queries, not a
  statistically sized eval.
