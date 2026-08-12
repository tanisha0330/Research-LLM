# Module 4: Calibrated Confidence (Conformal Prediction)

## Concept

An LLM's own stated confidence (or a raw audit label) is not statistically
trustworthy on its own — nothing prevents a model from sounding equally
confident whether it's right or wrong. **Conformal prediction** offers a
different guarantee: instead of asking the model "how sure are you?", we
take a signal that correlates with correctness (a *non-conformity score* —
higher means "less trustworthy"), compute it on a held-out **calibration
set** where we already know the ground truth, and find the score threshold
that would have correctly captured a target fraction (here, 80%) of the
correct answers in that calibration set. That threshold is then applied to
new, unseen queries. The result is a **"high confidence" / "low
confidence — review recommended"** split with a coverage guarantee that is
grounded in the calibration data itself, not in anything the LLM claims
about its own certainty. This module (`stage4_conformal.py`) builds exactly
that pipeline on top of Module 3's audit output and Module 2's retrieval
scores.

## Non-Conformity Score Design

**`compute_confidence_signal(query, chunks, answer, audit_result)`**
produces a continuous score in `[0.0, 1.0]` (0.0 = most trustworthy, 1.0 =
least trustworthy), combining three signals on top of the audit-status base:

- **Base score from `audit_status`** (Module 3's red-team audit label):
  `skipped_metadata = 0.0`, `passed = 0.2`, `metadata_mismatch_corrected =
  0.5`, `flagged = 0.8`.
- **Similarity adjustment**: `(1 - top1_similarity) * 0.15` — weaker top-1
  retrieval evidence pushes the score higher, even when the audit didn't
  flag anything.
- **Spread adjustment**: `(1 - score_gap) * 0.15`, where `score_gap` is the
  top-1-minus-bottom-5 similarity spread — a *small* gap means all
  retrieved chunks look similarly (ir)relevant (no clearly strongest
  match), itself a sign of weak retrieval independent of the top score.
- **Length penalty**: a flat `0.05` for answers under 20 or over 150 words
  (both extremes correlated with lower accuracy in manual review);
  abstentions are exempt.
- `metadata_lookup` answers never run retrieval, so they get no
  similarity/spread adjustment (base score stands alone).
- Final score = `base + similarity_adj + spread_adj + length_penalty`,
  clipped to `[0.0, 1.0]`.

This is the second iteration of the score — see "First version" below for
why a single similarity adjustment wasn't added straight to production
without first trying (and rejecting) an audit-status-only design.

### First version: why the audit_status-only score failed

The initial version of this module used only the 3-bucket `audit_status`
mapping (`skipped_metadata`/`passed → 0.0`, `metadata_mismatch_corrected →
0.5`, `flagged → 1.0`) as the entire non-conformity score, with no
continuous adjustment. On the 14-query calibration set, 8 of the 14 scores
were exactly `1.0` ("flagged"), so the 80%-coverage quantile (order
statistic k=12 of 14) landed **exactly on the score ceiling of 1.0**. Since
no score in the dataset can exceed 1.0, **every test case automatically
qualified as "high confidence"** — the calibration produced a threshold
that discriminated nothing. The reported "87.5% empirical coverage" in that
version was really just the test set's raw accuracy rate, not evidence
that the calibration was doing useful filtering. This motivated the
redesign into a continuous score with the retrieval-similarity adjustment
term.

## Final Results

- **Calibration threshold: 0.9792** (no longer at the score ceiling —
  confirms the fix resolved the degenerate case).
- **Test set breakdown (14 unseen queries):** 5 labeled **high
  confidence**, 9 labeled **low confidence — review recommended**.
- **Empirical coverage: 5/5 = 100.0%** of high-confidence test cases were
  actually `llm_judge_correct = True`, against an 80% target — the
  coverage guarantee held on this split, with no misses at all. **Sample
  size is small at this split ratio (5 held-out high-confidence cases) —
  treat this as directionally consistent with the 80% target rather than a
  statistically precise measurement**, not as proof of a true 100% coverage
  rate going forward.
- **Discrimination confirmed:** the original degenerate version labeled
  0/14 test cases low confidence; this version labels 9/14 low confidence
  — the calibration is now genuinely separating cases rather than
  accepting everything.

These numbers were last re-measured after making k=12 retry-widening
permanent in `stage2_self_correct.py` (see
[README_module2.md](README_module2.md)'s Finding 4/5) — end-to-end
correctness rose from 20/28 to 25/28 `llm_judge_correct`, and coverage rose
from 83.3% (5/6) to 100.0% (5/5) as a direct consequence: the same 3-signal
scoring logic, run against a now-more-accurate answer set, classified one
fewer test case as high-confidence and got every one of those right. These
are the numbers on the current production pipeline (`dense_search`
retrieval — see [README_module1.md](README_module1.md)'s Decision Reversal
section for why a `hybrid_rerank_search` experiment was tried and reverted
here). Retrieval method changes shift these scores, since the similarity
and spread adjustments are computed from whatever retrieval method
`get_retrieval_chunks` calls — currently `dense_search`, independent of
whatever method actually produced the answer being scored.

## Known Limitation: Compressed Score Range Within `flagged`

Most `flagged` cases still cluster tightly (within roughly a 0.03 band —
narrower in relative terms than the score's full range, though wider in
absolute terms than the original 2-signal version's ~0.02 band). Since the
`audit_status` base score for `flagged` is fixed at 0.8 and most retrieved
evidence in this corpus has middling-high top-1 similarity, the majority of
`flagged` cases still end up close together. The calibration threshold
(0.9792) lands inside this cluster, meaning the high/low confidence split
for `flagged` cases is still driven by comparatively small differences in
similarity/spread rather than a clean substantive separation.

**Small sample size caveat:** with only 14 calibration and 14 test queries,
the exact 100.0% coverage figure (5/5 — a perfect score on this split, not
a guarantee) and the 0.9792 threshold itself should be treated as
illustrative, not statistically reliable. A single query flipping category
near the margin would materially shift both the threshold and the reported
coverage — this sample is too small to treat either number as a stable
estimate of true long-run coverage, and a 5-case high-confidence bucket is
if anything an even smaller sample than the 6/14 it replaced. (This
sensitivity was directly observed: the same 3-signal scoring logic
produced a 44.4% coverage reading when it was briefly run against
`hybrid_rerank_search`-sourced answers instead of `dense_search`-sourced
ones — see README_module1.md's Decision Reversal section.)

## Future Work

Retrieval score spread and answer length have already been folded into the
score (see Non-Conformity Score Design above) — the remaining known gap:

- **Cross-encoder rerank score** (from `stage4_rerank.py`, currently
  unused in the production pipeline but available) — a second, differently
  trained relevance signal that could decorrelate from the embedding
  similarity already in use, and further widen the `flagged` cluster.
- **Consistency between the confidence signal's retrieval method and the
  production answer's retrieval method** — `get_retrieval_chunks` always
  re-runs `dense_search` regardless of which method actually produced the
  answer. This was a deliberate simplification (the score was designed as
  an independent quality signal, not required to match the answering
  method) but it means the score's similarity/spread terms would silently
  go stale if production retrieval ever changed again without a
  corresponding update here — worth revisiting if Module 1's decision is
  ever reopened.

Combining several weakly-correlated signals rather than one dominant
categorical bucket has already reduced (though not eliminated) the
clustering seen in the `flagged` scores; a cross-encoder signal is the next
lever to pull for further discrimination.

## Module 4 Summary

| Component | File | Purpose |
|---|---|---|
| `compute_confidence_signal` | `stage4_conformal.py` | Continuous non-conformity score combining Module 3's `audit_status` with a top-1 retrieval-similarity adjustment |
| `compute_conformal_threshold` | `stage4_conformal.py` | Split-conformal calibration: computes the empirical quantile threshold at the target coverage level from the calibration set |
| Coverage evaluation | `stage4_conformal.py` (`main`) | Applies the calibrated threshold to the held-out test set and reports empirical coverage (fraction of high-confidence cases that are actually correct) |

**Total: 3 components built**, replacing an earlier degenerate 3-bucket
version of the non-conformity score with a continuous, threshold-sensitive
one.
