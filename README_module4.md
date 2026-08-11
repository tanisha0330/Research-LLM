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
least trustworthy):

- **Base score from `audit_status`** (Module 3's red-team audit label):
  `skipped_metadata = 0.0`, `passed = 0.2`, `metadata_mismatch_corrected =
  0.5`, `flagged = 0.8`.
- **Adjustment from top-1 retrieval similarity**: `(1 - top_similarity_score)
  * 0.2` — weaker retrieval evidence pushes the score higher (less
  trustworthy), even when the audit itself didn't flag anything.
  `metadata_lookup` answers never ran `dense_search`, so they get no
  adjustment (base score stands alone).
- Final score = `base + adjustment`, clipped to `[0.0, 1.0]`.

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

- **Calibration threshold: 0.8447** (no longer at the score ceiling —
  confirms the fix resolved the degenerate case).
- **Test set breakdown (14 unseen queries):** 8 labeled **high
  confidence**, 6 labeled **low confidence — review recommended**.
- **Empirical coverage: 7/8 = 87.5%** of high-confidence test cases were
  actually `llm_judge_correct = True`, against an 80% target — the
  coverage guarantee held on this split.
- **Discrimination confirmed:** the previous version labeled 0/14 test
  cases low confidence; this version labels 6/14 low confidence — the
  calibration is now genuinely separating cases rather than accepting
  everything.

## Known Limitation: Compressed Score Range

Most `flagged` cases cluster very tightly in the **0.83–0.85** range. Since
the `audit_status` base score for `flagged` is fixed at 0.8 and the
similarity adjustment only contributes up to 0.2, and most retrieved
evidence in this corpus has middling-high top-1 similarity, the resulting
scores for the majority of `flagged` cases end up bunched within a ~0.02
band. The calibration threshold (0.8447) landed **directly inside this
cluster**, meaning the high/low confidence split is effectively driven by
small, noisy differences in top-1 similarity rather than a clean
substantive separation between trustworthy and untrustworthy answers.

**Concrete example of the resulting misclassification:** the query *"What
does DocuSign say about expanding internationally or its global
operations?"* scored **0.5526** (a `metadata_mismatch_corrected` case) and
was labeled **high confidence** — but it was actually
`llm_judge_correct = False`. Because its base score (0.5) sat well below
the threshold (0.8447) regardless of the similarity adjustment, this case
was never close to being correctly flagged as low confidence, illustrating
that the coarse `audit_status` base score can still dominate and mask an
actual error, independent of the compressed-cluster issue among `flagged`
cases.

**Small sample size caveat:** with only 14 calibration and 14 test queries,
the exact 87.5% coverage figure (and the 0.8447 threshold itself) should be
treated as illustrative, not statistically reliable. A single query
flipping category near the margin would materially shift both the
threshold and the reported coverage — this sample is too small to treat
either number as a stable estimate of true long-run coverage.

## Future Work

The compressed score range points to a clear next step: add **independent,
lower-correlation signals** to widen the effective resolution of the
score, rather than relying almost entirely on the 4-value `audit_status`
base:

- **Retrieval score spread across all 5 chunks** (not just top-1) — a
  large or small gap between the top and lower-ranked chunk similarities
  may itself be informative about retrieval confidence.
- **Answer length** — very short or unusually long answers may correlate
  with under- or over-answering.
- **Cross-encoder rerank score** (from `stage4_rerank.py`, currently
  unused in the production pipeline but available) — a second, differently
  trained relevance signal that could decorrelate from the embedding
  similarity already in use.

Combining several weakly-correlated signals rather than one dominant
categorical bucket should reduce the clustering seen in the current
`flagged` scores and produce a smoother, more discriminative score
distribution.

## Module 4 Summary

| Component | File | Purpose |
|---|---|---|
| `compute_confidence_signal` | `stage4_conformal.py` | Continuous non-conformity score combining Module 3's `audit_status` with a top-1 retrieval-similarity adjustment |
| `compute_conformal_threshold` | `stage4_conformal.py` | Split-conformal calibration: computes the empirical quantile threshold at the target coverage level from the calibration set |
| Coverage evaluation | `stage4_conformal.py` (`main`) | Applies the calibrated threshold to the held-out test set and reports empirical coverage (fraction of high-confidence cases that are actually correct) |

**Total: 3 components built**, replacing an earlier degenerate 3-bucket
version of the non-conformity score with a continuous, threshold-sensitive
one.
