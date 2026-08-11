# Module 3: Adversarial Critique (Red-Team Audit)

## Design

Module 3 adds a single-pass adversarial audit layer (`stage3_redteam.py`)
on top of Module 2's `answer_with_self_correction`. It does not try to fix
or improve answers — it only checks and flags.

- **`redteam_critique(query, chunks, answer)`** — Uses a distinct persona
  from `critique_sufficiency` (Module 2): a *skeptical auditor*, not a
  sufficiency judge. Its instruction is deliberately narrow: find the
  **single weakest or most questionable claim** in the answer and check
  whether it is *directly and explicitly* supported by the retrieved
  excerpts — it does not evaluate the whole answer holistically. This is a
  **single pass, no loops, no revision cycles** — it produces one verdict
  and stops. Forces structured JSON output:
  `{"weakest_claim": "...", "is_supported": true/false, "explanation": "..."}`,
  with the same retry-once-with-stricter-prompt fallback used elsewhere in
  this codebase if the first response isn't valid JSON. If the answer is
  itself an abstention ("I don't have enough information..."), the
  abstention is treated as the claim being checked — it's judged
  "supported" if the excerpts genuinely lack the information.

- **`finalize_with_audit(query)`** — Orchestrates routing between full
  audit and a lighter check, depending on how Module 2 answered:
  - **`source_method == "dense_retrieval"`** → runs the full
    `redteam_critique` pass. Result is tagged `audit_status: "flagged"`
    (weakest claim found unsupported) or `"passed"` (weakest claim checked
    and found supported).
  - **`source_method == "metadata_lookup"`** → skips the full adversarial
    audit (structured cover-page fact lookups don't need claim-by-claim
    scrutiny) and instead runs a **lightweight sanity check**
    (`metadata_sanity_check`) — a single yes/no LLM call asking whether the
    returned answer actually addresses what was asked. If yes, tagged
    `audit_status: "skipped_metadata"`. If no, the query is re-routed
    through the real dense-retrieval flow instead of returning the
    mismatched answer, and the result is tagged
    `audit_status: "metadata_mismatch_corrected"` (see Bug Found below).

## Bug Found and Fixed: Metadata-Routing Misclassification

**The bug:** during initial testing, the query *"What does Zoom say about
foreign currency exchange rate risk?"* was incorrectly routed to
`metadata_lookup` and answered with *"Zoom's stock ticker symbol is ZM,
listed on The Nasdaq Global Select Market"* — a completely unrelated
answer. The word **"exchange"** in the query pattern-matched against the
`exchange` / `ticker_symbol` metadata fields (stock exchange), even though
the question was actually about **exchange *rate*** (currency conversion)
— an unrelated, narrative-content concept not present in the structured
metadata at all. Because `metadata_lookup` results skipped the audit
entirely under the original design, this bad answer had **no downstream
safety net** and would have been returned as final.

**The fix — two parts:**
1. **Tightened disambiguation in `try_metadata_lookup`'s prompt**
   (`stage2_self_correct.py`) — added an explicit instruction
   distinguishing STOCK EXCHANGE (in the metadata) from EXCHANGE RATE (not
   in the metadata, requires narrative content), with a directive to
   return `answered: false` whenever there is any ambiguity or the
   question involves financial/business narrative content.
2. **Added the `metadata_sanity_check` fallback** (`stage3_redteam.py`) —
   even when the prompt fix fails to prevent a misroute, a lightweight
   "does this answer actually address the query?" check now runs before a
   metadata answer is accepted. A "no" verdict triggers a fallback to the
   real `dense_retrieval` flow (via the newly extracted
   `run_dense_retrieval_flow` helper in `stage2_self_correct.py`) rather
   than returning the mismatched answer.

**Result after the fix:** re-testing the same Zoom currency query showed
the prompt tightening alone was sufficient — it now routes directly to
`dense_retrieval` without needing the fallback to trigger. The fallback
remains in place as a backstop for future ambiguous cases the prompt fix
doesn't anticipate, since metadata routing is a **defense-in-depth**
concern, not a one-shot fix.

## Key Finding: LLM Generation Non-Determinism

A 6-run stability test was run on the query *"What percentage of HubSpot's
total revenue comes from Payments?"* — a question for which **no answer
exists anywhere in the corpus** (the filings only break out "Subscription"
and "Professional services and other" revenue, never a "Payments"
category).

- **5 of 6 runs correctly abstained**, returning "I don't have enough
  information to answer this question based on the provided excerpts" (or
  a close paraphrase).
- **1 of 6 runs fabricated an answer** — computing a 98.4% / 2.2% revenue
  split by relabeling "Professional services and other" as if it were
  "Payments" revenue, a category the source excerpts never establish.

**This is inherent LLM run-to-run variance, not a fixable code bug.** No
code path changed between runs; the prompt, retrieved chunks, and routing
logic were all identical each time. The same query, same context, and same
model produced a materially different (and factually wrong) output roughly
1 time in 6. This directly motivates **Module 4's calibrated confidence
work**: a single generation pass — however well-prompted against
hallucination — cannot be trusted at face value on its own. What's needed
is a statistically grounded confidence signal (e.g., sampling multiple
generations and measuring agreement) rather than relying on any one pass,
however good the prompt, to be reliable every time.

## Effectiveness of the Red-Team Audit

When the fabrication occurred, `redteam_critique` **caught it** — flagging
the unsupported percentage claim with a specific, accurate explanation
noting the excerpts don't establish a "Payments" revenue category and the
cited breakdown wasn't directly supported. Across the observed sample, the
red-team audit's catch rate on actual fabrications is **1/1 (100%, small
sample)**. This demonstrates the audit layer functions as an effective
safety net precisely in the scenario it was designed for: **the underlying
generation is unreliable, but the audit step is currently the mechanism
that catches it before a bad answer would otherwise be reported as final.**
The sample size is too small to treat "100%" as a guarantee — it is a
report-only flag today, not a hard gate, and a larger stability study would
be needed before relying on it as a guaranteed catch-all.

## Module 3 Summary

| Component | File | Purpose |
|---|---|---|
| `redteam_critique` | `stage3_redteam.py` | Single-pass adversarial audit: finds and checks the one weakest claim in a `dense_retrieval` answer |
| `finalize_with_audit` | `stage3_redteam.py` | Orchestrates Module 2 → routing to full audit, lightweight metadata sanity check, or corrective fallback |
| `metadata_sanity_check` | `stage3_redteam.py` | Lightweight yes/no check that a metadata-sourced answer actually addresses the query; triggers fallback to dense retrieval on mismatch |
| `run_dense_retrieval_flow` | `stage2_self_correct.py` | Extracted/reusable dense-retrieval-with-retry flow, shared by Module 2's normal path and Module 3's metadata-mismatch fallback |

**Total: 4 components built** (1 new module-2 refactor to support module 3's fallback, 3 new module-3 functions).

### Final status of the 4 original test queries (post-fix)

| Query | source_method | audit_status | Outcome |
|---|---|---|---|
| Zoom foreign currency exchange rate risk | dense_retrieval (fixed — was metadata_lookup pre-fix) | flagged | Correct routing; audit flagged one soft overreach in an otherwise accurate, cited answer |
| Atlassian ticker symbol & exchange | metadata_lookup | skipped_metadata | Unchanged, correct — no regression |
| HubSpot % revenue from Payments | dense_retrieval | flagged (in the run examined here) | Correctly identified as unsupported when fabrication occurred; see stability test below for full picture |
| Salesforce generative AI / Agentforce | dense_retrieval | flagged | Mostly accurate, cited answer; audit flagged a paraphrase-vs-belief nuance ("aims to change" vs. source's "we believe will change") |

### 6-run stability test — HubSpot Payments query

| Outcome | Count |
|---|---|
| Correctly abstained | 5 / 6 |
| Fabricated an unsupported answer | 1 / 6 |
| Fabrications caught by red-team audit | 1 / 1 (100%, small sample) |
| Fabrications missed by red-team audit | 0 / 1 |

**Conclusion:** the metadata-routing bug is fixed at the prompt level, with
a fallback safety net in reserve. The bigger open risk exposed by this
module is generation-level non-determinism, not routing — which Module 3's
audit currently catches when it happens, but which is only detectable
after the fact on a single pass. That gap is the explicit motivation for
Module 4.
