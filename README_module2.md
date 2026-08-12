# Module 2: Self-Correcting RAG (Answer Generation)

## Pipeline Overview

Module 2 builds an LLM answer-generation layer on top of Module 1's
`dense_search` retrieval, using a local Ollama model (`llama3.1:8b`), with
a self-correction retry loop on top.

1. **`generate_answer(query, chunks)`** (`stage_generate.py`) — Builds a
   prompt from the query and retrieved chunks (cited by `source_file` and
   `page_number`) and calls the local LLM. The instruction set was
   iteratively strengthened after observing hallucination in testing:
   - Answer using ONLY the information in the provided excerpts.
   - Cite every fact with its source file and page number.
   - Explicitly forbid referencing external websites, URLs, or sources
     (e.g. Yahoo Finance, Bloomberg, Wikipedia) not present in the
     excerpts.
   - Do not guess or fabricate values, tickers, or facts not explicitly
     present in the excerpts.
   - Read through ALL provided excerpts individually before concluding
     information is missing — added after an earlier version of the
     prompt overcorrected into under-answering (declaring "insufficient
     information" without checking every chunk).
   - Only say "I don't have enough information..." after every excerpt has
     been checked and none contain the answer.

2. **`critique_sufficiency(query, chunks, answer)`** (`stage_generate.py`)
   — A second LLM call that judges whether the generated answer is
   well-supported by the retrieved chunks, forcing structured JSON output
   (`{"sufficient": true/false, "reason": "..."}`). The prompt instructs
   the model to judge sufficiency by *meaning*, not exact wording — an
   answer should not be marked insufficient just because the excerpts
   phrase the same fact differently than expected. If the model's first
   response isn't valid JSON, the function retries once with a stricter
   "JSON only, no commentary" instruction before giving up and returning
   `{"sufficient": None, "reason": "..."}`.

3. **`reformulate_query(original_query, reason)`** and
   **`answer_with_self_correction(query, max_retries=1)`**
   (`stage2_self_correct.py`) — The self-correction loop:
   - Run `dense_search` → `generate_answer` → `critique_sufficiency` on the
     original query.
   - If `sufficient` is `True`, return immediately (0 retries used).
   - If `sufficient` is `False`, `reformulate_query` rewrites the query
     into a shorter, keyword-focused, search-engine-style version (filler
     words like "what," "does," "which" stripped), then the full
     retrieve → generate → critique cycle runs once more on the
     reformulated query.
   - The loop is **hard-capped at 1 retry** (`MAX_RETRIES_HARD_LIMIT = 1`
     enforced in code, independent of the `max_retries` argument) — it
     never loops more than once, win or lose.

4. **`try_metadata_lookup(query)`** (`stage2_self_correct.py`) — A
   fast-path check that runs **before** `dense_search` is ever invoked.
   - **What it is:** a structured extraction of cover-page facts —
     `company_name`, `ticker_symbol`, `exchange`, `state_of_incorporation`,
     `fiscal_year_end`, `principal_office_address`, `irs_ein` — pulled once
     per document by `extract_metadata.py` (via the local LLM reading only
     the first 2 pages of each PDF) and cached in `company_metadata.json`,
     keyed by `source_file`.
   - **Why it was added:** `dense_search` consistently and reproducibly
     failed to retrieve short, structured/tabular cover-page text — most
     notably the "TEAM Nasdaq Global Select Market" line — no matter how
     the query was phrased or how the corpus was chunked (see the Known
     Limitations section, now resolved). Since these facts are static,
     enumerable, and cheap to extract once per document, a lookup table
     sidesteps the embedding-similarity problem entirely instead of trying
     to out-tune it.
   - **How routing works:** `answer_with_self_correction` calls
     `try_metadata_lookup(query)` first. The LLM is shown the full
     `company_metadata.json` contents and the question, and asked whether
     the question can be answered directly from that structured data alone
     (no outside knowledge). If yes, it returns
     `{"answered": true, "answer": "...", "source_company": "..."}` and the
     function returns immediately, tagged `"source_method": "metadata_lookup"`
     — retrieval, generation, and critique are skipped entirely. If the
     question needs narrative/contextual information not present in the
     metadata, it returns `{"answered": false}` and execution falls through
     to the existing `dense_search` → `generate_answer` →
     `critique_sufficiency` → retry flow, tagged
     `"source_method": "dense_retrieval"`.
   - **Answer quality refinements made after initial testing:** the prompt
     was tightened twice after review — once to require that *all*
     requested fields be included when a question asks about multiple
     attributes (e.g., both ticker and exchange, not just one), and once to
     require the `"answer"` field be a complete natural-language sentence
     rather than a raw Python dict or JSON fragment.

## Test Results

**HubSpot Payments query** — `"What percentage of HubSpot's total revenue
comes from Payments?"` (no answer exists anywhere in the corpus):
- First attempt: correctly abstains — *"I don't have enough information to
  answer this question based on the provided excerpts."*
- After reformulation and retry (`"HubSpot Payments revenue percentage"`):
  **still correctly abstains**, with the critique confirming *"There is no
  mention of 'HubSpot Payments' or a specific revenue percentage for this
  category."*
- **No hallucination at any stage**, including after being pushed through
  a full retry cycle — the abstention behavior is stable under retry
  pressure, not just on the first pass.

**Atlassian ticker query** — `"What is Atlassian's stock ticker symbol and
which exchange is it listed on?"` (the answer — "TEAM, Nasdaq Global
Select Market" — genuinely exists on page 1 of the source PDF):
- First attempt: `dense_search` does not retrieve the page-1 chunk
  containing the answer; generation correctly declines to guess rather
  than hallucinate.
- Reformulated query (`"Atlassian stock ticker symbol exchange listed
  on"`): retrieval still misses the correct chunk, this time surfacing a
  page-84 stock-performance table instead; the critique correctly flags
  the resulting answer as unsupported.
- **The retry/reformulation mechanism itself works exactly as designed** —
  it reformulates sensibly, avoids infinite looping, and the critique
  correctly identifies both attempts as insufficient rather than accepting
  a weak answer. But because the underlying retrieval never surfaces the
  correct chunk on either attempt, self-correction at the generation layer
  cannot succeed here — this is a retrieval-layer ceiling, not a
  generation-layer bug.

## Known Limitation: Atlassian Ticker/Exchange Retrieval Miss — RESOLVED via metadata fast-path routing

The "TEAM / Nasdaq Global Select Market" chunk on page 1 of
`attlasian-10k-2026.pdf` was **never retrieved by `dense_search` in its
top-5 results** for natural phrasings of the ticker/exchange question, and
this was **not resolved by three separate investigation approaches**:

1. **Query reformulation** (this module) — rewriting the query into a
   shorter, keyword-focused form did not surface the correct chunk; the
   reformulated query still missed it, instead surfacing a different wrong
   chunk (page 84's stock-performance table).
2. **Smaller chunking** — re-ingesting with `chunk_size=200` /
   `chunk_overlap=30` (vs. production `500`/`50`) to test whether the
   correct chunk was losing a similarity contest against longer, denser
   chunks. The correct chunk still did not appear in the top-5 even with
   much smaller, more granular chunks — the length disparity between the
   correct chunk (429 chars) and the incorrectly-winning competitor (480
   chars) was only 51 characters to begin with, so this result is
   consistent with chunk length not being the driver.
3. **Stability testing** — repeated evaluation runs confirmed the
   retrieval behavior is deterministic (not a flaky/non-reproducible
   result); the miss happens consistently, not intermittently.

**Diagnosis:** since neither rephrasing the query nor changing the
chunking granularity fixed the miss, and the behavior was stable/repeatable
rather than noisy, this pointed to an **embedding-model-level limitation**
rather than a fixable bug in the pipeline. `BAAI/bge-small-en-v1.5` appears
to represent short, structured/tabular text (a single line like "Class A
Common Stock, par value $0.00001 per share TEAM Nasdaq Global Select
Market") less discriminatively than prose-style narrative text, causing it
to lose the similarity contest against longer, narrative chunks that merely
mention related terms (e.g., "Nasdaq Composite," stock price tables).

**Resolution:** rather than continuing to tune retrieval or the embedding
model, the limitation was resolved **architecturally, not by fixing dense
retrieval itself** — `try_metadata_lookup` routes structured fact-lookup
queries (ticker, exchange, EIN, address, fiscal year end, etc.) around
`dense_search` entirely, answering them from a small pre-extracted metadata
table instead. Re-testing the exact same query
(`"What is Atlassian's stock ticker symbol and which exchange is it listed
on?"`) now returns `"Atlassian's stock ticker symbol is TEAM, listed on the
Nasdaq Global Select Market."` via `source_method: "metadata_lookup"` —
`dense_search`'s underlying weakness on short structured text is still
present and unaddressed at the embedding level, but it no longer matters
for this class of question because the query never reaches it.

## Decision

**Retain `chunk_size=500` / `chunk_overlap=50` (the original v1 chunking
configuration) as the production setting.** A v2 variant with smaller
chunks (`chunk_size=200`/`chunk_overlap=30`) was evaluated as a targeted
fix for the Atlassian ticker case and:
- did **not** fix the target case (the correct chunk still didn't appear
  in the v2 top-5), and
- **reduced overall retrieval performance** across the full 28-query eval
  set: hit_rate@5 dropped from 1.000 to 0.964 (**-3.6 pts**), and
  precision@1 dropped from 0.893 to 0.750 (**-14.3 pts**).

Since v2 traded away real, broad-based performance for no gain on the one
case it targeted, v1's chunking configuration remains in production.

## Note on this Module's Scope

`critique_sufficiency`'s current implementation uses an explicit
instruction ("judge by meaning, not exact wording") rather than worked
few-shot examples. In practice this instruction alone was not fully
sufficient — a follow-up test still saw the critique model flag a
correctly-supported answer as insufficient over wording pedantry. If this
proves to be a recurring issue, adding concrete few-shot examples to the
critique prompt (rather than relying on instruction alone) is the natural
next step, but that change has not yet been made in code.

## Known Limitation: Multi-Hop Reasoning (VFT Case Study)

**The test case:** *"Explain the financial relationship between Atlassian
and the Vertical First Trust (VFT). Following the completion of the
Australian HQ Property development by VFT, what exact long-term financial
commitments is Atlassian obligated to fulfill?"* — a genuinely multi-hop
question. Answering it correctly requires connecting two different sections
of the same filing: **Note 4 ("Investments")**, which describes VFT as an
equity-method investment tied to the Australian HQ Property construction,
and **Note 9 (leases)**, which states the resulting **$912.3 million**
future lease payment commitment over a fifteen-year initial term. Neither
section alone answers the question; the answer only exists at their
intersection.

**Finding 1 — k=5 (default): insufficient recall.** The first attempt
retrieved the VFT/equity-investment chunk (page 142) but not the lease
commitment chunk (page 148). `critique_sufficiency` correctly caught this —
it flagged the resulting answer as insufficient because it described the
VFT relationship but had no concrete commitment figure to cite. Retrieval
recall was the bottleneck at this k, and Module 2's self-correction loop
did exactly what it's designed to do: detect the gap rather than accept a
vague answer.

**Finding 2 — k=12 (widened retry, tested but not made permanent): recall
fixed, generation got worse.** As a bounded, non-permanent test (not a
production change — see the top-level README's "Known Limitations &
Lessons Learned"), the retry step's retrieval `k` was raised from 5 to 12
for this one query. Both target chunks were now retrieved — including the
page-148 lease chunk, which **explicitly cross-references the other
section by name** ("Please refer to Note 4, 'Investments,' for details of
the transaction"), about as strong a linking signal as retrieval could
hand the generator. But `generate_answer` on this wider 12-chunk context
did not connect them — it returned a flat *"I don't have enough
information to answer this question based on the provided excerpts,"* a
**regression from the k=5 attempt**, which had at least produced a
partial, honestly-hedged answer. Of the 12 retrieved chunks, only 2 were
actually relevant (VFT page 142, lease page 148); the other 10 were
unrelated financial-statement boilerplate, SEC filing headers, and
unrelated risk-factor/customer-growth paragraphs. The working theory is
that this noise-to-signal ratio pushed the deliberately conservative
anti-hallucination prompt (see Pipeline Overview above) toward abstention
rather than toward synthesizing the two relevant threads out of a mostly
irrelevant context.

**Finding 3 — retry-only generation-prompt fix (tested, not made
permanent): also did not resolve it.** As a second, independent bounded
test, one instruction was added to the retry attempt's prompt only (never
the first attempt, to avoid touching already-working simple-query
behavior): *"If one excerpt explicitly references another section or note
by name ... connect them explicitly in your answer rather than treating
them as unrelated. Do not abstain if the combination of two or more
excerpts together answers the question, even if no single excerpt does
alone."* This was re-run against the **exact same k=12 evidence set** as
Finding 2 — confirmed identical, with both the page-142 VFT chunk and the
page-148 $912.3M lease chunk present in the 12-chunk pool — to isolate the
prompt as the only variable. The result was unchanged: the answer was
still a flat *"I don't have enough information to answer this question
based on the provided excerpts."* A separate regression check confirmed
the retry-only scoping worked as intended: 3 already-passing queries (Zoom
foreign currency risk, HubSpot go-to-market approach, Salesforce
generative AI/Agentforce) all still succeeded on the first attempt with no
retry triggered, so the prompt change — even though it didn't fix the
target case — also didn't touch anything it wasn't supposed to.

**Refined root cause:** this is *not* simply "the model found both facts
but declined to connect them" — if that were the failure mode, an explicit
"connect cross-referencing excerpts" instruction should have helped, and
it didn't, at all, on identical evidence. The more accurate diagnosis is
that **the model's synthesis breaks down before it reliably locates the 2
sparse relevant chunks among the other 10 irrelevant ones** in a noisy
12-chunk context — a connection instruction is irrelevant if the two
things to connect are never surfaced as relevant to begin with. This is a
finding in its own right: it locates the failure earlier in the pipeline
(attention/relevance-filtering over a noisy context) than the original
hypothesis (a synthesis/connection step that has the right facts in view
but doesn't combine them).

**Finding 4 — 4-query survey across `eval_set.json`: the noisy-context
failure is specific to true multi-hop synthesis, not a general risk of
k=12 widening.** To check whether VFT's generation regression was
representative of retry cases generally or an unusually hard outlier, 4
other narrative queries that also trigger the retry path (`sufficient =
False` on the first k=5 attempt) were surveyed at k=12:

| Query | k=12 result |
|---|---|
| HubSpot % revenue from "Payments" | still insufficient — **correctly** abstains, since this answer genuinely does not exist anywhere in the corpus (the filings only break out "Subscription" and "Professional services and other" revenue) |
| DocuSign revenue generation (majority) | sufficient — found "98% subscription revenue" on page 5 |
| DocuSign typical subscription contract length | sufficient — found "one to three years" on page 75 |
| Salesforce sustainable growth / operating expenses approach | sufficient — synthesized content from pages 53 and 63 into a full answer |

**3/4 resolved cleanly; 1/4 correctly still abstained (not a failure — the
answer isn't in the corpus).** Critically, all 3 that resolved were
**single-hop recall misses**: the answer existed as one self-contained
fact/sentence on a single page that simply hadn't made the k=5 cutoff (as
opposed to VFT, which required combining two facts from two different
pages into one answer). In every one of these 3 cases, most of the 12
retrieved chunks were still irrelevant noise (e.g., only 1 of 12 chunks
was relevant for the DocuSign contract-length query) — yet generation
still succeeded cleanly, with **no abstention and no regression**, despite
a noise ratio at least as bad as VFT's.

**Revised conclusion: the noisy-context/synthesis breakdown seen in the
VFT case is NOT a general risk of k=12 widening — it is specific to cases
requiring true multi-hop synthesis** (combining facts from two separate
sections into one answer). Single-hop recall misses, even with 11/12
candidate chunks being noise, resolve cleanly at k=12 with no generation
degradation. This narrows and sharpens Findings 1–3 above: the problem was
never "wider context confuses the model," it's specifically "wider context
does not help the model combine two separate facts it needs to connect."

**Final, precise takeaway:** k=12 widening on retry is a **safe, low-risk
fix for single-hop recall misses** (verified on 3/4 retry cases in this
eval set, 0/4 regressions) and should be considered for permanent adoption
in the retry path. **True multi-hop synthesis (2+ sections combined)
remains an unresolved limitation** requiring either cross-reference-aware
retrieval or a reranking step — this is a narrower, better-scoped gap than
initially thought. This class of question — reasoning across two
non-adjacent sections of a single document — was deliberately scoped out
of this project from the start; see the top-level README's "What I'd Build
Next" section, which lists GraphRAG (entity/relationship graph over the
filings) and multi-agent debate as the originally-scoped-out mechanisms
for exactly this kind of cross-section, multi-hop reasoning. This case
study is evidence *for* that original scoping decision, not a regression
introduced by anything built so far.

### Future Work

**k=12 widening on retry is now a permanence candidate for the general
retry path** (see Finding 4) — this is a separate, smaller decision from
the multi-hop-specific fixes below, and has not yet been made permanent in
`stage2_self_correct.py` pending explicit confirmation.

For the remaining, narrower **multi-hop synthesis** gap specifically: two
low-effort fixes were tested and **both failed independently** — widening
retrieval alone (Finding 2) and a retry-only generation-prompt instruction
(Finding 3). A single explicit instruction telling the model to connect
cross-referencing excerpts is not enough on its own, because the evidence
points to the failure happening upstream of "connect the facts" (the model
not reliably surfacing 2 relevant chunks out of 12 when they must be
*combined*, even though single relevant chunks are found fine at the same
noise ratio per Finding 4). A more effective fix would likely need one or
both of:
- **A second-pass rerank/narrowing step before generation** — take the
  12 widened candidates and rerank or filter them down to the 2–3 most
  relevant (e.g., via the cross-encoder in `stage4_rerank.py`, currently
  unused in the production pipeline — see README_module4.md's Future
  Work) before handing them to `generate_answer`, rather than handing the
  LLM all 12 raw chunks and hoping it locates the relevant 2 unassisted.
- **Explicit cross-reference-aware retrieval** — detect "see Note X" /
  "refer to the Y section" style references in a first-pass chunk and
  directly pull in the referenced section as a second retrieval step,
  rather than relying on a single wider similarity search to accidentally
  include both halves of the answer.

Both were out of scope for this investigation and remain unimplemented.
Blanket k-widening on every retry is not recommended as a standalone fix —
tested here on one case, it solved recall but did not solve the underlying
problem, while adding real latency/compute cost (12-chunk retrieval and a
larger generation context) to every retry regardless of whether that
particular query is multi-hop at all. Given that two independent,
plausible fixes were tried and neither worked, the most honest framing is
that this remains a genuinely unresolved limitation of the current
architecture, not a small tuning gap — consistent with why multi-hop
reasoning (GraphRAG, multi-agent debate) was scoped out of this project
from the start rather than attempted as a bolt-on fix.

## Module 2 Summary

| Component | File | Purpose |
|---|---|---|
| `generate_answer` | `stage_generate.py` | Answers a query from retrieved chunks only, with anti-hallucination and full-excerpt-review instructions |
| `critique_sufficiency` | `stage_generate.py` | LLM-as-judge check of whether an answer is well-supported by its chunks, forced structured JSON output |
| `reformulate_query` | `stage2_self_correct.py` | Rewrites an insufficiently-answered query into a shorter, keyword-focused form for retry |
| `answer_with_self_correction` | `stage2_self_correct.py` | Orchestrates metadata fast-path → dense retrieval → generation → critique → single reformulated retry |
| `try_metadata_lookup` | `stage2_self_correct.py` | Fast-path structured lookup over pre-extracted cover-page facts, bypassing retrieval when possible |

**Total: 5 components built.**

### Final routing test — all 5 cases pass as expected

| Query | Expected routing | Actual `source_method` | Result |
|---|---|---|---|
| Atlassian ticker symbol & exchange | metadata_lookup | metadata_lookup | ✅ "Atlassian's stock ticker symbol is TEAM, listed on the Nasdaq Global Select Market." |
| DocuSign IRS EIN | metadata_lookup | metadata_lookup | ✅ "91-2183967" |
| Salesforce principal office address | metadata_lookup | metadata_lookup | ✅ Natural-language sentence with full address |
| Zoom foreign currency exchange rate risk | dense_retrieval (narrative) | dense_retrieval | ✅ Correct cited narrative answer |
| HubSpot % revenue from Payments | dense_retrieval (narrative, no answer exists) | dense_retrieval | ✅ Correctly abstains, even after retry — no hallucination |

All 5 test cases route correctly and produce the expected outcome. The
metadata fast-path answers structured cover-page questions immediately and
correctly (including the previously-unsolvable Atlassian ticker case),
while narrative/contextual questions still correctly fall through to the
full dense retrieval + self-correction pipeline, with abstention behavior
holding up under retry pressure.
