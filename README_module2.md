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
