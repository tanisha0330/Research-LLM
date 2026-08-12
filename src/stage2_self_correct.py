import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import ollama
from rapidfuzz import fuzz

from stage2_embed import dense_search
from stage_generate import GENERATION_MODEL, _collection, _embedding_model, critique_sufficiency, generate_answer

MAX_RETRIES_HARD_LIMIT = 1
METADATA_PATH = Path(__file__).parent / "company_metadata.json"
FUZZY_COMPANY_MATCH_THRESHOLD = 80
INITIAL_RETRIEVAL_K = 5
RETRY_RETRIEVAL_K = 12

_company_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

COMPANY_SOURCE_MAP = {
    "Atlassian": "attlasian-10k-2026.pdf",
    "DocuSign": "docusign-10k-2026.pdf",
    "HubSpot": "hubspot-10k-2026.pdf",
    "Salesforce": "salesforce-10k-2026.pdf",
    "Zoom": "zoom-10k-2026.pdf",
}


def detect_company(query: str) -> str | None:
    query_lower = query.lower()
    for company_name, source_file in COMPANY_SOURCE_MAP.items():
        if company_name.lower() in query_lower:
            return source_file

    # Fall back to fuzzy matching for misspelled company names (e.g.
    # "Atlasian", "Zom", "Salesforc's") that don't contain an exact
    # substring match above. Compare each word-token in the query against
    # each company name and take the best match if it clears the threshold.
    tokens = re.findall(r"[a-z]+", query_lower)
    best_source_file = None
    best_score = 0.0
    for token in tokens:
        for company_name, source_file in COMPANY_SOURCE_MAP.items():
            score = fuzz.ratio(token, company_name.lower())
            if score > best_score:
                best_score = score
                best_source_file = source_file

    if best_score >= FUZZY_COMPANY_MATCH_THRESHOLD:
        return best_source_file
    return None


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def try_metadata_lookup(query: str) -> dict | None:
    metadata_text = json.dumps(_company_metadata, indent=2)

    def build_prompt(strict: bool) -> str:
        format_instruction = (
            "Respond with ONLY a single valid JSON object and nothing else — "
            "no markdown code fences, no commentary before or after the JSON."
            if strict
            else "Respond with a JSON object."
        )
        return f"""Below is structured metadata extracted from the cover pages of several companies' SEC 10-K filings.

Metadata:
{metadata_text}

Question: {query}

Can this question be directly answered using ONLY this structured metadata? Do not use any outside knowledge.
- Be careful with ambiguous words. "Exchange" can mean a STOCK EXCHANGE (e.g., Nasdaq, NYSE — this IS in the metadata) or an EXCHANGE RATE (currency conversion — this is NOT in the metadata and requires narrative document content). Only answer "true" if the question is specifically about the structured fields provided: ticker symbol, stock exchange listing, state of incorporation, fiscal year end, office address, or EIN. If there is any ambiguity or the question involves financial/business narrative content, return answered: false so it can be properly researched.
- If yes, return {{"answered": true, "answer": "...", "source_company": "..."}} where "answer" is the direct answer to the question and "source_company" is the source_file key (e.g. "attlasian-10k-2026.pdf") the answer came from.
- If the question asks about multiple attributes (e.g., both ticker symbol and exchange), include all relevant fields from the metadata in your answer, not just one.
- The "answer" field must be a complete, natural-language sentence — never a dictionary, JSON object, or raw list of fields. For example, for a ticker+exchange question, write something like "Atlassian's stock ticker symbol is TEAM, listed on the Nasdaq Global Select Market." rather than a Python dict.
- If the question requires narrative/contextual information not present in this metadata, return {{"answered": false}}.

{format_instruction}

JSON:"""

    raw = ""
    for attempt, strict in enumerate((False, True)):
        prompt = build_prompt(strict)
        response = ollama.chat(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _extract_json(response["message"]["content"])

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict) or "answered" not in parsed:
            continue

        if parsed.get("answered") is True:
            return {
                "answered": True,
                "answer": parsed.get("answer"),
                "source_company": parsed.get("source_company"),
            }
        return {"answered": False}

    return None


def reformulate_query(original_query: str, reason: str) -> str:
    prompt = f"""The following search query did not retrieve enough information to answer the user's question.

Original query: {original_query}
Reason the previous attempt was judged insufficient: {reason}

Rewrite the query as a shorter, simpler, more keyword-focused version — similar to a search-engine query. Strip filler words like "what," "does," "which," "say about," "how," "is," "are," "the," "a," "an." Keep only the core entities and topic (e.g. company names, specific terms, numbers).

Return ONLY the reformulated query string, with no explanation, quotes, or extra text.

Reformulated query:"""

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip().strip('"')


def run_dense_retrieval_flow(query: str, max_retries: int = 1, filter_source: str = None) -> dict:
    max_retries = min(max_retries, MAX_RETRIES_HARD_LIMIT)

    chunks = dense_search(
        query, k=INITIAL_RETRIEVAL_K, model=_embedding_model, collection=_collection, filter_source=filter_source
    )
    answer = generate_answer(query, chunks)
    critique = critique_sufficiency(query, chunks, answer)

    if critique["sufficient"] is True:
        return {
            "final_answer": answer,
            "retries_used": 0,
            "query_used": query,
            "original_query": query,
            "reformulated_query": None,
            "sufficient": True,
            "critique_reason": critique["reason"],
            "source_method": "dense_retrieval",
        }

    if max_retries < 1:
        return {
            "final_answer": answer,
            "retries_used": 0,
            "query_used": query,
            "original_query": query,
            "reformulated_query": None,
            "sufficient": critique["sufficient"],
            "critique_reason": critique["reason"],
            "source_method": "dense_retrieval",
        }

    reformulated_query = reformulate_query(query, critique["reason"])

    # Retry uses a wider retrieval net (k=12 vs. the initial k=5) — a
    # 4-query survey found this resolves 3/4 single-hop recall misses with
    # zero regressions (see README_module2.md's Finding 4). It does not fix
    # true multi-hop synthesis failures (Findings 2-5 in the same doc), but
    # is a net win for the common single-hop retry case.
    chunks_2 = dense_search(
        reformulated_query, k=RETRY_RETRIEVAL_K, model=_embedding_model, collection=_collection,
        filter_source=filter_source,
    )
    answer_2 = generate_answer(reformulated_query, chunks_2)
    critique_2 = critique_sufficiency(reformulated_query, chunks_2, answer_2)

    return {
        "final_answer": answer_2,
        "retries_used": 1,
        "query_used": reformulated_query,
        "original_query": query,
        "reformulated_query": reformulated_query,
        "sufficient": critique_2["sufficient"],
        "critique_reason": critique_2["reason"],
        "source_method": "dense_retrieval",
    }


def answer_with_self_correction(query: str, max_retries: int = 1, filter_source: str = None) -> dict:
    """`filter_source` lets a caller pass an explicit company source_file
    (e.g. from a UI dropdown) to bypass detect_company()'s query-text
    inference entirely. If not provided, falls back to detect_company()."""
    metadata_result = try_metadata_lookup(query)
    if metadata_result is not None and metadata_result.get("answered") is True:
        return {
            "final_answer": metadata_result["answer"],
            "retries_used": 0,
            "query_used": query,
            "original_query": query,
            "reformulated_query": None,
            "sufficient": True,
            "critique_reason": "Answered directly from structured company metadata.",
            "source_method": "metadata_lookup",
            "source_company": metadata_result.get("source_company"),
        }

    if filter_source is None:
        filter_source = detect_company(query)
    return run_dense_retrieval_flow(query, max_retries=max_retries, filter_source=filter_source)


if __name__ == "__main__":
    test_queries = [
        "What is Atlassian's stock ticker symbol and which exchange is it listed on?",
        "What is DocuSign's IRS EIN?",
        "What does Zoom say about foreign currency exchange rate risk?",
        "What percentage of HubSpot's total revenue comes from Payments?",
        "What is Salesforce's principal office address?",
    ]

    for query in test_queries:
        result = answer_with_self_correction(query, max_retries=1)

        print("=" * 80)
        print(f"Query: {query}")
        print(f"Source method: {result['source_method']}")
        print(f"\nFinal answer:\n{result['final_answer']}")
        print(f"\nSucceeded (sufficient): {result['sufficient']}")
        print(f"Retries used: {result['retries_used']}")
        print()
