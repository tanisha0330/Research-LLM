import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

import ollama

from stage2_embed import dense_search
from stage2_self_correct import answer_with_self_correction, detect_company, route_query, run_dense_retrieval_flow
from stage_generate import GENERATION_MODEL, _collection, _embedding_model, build_context


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def redteam_critique(query: str, chunks: list, answer: str) -> dict:
    context = build_context(chunks)

    def build_prompt(strict: bool) -> str:
        format_instruction = (
            "Respond with ONLY a single valid JSON object and nothing else — "
            "no markdown code fences, no commentary before or after the JSON."
            if strict
            else "Respond with a JSON object."
        )
        return f"""You are a skeptical auditor reviewing an AI-generated answer for accuracy.

Your only job is to find the SINGLE weakest or most questionable claim in this answer and check whether it is directly and explicitly supported by the provided excerpts. Do not evaluate the whole answer — focus on the one claim most likely to be wrong, overstated, or unsupported.

If the answer states that there is insufficient information to answer the question (i.e. it makes no substantive factual claim), treat that abstention itself as the "claim" being checked, and judge whether the excerpts genuinely lack the information (in which case the abstention is supported).

Query: {query}

Excerpts:
{context}

Answer:
{answer}

{format_instruction}
Return a JSON object with exactly these keys:
{{"weakest_claim": "the single weakest or most questionable claim, quoted or paraphrased from the answer", "is_supported": true or false, "explanation": "a short explanation of why this claim is or is not directly and explicitly supported by the excerpts"}}

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

        if isinstance(parsed, dict) and {"weakest_claim", "is_supported", "explanation"} <= parsed.keys():
            return {
                "weakest_claim": parsed["weakest_claim"],
                "is_supported": bool(parsed["is_supported"]),
                "explanation": parsed["explanation"],
            }

    return {
        "weakest_claim": None,
        "is_supported": None,
        "explanation": f"Red-team critique did not return valid JSON after 2 attempts. Last raw output: {raw!r}",
    }


def metadata_sanity_check(query: str, answer: str) -> bool:
    prompt = f"""Does this answer actually address what was asked in the original query?

Query: {query}
Answer: {answer}

Respond with exactly one word: "yes" or "no". Do not explain, do not add punctuation or any other text."""

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"].strip().lower().strip(".")
    return raw.startswith("yes")


def _run_redteam_pass(query: str, result: dict, filter_source: str = None) -> dict:
    # Re-run the retrieval used for the final query (with the same company
    # filter self-correction would have applied) so the critique has the
    # same chunks the last generate_answer call actually saw.
    if filter_source is None:
        filter_source = detect_company(query)
    chunks = dense_search(
        result["query_used"], k=5, model=_embedding_model, collection=_collection, filter_source=filter_source
    )

    critique = redteam_critique(query, chunks, result["final_answer"])

    result["weakest_claim"] = critique["weakest_claim"]
    result["audit_explanation"] = critique["explanation"]
    return critique


def finalize_with_audit(query: str, filter_source: str = None) -> dict:
    """`filter_source` lets a caller pass an explicit company source_file
    (e.g. from a UI dropdown) that overrides detect_company()'s query-text
    inference throughout self-correction and the red-team audit. If not
    provided, each step falls back to detect_company()."""
    result = answer_with_self_correction(query, filter_source=filter_source)

    if result["source_method"] == "metadata_lookup":
        matches_query = metadata_sanity_check(query, result["final_answer"])

        if not matches_query:
            # The metadata fast-path returned an answer that doesn't actually
            # address what was asked (e.g. an "exchange" pattern-match false
            # positive) — fall through to dense retrieval instead of
            # returning the mismatched answer.
            route = (
                {"scope": "single", "sources": [filter_source]}
                if filter_source is not None
                else route_query(query)
            )
            result = run_dense_retrieval_flow(query, route=route)
            critique = _run_redteam_pass(query, result, filter_source=filter_source)
            result["audit_status"] = "metadata_mismatch_corrected"
            return result

        result["audit_status"] = "skipped_metadata"
        return result

    critique = _run_redteam_pass(query, result, filter_source=filter_source)

    if critique["is_supported"] is False:
        result["audit_status"] = "flagged"
    else:
        result["audit_status"] = "passed"

    return result


if __name__ == "__main__":
    test_queries = [
        "What does Zoom say about foreign currency exchange rate risk?",
        "What is Atlassian's stock ticker symbol and which exchange is it listed on?",
        "What percentage of HubSpot's total revenue comes from Payments?",
        "What does Salesforce say about integrating generative AI or Agentforce into its platform?",
    ]

    for query in test_queries:
        result = finalize_with_audit(query)

        print("=" * 80)
        print(f"Query: {query}")
        print(f"Source method: {result['source_method']}")
        print(f"Audit status: {result['audit_status']}")
        print(f"\nFinal answer:\n{result['final_answer']}")

        if result["audit_status"] in ("flagged", "metadata_mismatch_corrected"):
            print(f"\nWeakest claim: {result.get('weakest_claim')}")
            print(f"Explanation: {result.get('audit_explanation')}")
        elif result["audit_status"] == "passed":
            print(f"\nWeakest claim checked: {result.get('weakest_claim')}")
            print(f"Explanation: {result.get('audit_explanation')}")

        print()
