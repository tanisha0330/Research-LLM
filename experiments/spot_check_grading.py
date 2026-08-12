import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ollama

from stage2_embed import dense_search
from stage4_rerank import hybrid_rerank_search
from stage_generate import GENERATION_MODEL, _collection, _embedding_model

EVAL_SET_PATH = Path(__file__).parent.parent / "eval" / "eval_set.json"
K = 5
NEEDED_TYPE1 = 4  # dense: keyword_match=True, llm=False
NEEDED_TYPE2 = 4  # hybrid_rerank: keyword_match=False, llm=True
NEEDED_PRECISION1_SAMPLES = 5

random.seed(42)


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def keyword_match(expected_keywords: list, chunk_text: str) -> bool:
    text_lower = chunk_text.lower()
    return all(kw.lower() in text_lower for kw in expected_keywords)


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def judge_relevance_with_reasoning(query: str, expected_keywords: list, chunk_text: str) -> tuple:
    keywords_text = ", ".join(expected_keywords)

    def build_prompt(strict: bool) -> str:
        format_instruction = (
            "Respond with ONLY a single valid JSON object and nothing else — "
            "no markdown code fences, no commentary before or after the JSON."
            if strict
            else "Respond with a JSON object."
        )
        return f"""You are grading whether a retrieved document excerpt is relevant to a query.

Query: {query}

Expected keywords (a HINT of what relevant content should cover in substance — NOT a literal string match requirement): {keywords_text}

Retrieved excerpt:
{chunk_text}

Does this retrieved content contain the information needed to answer the query, based on what the expected keywords indicate?

{format_instruction}
Return a JSON object with exactly these keys:
{{"relevant": true or false, "reasoning": "a short explanation of your judgment"}}

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

        if isinstance(parsed, dict) and "relevant" in parsed:
            return bool(parsed["relevant"]), parsed.get("reasoning", "")

    return False, f"Judge did not return valid JSON. Raw: {raw!r}"


def main():
    eval_set = load_eval_set()

    type1_cases = []  # dense: keyword_match=True, llm=False
    type2_cases = []  # hybrid_rerank: keyword_match=False, llm=True
    precision1_not_relevant = []  # dense top-1 chunk where llm says not relevant

    for case in eval_set:
        query = case["query"]
        expected_keywords = case.get("expected_keywords", [])

        dense_results = dense_search(query, k=K, model=_embedding_model, collection=_collection)
        for idx, r in enumerate(dense_results):
            km = keyword_match(expected_keywords, r["text"])
            llm_relevant, reasoning = judge_relevance_with_reasoning(query, expected_keywords, r["text"])

            if km and not llm_relevant:
                type1_cases.append(
                    {
                        "query": query,
                        "expected_keywords": expected_keywords,
                        "chunk_text": r["text"],
                        "source_file": r["source_file"],
                        "page_number": r["page_number"],
                        "keyword_match": km,
                        "llm_relevant": llm_relevant,
                        "reasoning": reasoning,
                    }
                )

            if idx == 0 and not llm_relevant:
                precision1_not_relevant.append(
                    {
                        "query": query,
                        "expected_keywords": expected_keywords,
                        "chunk_text": r["text"],
                        "source_file": r["source_file"],
                        "page_number": r["page_number"],
                        "reasoning": reasoning,
                    }
                )

        rerank_results = hybrid_rerank_search(query, k=K)
        for r in rerank_results:
            km = keyword_match(expected_keywords, r["text"])
            llm_relevant, reasoning = judge_relevance_with_reasoning(query, expected_keywords, r["text"])

            if not km and llm_relevant:
                type2_cases.append(
                    {
                        "query": query,
                        "expected_keywords": expected_keywords,
                        "chunk_text": r["text"],
                        "source_file": r["source_file"],
                        "page_number": r["page_number"],
                        "keyword_match": km,
                        "llm_relevant": llm_relevant,
                        "reasoning": reasoning,
                    }
                )

        print(
            f"Scanned: {query!r}  "
            f"(type1 so far: {len(type1_cases)}, type2 so far: {len(type2_cases)}, "
            f"precision1-not-relevant so far: {len(precision1_not_relevant)})"
        )

    print("\n" + "=" * 100)
    print(f"TASK 1a: dense_search chunks — keyword_match=True but LLM-judge=NOT relevant  (found {len(type1_cases)}, showing up to {NEEDED_TYPE1})")
    print("=" * 100)
    for i, c in enumerate(type1_cases[:NEEDED_TYPE1], start=1):
        print(f"\n--- Case {i} ---")
        print(f"Query: {c['query']}")
        print(f"Expected keywords: {c['expected_keywords']}")
        print(f"Source: {c['source_file']}, page {c['page_number']}")
        print(f"\nFull retrieved chunk text:\n{c['chunk_text']}")
        print(f"\nKeyword-match verdict: RELEVANT (all keywords present as substrings)")
        print(f"LLM-judge verdict: NOT RELEVANT")
        print(f"LLM-judge reasoning: {c['reasoning']}")

    print("\n" + "=" * 100)
    print(f"TASK 1b: hybrid_rerank chunks — keyword_match=False but LLM-judge=relevant  (found {len(type2_cases)}, showing up to {NEEDED_TYPE2})")
    print("=" * 100)
    for i, c in enumerate(type2_cases[:NEEDED_TYPE2], start=1):
        print(f"\n--- Case {i} ---")
        print(f"Query: {c['query']}")
        print(f"Expected keywords: {c['expected_keywords']}")
        print(f"Source: {c['source_file']}, page {c['page_number']}")
        print(f"\nFull retrieved chunk text:\n{c['chunk_text']}")
        print(f"\nKeyword-match verdict: NOT RELEVANT (not all keywords present as substrings)")
        print(f"LLM-judge verdict: RELEVANT")
        print(f"LLM-judge reasoning: {c['reasoning']}")

    print("\n" + "=" * 100)
    sample_size = min(NEEDED_PRECISION1_SAMPLES, len(precision1_not_relevant))
    sampled = random.sample(precision1_not_relevant, sample_size) if precision1_not_relevant else []
    print(f"TASK 2: Random sample of dense_search precision@1 cases graded NOT relevant by LLM-judge  (found {len(precision1_not_relevant)} total, sampling {sample_size})")
    print("=" * 100)
    for i, c in enumerate(sampled, start=1):
        print(f"\n--- Sample {i} ---")
        print(f"Query: {c['query']}")
        print(f"Expected keywords: {c['expected_keywords']}")
        print(f"Source: {c['source_file']}, page {c['page_number']}")
        print(f"\nFull top-1 chunk text:\n{c['chunk_text']}")
        print(f"\nLLM-judge reasoning for NOT relevant: {c['reasoning']}")


if __name__ == "__main__":
    main()
