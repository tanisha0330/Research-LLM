import ollama

from stage_generate import (
    GENERATION_MODEL,
    _collection,
    _embedding_model,
    critique_sufficiency,
    generate_answer,
)
from stage2_embed import dense_search

MAX_RETRIES_HARD_LIMIT = 1


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


def answer_with_self_correction(query: str, max_retries: int = 1) -> dict:
    max_retries = min(max_retries, MAX_RETRIES_HARD_LIMIT)

    chunks = dense_search(query, k=5, model=_embedding_model, collection=_collection)
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
        }

    reformulated_query = reformulate_query(query, critique["reason"])

    chunks_2 = dense_search(reformulated_query, k=5, model=_embedding_model, collection=_collection)
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
    }


if __name__ == "__main__":
    test_queries = [
        "What is Atlassian's stock ticker symbol and which exchange is it listed on?",
        "What percentage of HubSpot's total revenue comes from Payments?",
    ]

    for query in test_queries:
        result = answer_with_self_correction(query, max_retries=1)

        print("=" * 80)
        print(f"Original query: {result['original_query']}")
        print(f"Reformulated query: {result['reformulated_query']}")
        print(f"\nFinal answer:\n{result['final_answer']}")
        print(f"\nSucceeded (sufficient): {result['sufficient']}")
        print(f"Critique reason: {result['critique_reason']}")
        print(f"Retries used: {result['retries_used']}")
        print()
