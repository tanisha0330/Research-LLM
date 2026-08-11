import json
from pathlib import Path

from stage2_embed import dense_search
from stage3_hybrid import hybrid_search
from stage4_rerank import _bm25, _chunks, _collection, _embedding_model, hybrid_rerank_search

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
K = 5
METHOD_NAMES = [
    "dense",
    "hybrid",
    "hybrid_preprocessed",
    "hybrid_rerank",
    "hybrid_rerank_preprocessed",
]


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def hit_at_k(results: list[dict], expected_source: str) -> int:
    return int(any(r["source_file"] == expected_source for r in results))


def precision_at_1(results: list[dict], expected_source: str) -> int:
    if not results:
        return 0
    return int(results[0]["source_file"] == expected_source)


def run_methods(query: str) -> dict[str, list[dict]]:
    return {
        "dense": dense_search(query, k=K, model=_embedding_model, collection=_collection),
        "hybrid": hybrid_search(
            query, k=K, model=_embedding_model, collection=_collection, bm25=_bm25, chunks=_chunks
        ),
        "hybrid_preprocessed": hybrid_search(
            query,
            k=K,
            model=_embedding_model,
            collection=_collection,
            bm25=_bm25,
            chunks=_chunks,
            preprocess=True,
        ),
        "hybrid_rerank": hybrid_rerank_search(query, k=K),
        "hybrid_rerank_preprocessed": hybrid_rerank_search(query, k=K, preprocess=True),
    }


def main():
    eval_set = load_eval_set()
    totals = {method: {"hit": 0, "p1": 0} for method in METHOD_NAMES}

    print("=== Per-Query Breakdown ===")
    for case in eval_set:
        query = case["query"]
        expected_source = case["expected_source"]
        results_by_method = run_methods(query)

        print(f"\nQuery: {query!r}")
        print(f"  expected_source: {expected_source}")
        for method in METHOD_NAMES:
            results = results_by_method[method]
            hit = hit_at_k(results, expected_source)
            p1 = precision_at_1(results, expected_source)
            totals[method]["hit"] += hit
            totals[method]["p1"] += p1
            print(f"  {method:15s} hit@5={hit}  precision@1={p1}")

    n = len(eval_set)
    print("\n=== Final Summary (averaged over {} queries) ===".format(n))
    print(f"{'Method':15s} {'hit_rate@5':>12s} {'precision@1':>12s}")
    for method in METHOD_NAMES:
        hit_rate = totals[method]["hit"] / n
        precision = totals[method]["p1"] / n
        print(f"{method:15s} {hit_rate:>12.3f} {precision:>12.3f}")


if __name__ == "__main__":
    main()
