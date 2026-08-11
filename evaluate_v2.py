import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from stage2_embed_v2 import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, dense_search
import chromadb

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
K = 5


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def hit_at_k(results: list[dict], expected_source: str) -> int:
    return int(any(r["source_file"] == expected_source for r in results))


def precision_at_1(results: list[dict], expected_source: str) -> int:
    if not results:
        return 0
    return int(results[0]["source_file"] == expected_source)


def main():
    eval_set = load_eval_set()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    total_hit = 0
    total_p1 = 0

    print("=== Per-Query Breakdown (v2, dense-only) ===")
    for case in eval_set:
        query = case["query"]
        expected_source = case["expected_source"]
        results = dense_search(query, k=K, model=model, collection=collection)

        hit = hit_at_k(results, expected_source)
        p1 = precision_at_1(results, expected_source)
        total_hit += hit
        total_p1 += p1

        print(f"Query: {query!r}  expected={expected_source}  hit@5={hit}  precision@1={p1}")

    n = len(eval_set)
    hit_rate = total_hit / n
    precision = total_p1 / n

    print("\n=== Final Summary (v2 dense-only, averaged over {} queries) ===".format(n))
    print(f"{'Method':20s} {'hit_rate@5':>12s} {'precision@1':>12s}")
    print(f"{'v2_dense':20s} {hit_rate:>12.3f} {precision:>12.3f}")
    print()
    print(f"{'v1_dense (baseline)':20s} {1.000:>12.3f} {0.893:>12.3f}")


if __name__ == "__main__":
    main()
