import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stage2_embed import dense_search
from stage3_hybrid import hybrid_search
from stage4_rerank import _bm25, _chunks, _collection, _embedding_model, hybrid_rerank_search

CHUNKS_PATH = Path(__file__).parent.parent / "src" / "chunks.json"
QUERY = "Zoom foreign currency exchange rate risk"
K = 5
SOURCE_FILE = "zoom-10k-2026.pdf"
KEYWORDS = ["foreign currency", "exchange rate", "currency risk", "functional currency"]


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def find_matching_chunks(chunks: list[dict]) -> list[dict]:
    matches = []
    for chunk in chunks:
        if chunk["source_file"] != SOURCE_FILE:
            continue
        text_lower = chunk["text"].lower()
        if any(keyword in text_lower for keyword in KEYWORDS):
            matches.append(chunk)
    return matches


def print_results(label: str, results: list[dict], score_key: str):
    print(f"\n-- {label} --")
    for rank, r in enumerate(results, start=1):
        print(
            f"  #{rank} {score_key}={r[score_key]:.4f} "
            f"source={r['source_file']} page={r['page_number']}"
        )
        snippet = r["text"].replace("\n", " ")
        if len(snippet) > 150:
            snippet = snippet[:150] + "..."
        print(f"      {snippet}")


def main():
    all_chunks = load_chunks()
    matches = find_matching_chunks(all_chunks)

    print("=== Step 1: Matching chunks in zoom-10k-2026.pdf ===")
    print(f"Keywords: {KEYWORDS}")
    print(f"Found {len(matches)} matching chunk(s)\n")
    match_ids = set()
    for m in matches:
        match_ids.add(m["chunk_id"])
        print(f"chunk_id={m['chunk_id']} page={m['page_number']}")
        print(f"text:\n{m['text']}\n")
        print("-" * 80)

    print(f"\n=== Step 2-4: Running all three search methods for query: {QUERY!r} ===")

    dense_results = dense_search(QUERY, k=K, model=_embedding_model, collection=_collection)
    print_results("dense_search", dense_results, "similarity_score")

    hybrid_results = hybrid_search(
        QUERY, k=K, model=_embedding_model, collection=_collection, bm25=_bm25, chunks=_chunks
    )
    print_results("hybrid_search", hybrid_results, "rrf_score")

    rerank_results = hybrid_rerank_search(QUERY, k=K)
    print_results("hybrid_rerank_search", rerank_results, "rerank_score")

    print("\n=== Step 6: Comparison ===")

    def contains_match(results: list[dict]) -> bool:
        return any(r["text"] in {m["text"] for m in matches} for r in results)

    dense_hit = contains_match(dense_results)
    hybrid_hit = contains_match(hybrid_results)
    rerank_hit = contains_match(rerank_results)

    print(f"dense_search contains a matching chunk:         {'YES' if dense_hit else 'NO'}")
    print(f"hybrid_search contains a matching chunk:         {'YES' if hybrid_hit else 'NO'}")
    print(f"hybrid_rerank_search contains a matching chunk:  {'YES' if rerank_hit else 'NO'}")

    if dense_hit and not hybrid_hit and not rerank_hit:
        print(
            "\nCONCLUSION: YES — the actual matching chunk is retrieved by dense_search "
            "but is missing from both hybrid_search and hybrid_rerank_search's top-5 results."
        )
    elif dense_hit and (hybrid_hit or rerank_hit):
        print(
            "\nCONCLUSION: NO — the matching chunk is present in at least one of the "
            "hybrid/hybrid_rerank result sets, not just dense_search."
        )
    elif not dense_hit:
        print(
            "\nCONCLUSION: NO — dense_search itself does not contain a matching chunk "
            "in its top-5, so this isn't a dense-vs-hybrid discrepancy."
        )


if __name__ == "__main__":
    main()
