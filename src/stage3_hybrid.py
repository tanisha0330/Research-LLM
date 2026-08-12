# Retained for reference/comparison against dense_search and hybrid_rerank_search.
# See README_module1.md's Decision Reversal section: hybrid_rerank_search was
# tried as the production retrieval method and reverted after it regressed both
# final-answer correctness and conformal calibration coverage end-to-end.
# Not used in the production pipeline (stage2_self_correct.py uses dense_search).

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from stage2_embed import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, dense_search

CHUNKS_PATH = Path(__file__).parent / "chunks.json"
RRF_K = 60
CANDIDATE_POOL_SIZE = 15

BM25_STOPWORDS = {
    "what", "does", "do", "is", "are", "the", "a", "an", "say", "about",
    "how", "of", "in", "on", "to", "for", "and", "or", "?",
}


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def clean_for_bm25(query: str) -> str:
    tokens = query.lower().split()
    kept = [t for t in tokens if t.strip("?.,!") not in BM25_STOPWORDS and t not in BM25_STOPWORDS]
    return " ".join(kept)


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def sparse_search(
    query: str,
    k: int = 5,
    bm25: BM25Okapi = None,
    chunks: list[dict] = None,
    preprocess: bool = False,
    filter_source: str = None,
) -> list[dict]:
    query_for_bm25 = clean_for_bm25(query) if preprocess else query
    scores = bm25.get_scores(tokenize(query_for_bm25))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    if filter_source:
        ranked_indices = [i for i in ranked_indices if chunks[i]["source_file"] == filter_source]

    top_indices = ranked_indices[:k]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append(
            {
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "bm25_score": float(scores[idx]),
            }
        )
    return results


def hybrid_search(
    query: str,
    k: int = 5,
    model: SentenceTransformer = None,
    collection=None,
    bm25: BM25Okapi = None,
    chunks: list[dict] = None,
    preprocess: bool = False,
    filter_source: str = None,
) -> list[dict]:
    dense_results = dense_search(
        query, k=CANDIDATE_POOL_SIZE, model=model, collection=collection, filter_source=filter_source
    )
    sparse_results = sparse_search(
        query,
        k=CANDIDATE_POOL_SIZE,
        bm25=bm25,
        chunks=chunks,
        preprocess=preprocess,
        filter_source=filter_source,
    )

    rrf_scores: dict[str, float] = {}
    metadata: dict[str, tuple[str, int]] = {}

    for ranked_list in (dense_results, sparse_results):
        for rank, r in enumerate(ranked_list, start=1):
            key = r["text"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (RRF_K + rank)
            metadata[key] = (r["source_file"], r["page_number"])

    top_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for text, score in top_keys:
        source_file, page_number = metadata[text]
        results.append(
            {
                "text": text,
                "source_file": source_file,
                "page_number": page_number,
                "rrf_score": score,
            }
        )
    return results


def format_snippet(text: str, max_len: int = 130) -> str:
    snippet = text.replace("\n", " ")
    if len(snippet) > max_len:
        snippet = snippet[:max_len] + "..."
    return snippet


def print_method_results(label: str, results: list[dict], score_key: str):
    print(f"  -- {label} --")
    for rank, r in enumerate(results, start=1):
        print(
            f"    #{rank} {score_key}={r[score_key]:.4f} "
            f"source={r['source_file']} page={r['page_number']}"
        )
        print(f"        {format_snippet(r['text'])}")


def main():
    chunks = load_chunks()
    bm25 = build_bm25_index(chunks)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    print("=== Stage 3 Hybrid Search Setup ===")
    print(f"BM25 index built over {len(chunks)} chunks")
    print(f"Connected to Chroma collection '{COLLECTION_NAME}' ({collection.count()} vectors)")

    return model, collection, bm25, chunks


if __name__ == "__main__":
    model, collection, bm25, chunks = main()

    test_queries = [
        "TEAM Nasdaq Global Select Market",
        "how does the company think about long-term customer value",
        "operating margin improvement",
    ]

    print("\n=== Side-by-Side Comparison ===")
    for query in test_queries:
        print(f"\nQuery: {query!r}")

        dense_results = dense_search(query, k=3, model=model, collection=collection)
        sparse_results = sparse_search(query, k=3, bm25=bm25, chunks=chunks)
        hybrid_results = hybrid_search(query, k=3, model=model, collection=collection, bm25=bm25, chunks=chunks)

        print_method_results("dense_search", dense_results, "similarity_score")
        print_method_results("sparse_search (BM25)", sparse_results, "bm25_score")
        print_method_results("hybrid_search (RRF)", hybrid_results, "rrf_score")
