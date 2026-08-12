# Retained for reference/comparison. See README_module1.md's Decision Reversal
# section: hybrid_rerank_search was tried as the production retrieval method
# and reverted after it regressed both final-answer correctness and conformal
# calibration coverage end-to-end. Not used in the production pipeline.

import sys

sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from stage2_embed import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from stage3_hybrid import build_bm25_index, hybrid_search, load_chunks

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_CANDIDATE_POOL_SIZE = 10

_chunks = load_chunks()
_bm25 = build_bm25_index(_chunks)
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _chroma_client.get_collection(COLLECTION_NAME)
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)


def rerank(query: str, candidates: list, top_k: int = 5) -> list[dict]:
    pairs = [(query, c["text"]) for c in candidates]
    scores = _cross_encoder.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    results = []
    for candidate, score in scored[:top_k]:
        result = dict(candidate)
        result["rerank_score"] = float(score)
        results.append(result)
    return results


def hybrid_rerank_search(
    query: str, k: int = 5, preprocess: bool = False, filter_source: str = None
) -> list[dict]:
    candidates = hybrid_search(
        query,
        k=HYBRID_CANDIDATE_POOL_SIZE,
        model=_embedding_model,
        collection=_collection,
        bm25=_bm25,
        chunks=_chunks,
        preprocess=preprocess,
        filter_source=filter_source,
    )
    return rerank(query, candidates, top_k=k)


def format_snippet(text: str, max_len: int = 130) -> str:
    snippet = text.replace("\n", " ")
    if len(snippet) > max_len:
        snippet = snippet[:max_len] + "..."
    return snippet


if __name__ == "__main__":
    demo_query = "operating margin improvement"
    results = hybrid_rerank_search(demo_query, k=5)

    print("=== Stage 4 Rerank Demo ===")
    print(f"Query: {demo_query!r}\n")
    for rank, r in enumerate(results, start=1):
        print(
            f"#{rank} rerank_score={r['rerank_score']:.4f} "
            f"source={r['source_file']} page={r['page_number']}"
        )
        print(f"    {format_snippet(r['text'])}")
