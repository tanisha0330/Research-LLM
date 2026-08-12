import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path(__file__).parent / "chunks_v2.json"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "saas_10k_filings_v2"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
ADD_BATCH_SIZE = 500


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def build_collection(client: chromadb.PersistentClient, chunks: list[dict], embeddings: list[list[float]]):
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source_file": c["source_file"], "page_number": c["page_number"]} for c in chunks]

    for start in range(0, len(chunks), ADD_BATCH_SIZE):
        end = start + ADD_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    return collection


def dense_search(query: str, k: int = 5, model: SentenceTransformer = None, collection=None) -> list[dict]:
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    output = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        output.append(
            {
                "text": text,
                "source_file": metadata["source_file"],
                "page_number": metadata["page_number"],
                "similarity_score": 1 - distance,
            }
        )
    return output


def main():
    chunks = load_chunks()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(model, texts)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = build_collection(client, chunks, embeddings)

    print("=== Stage 2 v2 Embed Summary ===")
    print(f"Total chunks embedded: {len(chunks)}")
    print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
    print(f"Collection '{COLLECTION_NAME}' populated at {CHROMA_DIR}")

    return model, collection


if __name__ == "__main__":
    model, collection = main()

    query = "What is Atlassian's stock ticker symbol and which exchange is it listed on?"
    results = dense_search(query, k=5, model=model, collection=collection)

    print(f"\n=== dense_search (v2) for: {query!r} ===")
    found_team_chunk = False
    for rank, r in enumerate(results, start=1):
        snippet = r["text"].replace("\n", " ")
        is_team_chunk = "TEAM Nasdaq Global Select Market" in r["text"]
        if is_team_chunk:
            found_team_chunk = True
        marker = "  <-- TEAM/Nasdaq chunk" if is_team_chunk else ""
        print(
            f"#{rank} similarity_score={r['similarity_score']:.4f} "
            f"source={r['source_file']} page={r['page_number']}{marker}"
        )
        print(f"    {snippet}")

    print(f"\nTEAM/Nasdaq chunk in top-5: {'YES' if found_team_chunk else 'NO'}")
