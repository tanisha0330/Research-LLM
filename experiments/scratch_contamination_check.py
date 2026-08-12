import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stage2_embed import dense_search
from stage_generate import _collection, _embedding_model

QUERIES = {
    "What is the typical length of a DocuSign subscription contract?": "docusign-10k-2026.pdf",
    "What competitive factors does Atlassian identify against its competitors?": "attlasian-10k-2026.pdf",
    "What risks does Zoom identify related to devices and networks or emerging technologies?": "zoom-10k-2026.pdf",
}

for query, expected_source in QUERIES.items():
    results = dense_search(query, k=5, model=_embedding_model, collection=_collection)

    print("=" * 90)
    print(f"Query: {query}")
    print(f"Expected company source: {expected_source}\n")

    wrong_count = 0
    for rank, r in enumerate(results, start=1):
        is_wrong = r["source_file"] != expected_source
        if is_wrong:
            wrong_count += 1
        marker = "  <-- WRONG COMPANY" if is_wrong else ""
        print(f"#{rank} source={r['source_file']} page={r['page_number']} score={r['similarity_score']:.4f}{marker}")

    print(f"\nWrong-company chunks in top-5: {wrong_count}/5")
    print()
