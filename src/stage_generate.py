import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from stage2_embed import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, dense_search

EVAL_SET_PATH = Path(__file__).parent.parent / "eval" / "eval_set.json"
GENERATION_MODEL = "llama3.1:8b"

_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _chroma_client.get_collection(COLLECTION_NAME)


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def build_context(chunks: list) -> str:
    parts = []
    for chunk in chunks:
        parts.append(f"[Source: {chunk['source_file']}, page {chunk['page_number']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list) -> str:
    context = build_context(chunks)
    prompt = f"""You are answering a question using ONLY the excerpts below, taken from SEC 10-K filings.

Rules:
- Answer using ONLY the information contained in the excerpts.
- When you use a fact, cite it with the source file and page number, e.g. (source: zoom-10k-2026.pdf, page 34).
- Before concluding that information is missing, carefully read through ALL provided excerpts individually — the answer may be in a chunk you haven't focused on yet.
- You must NOT reference any external websites, URLs, or sources such as Yahoo Finance, Bloomberg, Wikipedia, or any source not included in the excerpts below. If the excerpts don't contain the answer, say so explicitly and stop.
- Do not guess or fabricate any values, tickers, or facts not explicitly present in the excerpts.
- Only say "insufficient information" if you have checked every excerpt and none of them contain the answer. If the excerpts do not contain enough information to answer the question, respond with exactly this sentence and nothing else: "I don't have enough information to answer this question based on the provided excerpts."

Excerpts:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def critique_sufficiency(query: str, chunks: list, answer: str) -> dict:
    context = build_context(chunks)

    def build_prompt(strict: bool) -> str:
        format_instruction = (
            "Respond with ONLY a single valid JSON object and nothing else — "
            "no markdown code fences, no commentary before or after the JSON."
            if strict
            else "Respond with a JSON object."
        )
        return f"""Given the query, the retrieved excerpts, and the generated answer below, evaluate whether the answer is well-supported by the excerpts, or whether evidence is missing or weak.

Judge sufficiency based on MEANING, not exact wording. If the excerpts state the same fact the answer relies on — even in different phrasing, format, or with additional surrounding detail — treat it as sufficient. Only mark it insufficient if the answer relies on information that is genuinely absent from, contradicted by, or not implied by the excerpts.

{format_instruction}
The JSON object must have exactly these two keys:
{{"sufficient": true or false, "reason": "a short explanation"}}

Query: {query}

Excerpts:
{context}

Answer:
{answer}

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

        if isinstance(parsed, dict) and "sufficient" in parsed and "reason" in parsed:
            return {"sufficient": bool(parsed["sufficient"]), "reason": str(parsed["reason"])}

    return {
        "sufficient": None,
        "reason": f"Model did not return valid JSON after 2 attempts. Last raw output: {raw!r}",
    }


if __name__ == "__main__":
    eval_set = load_eval_set()
    test_cases = eval_set[:3]

    for case in test_cases:
        query = case["query"]
        chunks = dense_search(query, k=5, model=_embedding_model, collection=_collection)

        answer = generate_answer(query, chunks)
        critique = critique_sufficiency(query, chunks, answer)

        print("=" * 80)
        print(f"Query: {query}")
        print(f"\nAnswer:\n{answer}")
        print(f"\nCritique: {critique}")
        print()
