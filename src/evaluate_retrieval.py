import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import ollama

from stage2_embed import dense_search
from stage3_hybrid import hybrid_search
from stage4_rerank import _bm25, _chunks, _collection, _embedding_model, hybrid_rerank_search
from stage_generate import GENERATION_MODEL

EVAL_SET_PATH = Path(__file__).parent.parent / "eval" / "eval_set.json"
K = 5
METHOD_NAMES = ["dense", "hybrid", "hybrid_rerank"]

# Original keyword/source-file-based baseline numbers, kept here for the
# side-by-side comparison this script prints.
BASELINE_RESULTS = {
    "dense": {"hit_rate@5": 1.000, "precision@1": 0.893},
    "hybrid": {"hit_rate@5": 0.964, "precision@1": 0.679},
    "hybrid_rerank": {"hit_rate@5": 0.964, "precision@1": 0.821},
}

# LLM-judge results from the previous (lenient, no company-alignment check)
# run, kept here for the side-by-side comparison this script prints.
LENIENT_JUDGE_RESULTS = {
    "dense": {"hit_rate@5": 0.714, "precision@1": 0.321},
    "hybrid": {"hit_rate@5": 0.750, "precision@1": 0.357},
    "hybrid_rerank": {"hit_rate@5": 0.786, "precision@1": 0.393},
}

_strict_relevance_cache: dict[tuple[str, str, str], bool] = {}
_lenient_relevance_cache: dict[tuple[str, str], bool] = {}


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _run_judge(prompt: str) -> bool | None:
    raw = ""
    for attempt, strict_format in enumerate((False, True)):
        format_instruction = (
            "Respond with ONLY a single valid JSON object and nothing else — "
            "no markdown code fences, no commentary before or after the JSON."
            if strict_format
            else "Respond with a JSON object."
        )
        response = ollama.chat(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt.format(format_instruction=format_instruction)}],
        )
        raw = _extract_json(response["message"]["content"])

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict) and "relevant" in parsed:
            return bool(parsed["relevant"])

    return None


def judge_relevance(query: str, expected_keywords: list, chunk_text: str, source_file: str) -> bool:
    """LLM-as-judge relevance check for a single retrieved chunk, with a
    hard company-alignment prerequisite: a chunk from a different company
    than the one the query is asking about must be graded NOT relevant
    regardless of topical similarity. Cached per (query, chunk_text,
    source_file)."""
    cache_key = (query, chunk_text, source_file)
    if cache_key in _strict_relevance_cache:
        return _strict_relevance_cache[cache_key]

    keywords_text = ", ".join(expected_keywords)

    prompt = f"""You are grading whether a retrieved document excerpt is relevant to a query.

Query: {query}

Retrieved excerpt (from source document: {source_file}):
{{chunk_text_placeholder}}

Expected keywords (a HINT of what relevant content should cover in substance — NOT a literal string match requirement): {keywords_text}

First, check if this chunk is from the same company that the query is asking about. If the chunk's source document is for a DIFFERENT company than the one named in the query, it must be graded NOT relevant regardless of topical similarity — even if it discusses a similar theme. Only evaluate topical content relevance if the company matches.

Does this retrieved content contain the information needed to answer the query, based on what the expected keywords indicate?

{{format_instruction}}
Return a JSON object with exactly this key:
{{{{"relevant": true or false}}}}

JSON:"""
    prompt = prompt.replace("{chunk_text_placeholder}", chunk_text)

    result = _run_judge(prompt)
    if result is None:
        result = False

    _strict_relevance_cache[cache_key] = result
    return result


def judge_relevance_lenient(query: str, expected_keywords: list, chunk_text: str) -> bool:
    """The ORIGINAL judge prompt, without the company-alignment check —
    kept only so we can measure how many hybrid_rerank verdicts flip once
    the company-alignment fix is applied."""
    cache_key = (query, chunk_text)
    if cache_key in _lenient_relevance_cache:
        return _lenient_relevance_cache[cache_key]

    keywords_text = ", ".join(expected_keywords)

    prompt = f"""You are grading whether a retrieved document excerpt is relevant to a query.

Query: {query}

Expected keywords (a HINT of what relevant content should cover in substance — NOT a literal string match requirement): {keywords_text}

Retrieved excerpt:
{chunk_text}

Does this retrieved content contain the information needed to answer the query, based on what the expected keywords indicate?

{{format_instruction}}
Return a JSON object with exactly this key:
{{{{"relevant": true or false}}}}

JSON:"""

    result = _run_judge(prompt)
    if result is None:
        result = False

    _lenient_relevance_cache[cache_key] = result
    return result


def hit_at_k_llm(results: list[dict], query: str, expected_keywords: list) -> int:
    return int(
        any(judge_relevance(query, expected_keywords, r["text"], r["source_file"]) for r in results)
    )


def precision_at_1_llm(results: list[dict], query: str, expected_keywords: list) -> int:
    if not results:
        return 0
    top = results[0]
    return int(judge_relevance(query, expected_keywords, top["text"], top["source_file"]))


def run_methods(query: str) -> dict[str, list[dict]]:
    return {
        "dense": dense_search(query, k=K, model=_embedding_model, collection=_collection),
        "hybrid": hybrid_search(
            query, k=K, model=_embedding_model, collection=_collection, bm25=_bm25, chunks=_chunks
        ),
        "hybrid_rerank": hybrid_rerank_search(query, k=K),
    }


def main():
    eval_set = load_eval_set()
    totals = {method: {"hit": 0, "p1": 0} for method in METHOD_NAMES}

    flip_count = 0
    flip_total_checked = 0

    print("=== Per-Query Breakdown (LLM-judge relevance grading, company-alignment fix) ===")
    for case in eval_set:
        query = case["query"]
        expected_keywords = case.get("expected_keywords", [])
        results_by_method = run_methods(query)

        print(f"\nQuery: {query!r}")
        for method in METHOD_NAMES:
            results = results_by_method[method]
            hit = hit_at_k_llm(results, query, expected_keywords)
            p1 = precision_at_1_llm(results, query, expected_keywords)
            totals[method]["hit"] += hit
            totals[method]["p1"] += p1
            print(f"  {method:15s} hit@5={hit}  precision@1={p1}")

        # Flip-count measurement, scoped to hybrid_rerank only, per request:
        # for each hybrid_rerank chunk, compare the lenient (old) verdict
        # against the strict (new, company-checked) verdict.
        for r in results_by_method["hybrid_rerank"]:
            lenient_verdict = judge_relevance_lenient(query, expected_keywords, r["text"])
            strict_verdict = judge_relevance(query, expected_keywords, r["text"], r["source_file"])
            flip_total_checked += 1
            if lenient_verdict is True and strict_verdict is False:
                flip_count += 1

    n = len(eval_set)

    print("\n=== Final Summary (strict, company-checked LLM-judge grading, averaged over {} queries) ===".format(n))
    print(f"{'Method':15s} {'hit_rate@5':>12s} {'precision@1':>12s}")
    strict_results = {}
    for method in METHOD_NAMES:
        hit_rate = totals[method]["hit"] / n
        precision = totals[method]["p1"] / n
        strict_results[method] = {"hit_rate@5": hit_rate, "precision@1": precision}
        print(f"{method:15s} {hit_rate:>12.3f} {precision:>12.3f}")

    print("\n=== Comparison: source-file baseline vs lenient LLM-judge vs strict (company-checked) LLM-judge ===")
    header = (
        f"{'Method':15s} {'baseline':>10s} {'lenient hit@5':>15s} {'strict hit@5':>14s} "
        f"{'baseline p@1':>14s} {'lenient p@1':>13s} {'strict p@1':>12s}"
    )
    print(header)
    for method in METHOD_NAMES:
        b = BASELINE_RESULTS[method]
        lj = LENIENT_JUDGE_RESULTS[method]
        sj = strict_results[method]
        print(
            f"{method:15s} {'':>10s} {lj['hit_rate@5']:>15.3f} {sj['hit_rate@5']:>14.3f} "
            f"{b['precision@1']:>14.3f} {lj['precision@1']:>13.3f} {sj['precision@1']:>12.3f}"
        )

    print("\n=== Ranking Verdict ===")
    dense_p1 = strict_results["dense"]["precision@1"]
    rerank_p1 = strict_results["hybrid_rerank"]["precision@1"]
    dense_hit = strict_results["dense"]["hit_rate@5"]
    rerank_hit = strict_results["hybrid_rerank"]["hit_rate@5"]

    if rerank_p1 > dense_p1 and rerank_hit >= dense_hit:
        verdict = "hybrid_rerank STILL ranks above dense — the reversal survives the company-alignment fix."
    elif dense_p1 > rerank_p1 and dense_hit >= rerank_hit:
        verdict = "dense's advantage is RESTORED — the earlier reversal was primarily due to the leniency bug."
    else:
        verdict = "Result is MIXED — the two methods trade off across metrics; no clean reversal or restoration."
    print(verdict)
    print(f"  dense:         hit_rate@5={dense_hit:.3f}  precision@1={dense_p1:.3f}")
    print(f"  hybrid_rerank: hit_rate@5={rerank_hit:.3f}  precision@1={rerank_p1:.3f}")

    print("\n=== Leniency Bug Impact (hybrid_rerank chunks only) ===")
    print(f"Total hybrid_rerank chunk verdicts checked: {flip_total_checked}")
    print(f"Chunks that flipped from lenient=RELEVANT to strict=NOT RELEVANT: {flip_count}")
    if flip_total_checked > 0:
        print(f"Flip rate: {flip_count / flip_total_checked:.1%} of all hybrid_rerank chunk verdicts")


if __name__ == "__main__":
    main()
