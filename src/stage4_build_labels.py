import json
from pathlib import Path

import ollama

from stage3_redteam import finalize_with_audit
from stage_generate import GENERATION_MODEL

EVAL_SET_PATH = Path(__file__).parent.parent / "eval" / "eval_set.json"
OUTPUT_PATH = Path(__file__).parent.parent / "eval" / "calibration_dataset.json"

ABSTENTION_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "insufficient information",
    "not enough information",
]

SHORT_ANSWER_CHAR_THRESHOLD = 20


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def is_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)


def keyword_grade(case: dict, final_answer: str) -> tuple[bool, bool]:
    """Returns (keyword_correct, needs_manual_review). The original strict
    substring-match grader — kept for comparison against the LLM judge."""
    expected_keywords = case.get("expected_keywords", [])
    answer_lower = final_answer.lower()

    if is_abstention(final_answer):
        return False, True

    all_keywords_present = all(kw.lower() in answer_lower for kw in expected_keywords)
    needs_review = len(final_answer.strip()) < SHORT_ANSWER_CHAR_THRESHOLD

    return all_keywords_present, needs_review


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def judge_correctness(query: str, expected_keywords: list, final_answer: str) -> dict:
    keywords_text = ", ".join(expected_keywords)

    def build_prompt(strict: bool) -> str:
        format_instruction = (
            "Respond with ONLY a single valid JSON object and nothing else — "
            "no markdown code fences, no commentary before or after the JSON."
            if strict
            else "Respond with a JSON object."
        )
        return f"""You are grading whether an AI-generated answer correctly and substantively addresses a question.

Query: {query}

Expected keywords (a HINT of what a correct answer should cover in substance — NOT a literal string match requirement; the answer can be phrased entirely differently and still be correct if it conveys the same facts): {keywords_text}

Answer given:
{final_answer}

Does this answer correctly and substantively address what the expected keywords indicate, even if phrased differently?

Important: an abstention (e.g. "I don't have enough information to answer this question") must be graded correct=false, UNLESS the question is genuinely unanswerable from a SEC 10-K filing (none of the questions in this evaluation set are unanswerable — every one of them has a real answer available in the source documents). An abstention is a failure to find/use available information, not a valid answer.

{format_instruction}
Return a JSON object with exactly these keys:
{{"correct": true or false, "reasoning": "a short explanation of your judgment"}}

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

        if isinstance(parsed, dict) and "correct" in parsed and "reasoning" in parsed:
            return {"correct": bool(parsed["correct"]), "reasoning": parsed["reasoning"]}

    return {
        "correct": None,
        "reasoning": f"Judge did not return valid JSON after 2 attempts. Last raw output: {raw!r}",
    }


def main():
    eval_set = load_eval_set()

    dataset = []
    for case in eval_set:
        query = case["query"]
        expected_keywords = case.get("expected_keywords", [])
        result = finalize_with_audit(query)

        final_answer = result["final_answer"]
        keyword_correct, needs_manual_review = keyword_grade(case, final_answer)

        judge_result = judge_correctness(query, expected_keywords, final_answer)
        llm_judge_correct = judge_result["correct"]

        entry = {
            "query": query,
            "final_answer": final_answer,
            "source_method": result["source_method"],
            "audit_status": result["audit_status"],
            "keyword_correct": keyword_correct,
            "llm_judge_correct": llm_judge_correct,
            "llm_judge_reasoning": judge_result["reasoning"],
            "needs_manual_review": needs_manual_review,
        }
        if result["audit_status"] in ("flagged", "metadata_mismatch_corrected"):
            entry["weakest_claim"] = result.get("weakest_claim")

        dataset.append(entry)

        print(
            f"Graded: {query!r} -> keyword_correct={keyword_correct} "
            f"llm_judge_correct={llm_judge_correct} needs_manual_review={needs_manual_review}"
        )

    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    total = len(dataset)

    print("\n=== Full 28-query results: keyword grade vs LLM judge grade ===")
    header = f"{'#':>3s} {'keyword':>8s} {'llm_judge':>10s} {'agree':>6s}  query"
    print(header)
    disagree_count = 0
    for i, d in enumerate(dataset, start=1):
        agree = d["keyword_correct"] == d["llm_judge_correct"]
        if not agree:
            disagree_count += 1
        print(
            f"{i:>3d} {str(d['keyword_correct']):>8s} {str(d['llm_judge_correct']):>10s} "
            f"{'Y' if agree else 'N':>6s}  {d['query']}"
        )

    keyword_correct_count = sum(1 for d in dataset if d["keyword_correct"])
    llm_correct_count = sum(1 for d in dataset if d["llm_judge_correct"])

    print("\n=== Summary ===")
    print(f"Total queries: {total}")
    print(f"Keyword grader — correct: {keyword_correct_count}, incorrect: {total - keyword_correct_count}")
    print(f"LLM judge grader — correct: {llm_correct_count}, incorrect: {total - llm_correct_count}")
    print(f"Cases where the two graders disagree: {disagree_count} / {total}")

    print("\n=== audit_status x llm_judge_correct cross-tab ===")
    statuses = sorted({d["audit_status"] for d in dataset})
    header = f"{'audit_status':25s} {'correct':>10s} {'incorrect':>10s} {'total':>8s}"
    print(header)
    for status in statuses:
        subset = [d for d in dataset if d["audit_status"] == status]
        correct = sum(1 for d in subset if d["llm_judge_correct"])
        incorrect = len(subset) - correct
        print(f"{status:25s} {correct:>10d} {incorrect:>10d} {len(subset):>8d}")

    print(f"\nSaved calibration dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
