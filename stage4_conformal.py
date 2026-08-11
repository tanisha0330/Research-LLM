import json
import math
from pathlib import Path

from stage2_embed import dense_search
from stage2_self_correct import detect_company
from stage_generate import _collection, _embedding_model

CALIBRATION_DATASET_PATH = Path(__file__).parent / "calibration_dataset.json"
TARGET_COVERAGE = 0.8

# Base non-conformity score per audit_status — this is the same audit-pipeline
# signal used in the earlier (degenerate) version of this module, but it is
# now only the STARTING point for a continuous score, not the final value.
BASE_SCORE_BY_AUDIT_STATUS = {
    "skipped_metadata": 0.0,
    "passed": 0.2,
    "metadata_mismatch_corrected": 0.5,
    "flagged": 0.8,
}

SIMILARITY_ADJUSTMENT_WEIGHT = 0.2


def load_calibration_dataset() -> list[dict]:
    return json.loads(CALIBRATION_DATASET_PATH.read_text(encoding="utf-8"))


def compute_confidence_signal(query: str, chunks: list, answer: str, audit_result: dict) -> float:
    """Continuous non-conformity score in [0.0, 1.0]; 0.0 = most trustworthy,
    1.0 = least trustworthy.

    base: audit_status bucket (coarse signal from the Module 3 red-team audit)
    adjustment: pushed higher when the top-1 dense_search similarity for the
      retrieved evidence is low — weak retrieval evidence should reduce
      trust even when the audit itself didn't flag anything, and vice versa.
    """
    base = BASE_SCORE_BY_AUDIT_STATUS[audit_result["audit_status"]]

    if chunks:
        top_similarity_score = max(c["similarity_score"] for c in chunks)
        adjustment = (1 - top_similarity_score) * SIMILARITY_ADJUSTMENT_WEIGHT
    else:
        # metadata_lookup answers never ran dense_search — no retrieval
        # evidence to adjust against, so the audit_status base stands alone.
        adjustment = 0.0

    score = base + adjustment
    return max(0.0, min(1.0, score))


def get_top1_chunks(entry: dict) -> list:
    """Re-runs dense_search on the original query (with the same company
    filter the pipeline would have applied) to recover the top-1 similarity
    score, since it wasn't persisted in calibration_dataset.json. Returns []
    for metadata_lookup answers, which never ran dense_search."""
    if entry["source_method"] == "metadata_lookup":
        return []

    filter_source = detect_company(entry["query"])
    return dense_search(
        entry["query"], k=5, model=_embedding_model, collection=_collection, filter_source=filter_source
    )


def split_calibration_test(dataset: list[dict]) -> tuple[list[dict], list[dict]]:
    # Same fixed, simple split as before: first 14 (eval_set.json order) are
    # calibration, remaining 14 are test.
    return dataset[:14], dataset[14:]


def compute_conformal_threshold(calibration_scores: list[float], target_coverage: float) -> tuple[float, int, float]:
    n = len(calibration_scores)
    level = math.ceil((n + 1) * target_coverage) / n
    k = math.ceil(level * n)  # order-statistic index (1-indexed)
    sorted_scores = sorted(calibration_scores)
    threshold = sorted_scores[k - 1]
    return threshold, k, level


def main():
    dataset = load_calibration_dataset()

    print("Recomputing continuous non-conformity scores for all 28 cases...")
    scored_dataset = []
    for entry in dataset:
        chunks = get_top1_chunks(entry)
        score = compute_confidence_signal(
            query=entry["query"],
            chunks=chunks,
            answer=entry["final_answer"],
            audit_result={"audit_status": entry["audit_status"]},
        )
        entry = dict(entry)
        entry["nonconformity_score"] = score
        scored_dataset.append(entry)
        print(f"  {entry['query']!r} -> score={score:.4f} (audit_status={entry['audit_status']})")

    calibration_set, test_set = split_calibration_test(scored_dataset)

    print("\n=== Split ===")
    print(f"Calibration set: {len(calibration_set)} queries (first 14, eval_set.json order)")
    print(f"Test set: {len(test_set)} queries (remaining 14)")

    calibration_scores = [e["nonconformity_score"] for e in calibration_set]
    threshold, k, level = compute_conformal_threshold(calibration_scores, TARGET_COVERAGE)

    print("\n=== Calibration ===")
    print(f"Calibration non-conformity scores: {[round(s, 4) for s in calibration_scores]}")
    print(f"n = {len(calibration_set)}, target coverage = {TARGET_COVERAGE}")
    print(f"quantile level = ceil((n+1)*0.8)/n = {level:.6f}  (order statistic k = {k})")
    print(f"Calibration threshold (non-conformity score cutoff): {threshold:.4f}")

    print("\n=== Test Set Breakdown ===")
    header = f"{'score':>8s} {'confidence label':<28s} {'llm_judge_correct':>18s}  query"
    print(header)

    high_confidence_entries = []
    low_confidence_entries = []
    for entry in test_set:
        score = entry["nonconformity_score"]
        if score <= threshold:
            label = "high confidence"
            high_confidence_entries.append(entry)
        else:
            label = "low confidence — review recommended"
            low_confidence_entries.append(entry)

        print(f"{score:>8.4f} {label:<28s} {str(entry['llm_judge_correct']):>18s}  {entry['query']}")

    print("\n=== Empirical Coverage ===")
    n_high_confidence = len(high_confidence_entries)
    n_high_confidence_correct = sum(1 for e in high_confidence_entries if e["llm_judge_correct"] is True)

    print(f"High-confidence test cases: {n_high_confidence} / {len(test_set)}")
    print(f"Low-confidence test cases (review recommended): {len(low_confidence_entries)} / {len(test_set)}")

    if n_high_confidence > 0:
        coverage = n_high_confidence_correct / n_high_confidence
        print(f"Of high-confidence cases, actually llm_judge_correct=True: {n_high_confidence_correct} / {n_high_confidence}")
        print(f"Empirical coverage: {coverage:.1%}  (target was {TARGET_COVERAGE:.0%})")
    else:
        print("No test cases were labeled high confidence — empirical coverage is undefined (0/0).")

    print("\n=== Discrimination Check ===")
    print("Previous (degenerate, audit_status-only) version: 0 / 14 test cases were low confidence.")
    print(f"This (continuous) version: {len(low_confidence_entries)} / 14 test cases are low confidence.")


if __name__ == "__main__":
    main()
