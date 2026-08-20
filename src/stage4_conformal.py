import json
import math
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from stage2_embed import dense_search
from stage2_self_correct import detect_company
from stage_generate import _collection, _embedding_model

CALIBRATION_DATASET_PATH = Path(__file__).parent.parent / "eval" / "calibration_dataset.json"
TARGET_COVERAGE = 0.8
SPLIT_RANDOM_SEED = 42

# Base non-conformity score per audit_status — this is the same audit-pipeline
# signal used in earlier versions of this module, but it is only ONE of
# three signals feeding the final continuous score now.
BASE_SCORE_BY_AUDIT_STATUS = {
    "skipped_metadata": 0.0,
    "passed": 0.2,
    "metadata_mismatch_corrected": 0.5,
    "flagged": 0.8,
}

SIMILARITY_ADJUSTMENT_WEIGHT = 0.15
SPREAD_ADJUSTMENT_WEIGHT = 0.15
LENGTH_PENALTY = 0.05
SHORT_ANSWER_WORD_THRESHOLD = 20
LONG_ANSWER_WORD_THRESHOLD = 150

ABSTENTION_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "insufficient information",
    "not enough information",
]


def load_calibration_dataset() -> list[dict]:
    return json.loads(CALIBRATION_DATASET_PATH.read_text(encoding="utf-8"))


def is_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)


def compute_confidence_signal(query: str, chunks: list, answer: str, audit_result: dict) -> float:
    """Continuous non-conformity score in [0.0, 1.0]; 0.0 = most trustworthy,
    1.0 = least trustworthy. Combines three signals:

    1. base: audit_status bucket (coarse signal from the Module 3 red-team audit)
    2. similarity_adjustment: pushed higher when the top-1 dense_search
       similarity for the retrieved evidence is low — weak top match should
       reduce trust even when the audit itself didn't flag anything.
    3. spread_adjustment: pushed higher when the gap between the top-1 and
       top-5 similarity scores is SMALL — a small gap means all retrieved
       chunks look similarly (ir)relevant to the query (no clearly
       strongest match), which is itself a sign of weak retrieval,
       independent of how high the top score is in isolation.
    4. length_penalty: a small flat penalty for answers that are unusually
       short (<20 words) or unusually long (>150 words), since both
       extremes correlated with lower accuracy in manual review.
       Abstentions are exempt — an abstention being short is expected and
       not itself a sign of a bad answer.
    """
    base = BASE_SCORE_BY_AUDIT_STATUS[audit_result["audit_status"]]

    if chunks:
        scores_sorted = sorted((c["similarity_score"] for c in chunks), reverse=True)
        top1_score = scores_sorted[0]
        top5_score = scores_sorted[-1]
        score_gap = top1_score - top5_score

        similarity_adjustment = (1 - top1_score) * SIMILARITY_ADJUSTMENT_WEIGHT
        spread_adjustment = (1 - score_gap) * SPREAD_ADJUSTMENT_WEIGHT
    else:
        # metadata_lookup answers never ran dense_search — no retrieval
        # evidence to adjust against.
        similarity_adjustment = 0.0
        spread_adjustment = 0.0

    length_penalty = 0.0
    if not is_abstention(answer):
        word_count = len(answer.split())
        if word_count < SHORT_ANSWER_WORD_THRESHOLD or word_count > LONG_ANSWER_WORD_THRESHOLD:
            length_penalty = LENGTH_PENALTY

    score = base + similarity_adjustment + spread_adjustment + length_penalty
    return max(0.0, min(1.0, score))


def get_retrieval_chunks(entry: dict) -> list:
    """Re-runs dense_search on the original query (with the same company
    filter the pipeline would have applied) to recover the top-5 chunks
    and their similarity scores, since they weren't persisted in
    calibration_dataset.json. Returns [] for metadata_lookup answers,
    which never ran dense_search."""
    if entry["source_method"] == "metadata_lookup":
        return []

    filter_source = detect_company(entry["query"])
    return dense_search(
        entry["query"], k=5, model=_embedding_model, collection=_collection, filter_source=filter_source
    )


def split_calibration_test(dataset: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stratified ~50/50 split, seeded for reproducibility. Stratifies on
    (is_graceful_decline, llm_judge_correct) jointly, so both correctness
    classes AND the graceful_decline query type are proportionally spread
    across calibration and test -- replaces the old dataset[:14]/[14:]
    slice, which was sized for the original 28-query eval set and put all
    4 graceful_decline queries on one side once the eval set grew to 49."""
    rng = random.Random(SPLIT_RANDOM_SEED)

    def stratum_key(entry: dict) -> tuple[bool, bool]:
        is_decline = entry.get("expected_behavior") == "graceful_decline"
        return (is_decline, entry["llm_judge_correct"])

    strata: dict[tuple[bool, bool], list[dict]] = {}
    for entry in dataset:
        strata.setdefault(stratum_key(entry), []).append(entry)

    calibration: list[dict] = []
    test: list[dict] = []
    for entries in strata.values():
        entries = list(entries)
        rng.shuffle(entries)
        n_calibration = round(len(entries) / 2)
        calibration.extend(entries[:n_calibration])
        test.extend(entries[n_calibration:])

    rng.shuffle(calibration)
    rng.shuffle(test)
    return calibration, test


def bootstrap_coverage_ci(
    high_confidence_correct_flags: list[bool],
    n_resamples: int = 10000,
    ci: float = 0.90,
    seed: int = SPLIT_RANDOM_SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for empirical coverage.

    Empirical coverage on a small held-out test set (here ~18 high-confidence
    cases) is a single point estimate with wide uncertainty. This resamples
    the high-confidence correctness flags with replacement to report the
    coverage point estimate alongside a (default 90%) CI, so the headline
    number is read as a range rather than a precise measurement.

    Returns (point_estimate, ci_low, ci_high). Returns (nan, nan, nan) if
    there are no high-confidence cases to resample.
    """
    import numpy as np

    n = len(high_confidence_correct_flags)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan

    flags = np.asarray(high_confidence_correct_flags, dtype=float)
    point_estimate = float(flags.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = flags[idx].mean(axis=1)

    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    ci_low, ci_high = np.percentile(resample_means, [lower_pct, upper_pct])
    return point_estimate, float(ci_low), float(ci_high)


def compute_conformal_threshold(calibration_scores: list[float], target_coverage: float) -> tuple[float, int, float]:
    n = len(calibration_scores)
    level = math.ceil((n + 1) * target_coverage) / n
    k = math.ceil(level * n)  # order-statistic index (1-indexed)
    sorted_scores = sorted(calibration_scores)
    threshold = sorted_scores[k - 1]
    return threshold, k, level


def main():
    dataset = load_calibration_dataset()

    print(f"Recomputing continuous non-conformity scores for all {len(dataset)} cases (3-signal version)...")
    scored_dataset = []
    for entry in dataset:
        chunks = get_retrieval_chunks(entry)
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

    all_scores = [e["nonconformity_score"] for e in scored_dataset]
    overall_min, overall_max = min(all_scores), max(all_scores)
    overall_spread = overall_max - overall_min

    flagged_scores = [e["nonconformity_score"] for e in scored_dataset if e["audit_status"] == "flagged"]
    flagged_min, flagged_max = (min(flagged_scores), max(flagged_scores)) if flagged_scores else (None, None)
    flagged_spread = (flagged_max - flagged_min) if flagged_scores else None

    print(f"\n=== Score Distribution (new 3-signal version, all {len(scored_dataset)} cases) ===")
    print(f"Overall: min={overall_min:.4f}  max={overall_max:.4f}  spread={overall_spread:.4f}")
    if flagged_spread is not None:
        print(
            f"Within 'flagged' audit_status only ({len(flagged_scores)} cases): "
            f"min={flagged_min:.4f}  max={flagged_max:.4f}  spread={flagged_spread:.4f}"
        )

    print("\n=== Comparison to previous (2-signal) version ===")
    prev_flagged_min, prev_flagged_max = 0.8320, 0.8524
    prev_flagged_spread = prev_flagged_max - prev_flagged_min
    print(
        f"Previous 'flagged' cluster: min={prev_flagged_min:.4f}  max={prev_flagged_max:.4f}  "
        f"spread={prev_flagged_spread:.4f}"
    )
    if flagged_spread is not None:
        print(
            f"New      'flagged' cluster: min={flagged_min:.4f}  max={flagged_max:.4f}  spread={flagged_spread:.4f}"
        )
        widened_by = flagged_spread - prev_flagged_spread
        print(f"'flagged' cluster spread {'widened' if widened_by > 0 else 'narrowed'} by {abs(widened_by):.4f}")

    calibration_set, test_set = split_calibration_test(scored_dataset)

    print("\n=== Split ===")
    print(f"Calibration set: {len(calibration_set)} queries (stratified random split, seed={SPLIT_RANDOM_SEED})")
    print(f"Test set: {len(test_set)} queries (stratified random split, seed={SPLIT_RANDOM_SEED})")

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

        correct_flags = [e["llm_judge_correct"] is True for e in high_confidence_entries]
        _, ci_low, ci_high = bootstrap_coverage_ci(correct_flags)
        print(
            f"90% bootstrap CI on coverage: [{ci_low:.1%}, {ci_high:.1%}]  "
            f"(n={n_high_confidence} high-confidence cases — wide by design at this sample size)"
        )
    else:
        print("No test cases were labeled high confidence — empirical coverage is undefined (0/0).")

    print("\n=== Discrimination Check ===")
    print(f"This (3-signal) version: {len(low_confidence_entries)} / {len(test_set)} test cases are low confidence.")


if __name__ == "__main__":
    main()
