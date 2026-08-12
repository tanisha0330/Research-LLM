import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stage3_redteam import finalize_with_audit

QUERY = "What percentage of HubSpot's total revenue comes from Payments?"
N_RUNS = 5


def looks_like_abstention(answer: str) -> bool:
    return "don't have enough information" in answer.lower() or "insufficient information" in answer.lower()


if __name__ == "__main__":
    results = []
    for i in range(1, N_RUNS + 1):
        result = finalize_with_audit(QUERY)
        abstained = looks_like_abstention(result["final_answer"])
        results.append((i, result, abstained))

        print("=" * 80)
        print(f"Run {i}")
        print(f"Source method: {result['source_method']}")
        print(f"Audit status: {result['audit_status']}")
        print(f"Abstained: {abstained}")
        print(f"\nFinal answer:\n{result['final_answer']}")
        if result["audit_status"] in ("flagged", "metadata_mismatch_corrected"):
            print(f"\nWeakest claim: {result.get('weakest_claim')}")
            print(f"Explanation: {result.get('audit_explanation')}")
        print()

    abstain_count = sum(1 for _, _, a in results if a)
    fabricate_count = N_RUNS - abstain_count
    fabricated_and_flagged = sum(1 for _, r, a in results if not a and r["audit_status"] == "flagged")
    fabricated_and_missed = fabricate_count - fabricated_and_flagged

    print("=" * 80)
    print("=== Summary across 5 runs ===")
    print(f"Correctly abstained: {abstain_count}/{N_RUNS}")
    print(f"Fabricated an answer: {fabricate_count}/{N_RUNS}")
    print(f"Of the fabrications, caught by red-team audit (flagged): {fabricated_and_flagged}/{fabricate_count if fabricate_count else 0}")
    print(f"Of the fabrications, MISSED by red-team audit (passed): {fabricated_and_missed}/{fabricate_count if fabricate_count else 0}")
