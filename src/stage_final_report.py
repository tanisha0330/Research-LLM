import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from report_schema import ResearchReport
from stage2_embed import dense_search
from stage2_self_correct import detect_company
from stage3_redteam import finalize_with_audit
from stage4_conformal import (
    TARGET_COVERAGE,
    compute_confidence_signal,
    compute_conformal_threshold,
    get_retrieval_chunks,
    load_calibration_dataset,
    split_calibration_test,
)
from stage_generate import _collection, _embedding_model

METADATA_PATH = Path(__file__).parent / "company_metadata.json"
EVIDENCE_EXCERPT_MAX_LEN = 300

_company_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

_cached_threshold = None


def get_calibration_threshold() -> float:
    """Computes (and caches) the split-conformal threshold from the same
    14-query calibration set used in stage4_conformal.py, so report
    generation doesn't recompute it from scratch on every call."""
    global _cached_threshold
    if _cached_threshold is not None:
        return _cached_threshold

    dataset = load_calibration_dataset()
    calibration_set, _ = split_calibration_test(dataset)

    calibration_scores = []
    for entry in calibration_set:
        chunks = get_retrieval_chunks(entry)
        score = compute_confidence_signal(
            query=entry["query"],
            chunks=chunks,
            answer=entry["final_answer"],
            audit_result={"audit_status": entry["audit_status"]},
        )
        calibration_scores.append(score)

    threshold, _, _ = compute_conformal_threshold(calibration_scores, TARGET_COVERAGE)
    _cached_threshold = threshold
    return threshold


def _format_metadata_excerpt(metadata_entry: dict) -> str:
    company_name = metadata_entry.get("company_name") or "Unknown company"
    fields = [
        ("Ticker", metadata_entry.get("ticker_symbol")),
        ("Exchange", metadata_entry.get("exchange")),
        ("Incorporated in", metadata_entry.get("state_of_incorporation")),
        ("Fiscal Year End", metadata_entry.get("fiscal_year_end")),
        ("Address", metadata_entry.get("principal_office_address")),
        ("EIN", metadata_entry.get("irs_ein")),
    ]
    parts = [f"{label}: {value}" for label, value in fields if value]
    return f"{company_name} — " + ", ".join(parts)


def _build_evidence_trail(result: dict, original_query: str) -> list[dict]:
    if result["source_method"] == "metadata_lookup":
        source_company = result.get("source_company")
        metadata_entry = _company_metadata.get(source_company, {})
        return [
            {
                "source_file": source_company,
                "page_number": 1,
                "excerpt": _format_metadata_excerpt(metadata_entry),
            }
        ]

    filter_source = detect_company(original_query)
    chunks = dense_search(
        result["query_used"],
        k=5,
        model=_embedding_model,
        collection=_collection,
        filter_source=filter_source,
    )
    trail = []
    for chunk in chunks:
        excerpt = chunk["text"].replace("\n", " ").strip()
        if len(excerpt) > EVIDENCE_EXCERPT_MAX_LEN:
            excerpt = excerpt[:EVIDENCE_EXCERPT_MAX_LEN] + "..."
        trail.append(
            {
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "excerpt": excerpt,
            }
        )
    return trail


def _build_executive_summary(final_answer: str, max_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", final_answer.strip())
    sentences = [s for s in sentences if s]
    return " ".join(sentences[:max_sentences])


def _build_audit_note(result: dict) -> str | None:
    if result["audit_status"] not in ("flagged", "metadata_mismatch_corrected"):
        return None
    weakest_claim = result.get("weakest_claim")
    explanation = result.get("audit_explanation")
    return f"Weakest claim: {weakest_claim}\nExplanation: {explanation}"


def generate_report(query: str) -> ResearchReport:
    result = finalize_with_audit(query)

    evidence_trail = _build_evidence_trail(result, query)

    if result["source_method"] == "metadata_lookup":
        chunks_for_scoring = []
    else:
        filter_source = detect_company(query)
        chunks_for_scoring = dense_search(
            result["query_used"],
            k=5,
            model=_embedding_model,
            collection=_collection,
            filter_source=filter_source,
        )

    confidence_score = compute_confidence_signal(
        query=query,
        chunks=chunks_for_scoring,
        answer=result["final_answer"],
        audit_result={"audit_status": result["audit_status"]},
    )

    threshold = get_calibration_threshold()
    confidence_label = "high confidence" if confidence_score <= threshold else "low confidence — review recommended"

    return ResearchReport(
        query=query,
        executive_summary=_build_executive_summary(result["final_answer"]),
        final_answer=result["final_answer"],
        evidence_trail=evidence_trail,
        source_method=result["source_method"],
        audit_status=result["audit_status"],
        audit_note=_build_audit_note(result),
        confidence_label=confidence_label,
        confidence_score=confidence_score,
        retries_used=result["retries_used"],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def render_markdown(report: ResearchReport) -> str:
    if report.confidence_label == "high confidence":
        badge = "✅ HIGH CONFIDENCE"
    else:
        badge = "⚠️ LOW CONFIDENCE — REVIEW RECOMMENDED"

    lines = []
    lines.append(f"# Research Report")
    lines.append("")
    lines.append(f"**{badge}**  (score: {report.confidence_score:.4f})")
    lines.append("")
    lines.append(f"*Generated: {report.generated_at}*")
    lines.append("")
    lines.append("## Query")
    lines.append("")
    lines.append(report.query)
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(report.executive_summary)
    lines.append("")
    lines.append("## Full Answer")
    lines.append("")
    lines.append(report.final_answer)
    lines.append("")

    if report.audit_note:
        lines.append("## Audit Note")
        lines.append("")
        lines.append(f"> {report.audit_note}".replace("\n", "\n> "))
        lines.append("")

    lines.append("## Evidence Trail")
    lines.append("")
    if report.evidence_trail:
        for i, item in enumerate(report.evidence_trail, start=1):
            lines.append(
                f"**[{i}]** `{item['source_file']}`, page {item['page_number']}"
            )
            lines.append("")
            lines.append(f"> {item['excerpt']}")
            lines.append("")
    else:
        lines.append("_No retrieval evidence (structured metadata lookup)._")
        lines.append("")

    lines.append("## Pipeline Metadata")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Source method | `{report.source_method}` |")
    lines.append(f"| Audit status | `{report.audit_status}` |")
    lines.append(f"| Confidence label | {report.confidence_label} |")
    lines.append(f"| Confidence score (non-conformity) | {report.confidence_score:.4f} |")
    lines.append(f"| Retries used | {report.retries_used} |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    test_queries = [
        ("report_1.md", "What is Atlassian's stock ticker symbol and which exchange is it listed on?"),
        ("report_2.md", "How does HubSpot describe its customer acquisition strategy or go-to-market approach?"),
        ("report_3.md", "What risks does DocuSign describe related to data security or breaches?"),
    ]

    for filename, query in test_queries:
        report = generate_report(query)
        markdown = render_markdown(report)

        output_path = Path(__file__).parent.parent / "reports" / filename
        output_path.write_text(markdown, encoding="utf-8")

        print("=" * 80)
        print(f"Saved {filename} for query: {query!r}")
        print(f"  source_method={report.source_method}  audit_status={report.audit_status}  "
              f"confidence_label={report.confidence_label}  score={report.confidence_score:.4f}")
