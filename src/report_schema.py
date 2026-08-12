from pydantic import BaseModel


class ResearchReport(BaseModel):
    query: str
    executive_summary: str  # 2-3 sentence plain-language answer
    final_answer: str  # the full answer text
    evidence_trail: list[dict]  # [{source_file, page_number, excerpt}]
    source_method: str  # metadata_lookup / dense_retrieval
    audit_status: str  # passed / flagged / skipped_metadata / etc.
    audit_note: str | None  # weakest_claim + explanation, if flagged
    confidence_label: str  # "high confidence" / "low confidence — review recommended"
    confidence_score: float  # the non-conformity score
    retries_used: int
    generated_at: str  # timestamp
