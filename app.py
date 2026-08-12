import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from stage_final_report import generate_report

EXAMPLE_QUERIES = {
    "-- Select an example --": "",
    "Atlassian ticker & exchange": "What is Atlassian's stock ticker symbol and which exchange is it listed on?",
    "HubSpot go-to-market approach": "How does HubSpot describe its customer acquisition strategy or go-to-market approach?",
    "DocuSign data security risks": "What risks does DocuSign describe related to data security or breaches?",
}

st.set_page_config(page_title="Research Co-Pilot", page_icon="🔎", layout="wide")

st.title("🔎 Research Co-Pilot")
st.caption(
    "A RAG system over SaaS company 10-K filings with self-correction, adversarial auditing, "
    "and statistically calibrated confidence scoring."
)

example_label = st.selectbox("Try an example", list(EXAMPLE_QUERIES.keys()))
default_query = EXAMPLE_QUERIES[example_label]

query = st.text_input("Research question", value=default_query, placeholder="Ask a question about one of the 5 companies...")

run_clicked = st.button("Run Research", type="primary")

if run_clicked:
    if not query.strip():
        st.warning("Please enter a question or select an example first.")
    else:
        with st.spinner("Running retrieval, generation, self-correction, and audit..."):
            report = generate_report(query)

        if report.confidence_label == "high confidence":
            st.success(f"✅ HIGH CONFIDENCE  (score: {report.confidence_score:.4f})")
        else:
            st.warning(f"⚠️ LOW CONFIDENCE — REVIEW RECOMMENDED  (score: {report.confidence_score:.4f})")

        st.subheader("Executive Summary")
        st.info(report.executive_summary)

        st.subheader("Full Answer")
        st.write(report.final_answer)

        if report.audit_note:
            with st.expander("⚠️ Audit Note"):
                st.write(report.audit_note)

        with st.expander(f"📄 Evidence Trail ({len(report.evidence_trail)} source{'s' if len(report.evidence_trail) != 1 else ''})"):
            if report.evidence_trail:
                for i, item in enumerate(report.evidence_trail, start=1):
                    st.markdown(f"**[{i}] `{item['source_file']}`, page {item['page_number']}**")
                    st.markdown(f"> {item['excerpt']}")
                    st.markdown("---")
            else:
                st.write("No retrieval evidence (structured metadata lookup).")

        with st.sidebar:
            st.header("Pipeline Metadata")
            st.metric("Source method", report.source_method)
            st.metric("Audit status", report.audit_status)
            st.metric("Retries used", report.retries_used)
            st.metric("Confidence score", f"{report.confidence_score:.4f}")
            st.caption(f"Generated: {report.generated_at}")
