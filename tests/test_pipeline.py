"""Contract tests for the research-copilot pipeline.

Run with:  python -m unittest discover -s tests    (from repo root)
       or:  ./venv/Scripts/python.exe -m unittest tests.test_pipeline

Uses only the standard-library ``unittest`` framework — no new dependency.

Two tiers:

* **Tier 1** (``stage1_ingest``) is import-light and always runs, including in
  a bare CI checkout. It pins the header-stripping and character-offset ->
  page-number logic that the whole retrieval layer depends on.
* **Tier 2** (the conformal statistical core and ``detect_company``) requires
  importing modules that load the embedding model and open the *gitignored*
  ``chroma_db/`` at import time. Those tests are skipped cleanly when that
  environment isn't present (e.g. a fresh CI checkout without a built index),
  mirroring how the project already guards its Ollama-dependent paths. None of
  these tests require a running Ollama server.
"""

import math
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# --------------------------------------------------------------------------
# Tier 1: stage1_ingest — pure text/offset logic, no heavy imports
# --------------------------------------------------------------------------
import stage1_ingest as ingest


class TestNormalizeLine(unittest.TestCase):
    def test_lowercases_and_masks_digits(self):
        self.assertEqual(ingest.normalize_line("Page 42 of 100"), "page # of #")

    def test_strips_whitespace(self):
        self.assertEqual(ingest.normalize_line("   Hello  "), "hello")


class TestHeaderFooterDetection(unittest.TestCase):
    def test_detects_line_repeated_across_pages(self):
        # A short header line present on every page should be flagged; a line
        # unique to one page should not.
        pages = [
            "Annual Report 2026\nRevenue grew this year.",
            "Annual Report 2026\nCosts were stable.",
            "Annual Report 2026\nOutlook is positive.",
        ]
        repeated = ingest.find_repeated_lines(pages)
        # normalize_line collapses each run of digits to a single '#'
        self.assertIn("annual report #", repeated)
        self.assertNotIn("revenue grew this year.", repeated)

    def test_single_page_has_no_repeats(self):
        self.assertEqual(ingest.find_repeated_lines(["only one page of text"]), set())

    def test_strip_removes_repeated_keeps_content(self):
        pages = ["Confidential\nReal content line one"]
        repeated = {"confidential"}
        cleaned = ingest.strip_headers_footers(pages, repeated)
        self.assertNotIn("Confidential", cleaned[0])
        self.assertIn("Real content line one", cleaned[0])


class TestOffsetToPage(unittest.TestCase):
    def test_offset_maps_to_correct_page(self):
        pages = ["aaaa", "bbbb", "cccc"]  # each 4 chars, separators between
        full_text, offsets = ingest.build_full_text_with_offsets(pages)
        # first page region
        self.assertEqual(ingest.page_for_offset(0, offsets), 1)
        # an offset inside the second page's char range
        second_page_start = offsets[1][0]
        self.assertEqual(ingest.page_for_offset(second_page_start + 1, offsets), 2)
        # third page
        third_page_start = offsets[2][0]
        self.assertEqual(ingest.page_for_offset(third_page_start, offsets), 3)

    def test_empty_offsets_falls_back_to_one(self):
        self.assertEqual(ingest.page_for_offset(0, []), 1)


# --------------------------------------------------------------------------
# Tier 2: conformal statistical core + detect_company (guarded import)
# --------------------------------------------------------------------------
try:
    import stage4_conformal as conformal
    from stage2_self_correct import (
        _metadata_fields_in_query,
        comparative_metadata_answer,
        detect_companies,
        detect_company,
        is_comparative,
        route_query,
        routed_dense_search,
    )

    _HEAVY_IMPORT_OK = True
    _HEAVY_IMPORT_ERR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    _HEAVY_IMPORT_OK = False
    _HEAVY_IMPORT_ERR = f"{type(exc).__name__}: {exc}"


@unittest.skipUnless(_HEAVY_IMPORT_OK, f"heavy import unavailable ({_HEAVY_IMPORT_ERR})")
class TestConformalThreshold(unittest.TestCase):
    def test_known_order_statistic(self):
        # scores 0.1..1.0, n=10, target 0.8:
        #   level = ceil((10+1)*0.8)/10 = ceil(8.8)/10 = 9/10 = 0.9
        #   k = ceil(0.9*10) = 9  ->  sorted_scores[8] = 0.9
        scores = [i / 10 for i in range(1, 11)]
        threshold, k, level = conformal.compute_conformal_threshold(scores, 0.8)
        self.assertEqual(k, 9)
        self.assertAlmostEqual(level, 0.9)
        self.assertAlmostEqual(threshold, 0.9)

    def test_threshold_inside_score_distribution(self):
        # Pins the degeneracy bug the README documents: the threshold must sit
        # strictly within the calibration score range, not on/above the ceiling.
        scores = [0.05, 0.2, 0.5, 0.5, 0.8, 0.97, 0.98, 0.98, 1.0, 0.3]
        threshold, _, _ = conformal.compute_conformal_threshold(scores, 0.8)
        self.assertGreaterEqual(threshold, min(scores))
        self.assertLessEqual(threshold, max(scores))


@unittest.skipUnless(_HEAVY_IMPORT_OK, f"heavy import unavailable ({_HEAVY_IMPORT_ERR})")
class TestBootstrapCoverageCI(unittest.TestCase):
    def test_all_correct_gives_full_coverage_and_zero_width(self):
        point, lo, hi = conformal.bootstrap_coverage_ci([True] * 8)
        self.assertEqual(point, 1.0)
        self.assertEqual(lo, 1.0)
        self.assertEqual(hi, 1.0)

    def test_empty_returns_nan(self):
        point, lo, hi = conformal.bootstrap_coverage_ci([])
        self.assertTrue(math.isnan(point) and math.isnan(lo) and math.isnan(hi))

    def test_mixed_brackets_point_estimate_and_is_deterministic(self):
        flags = [True, True, True, True, True, True, False, False]  # 0.75
        point, lo, hi = conformal.bootstrap_coverage_ci(flags)
        self.assertAlmostEqual(point, 0.75)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)
        self.assertLess(lo, hi)  # real uncertainty band
        # seeded -> identical on a second call
        again = conformal.bootstrap_coverage_ci(flags)
        self.assertEqual((point, lo, hi), again)


@unittest.skipUnless(_HEAVY_IMPORT_OK, f"heavy import unavailable ({_HEAVY_IMPORT_ERR})")
class TestConfidenceSignal(unittest.TestCase):
    def test_metadata_answer_has_no_retrieval_adjustment(self):
        # No chunks (metadata_lookup path): base(skipped_metadata)=0.0, plus a
        # short-answer length penalty (0.05) -> 0.05, matching real output.
        score = conformal.compute_confidence_signal(
            query="ticker?",
            chunks=[],
            answer="TEAM on Nasdaq",
            audit_result={"audit_status": "skipped_metadata"},
        )
        self.assertAlmostEqual(score, 0.05)

    def test_abstention_is_exempt_from_length_penalty(self):
        score = conformal.compute_confidence_signal(
            query="anything",
            chunks=[],
            answer="I don't have enough information to answer.",
            audit_result={"audit_status": "passed"},
        )
        self.assertAlmostEqual(score, 0.2)  # base(passed) only, no length penalty

    def test_score_is_clamped_to_unit_interval(self):
        chunks = [{"similarity_score": 0.1}, {"similarity_score": 0.05}]
        score = conformal.compute_confidence_signal(
            query="q",
            chunks=chunks,
            answer="a " * 300,  # very long -> length penalty
            audit_result={"audit_status": "flagged"},
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


@unittest.skipUnless(_HEAVY_IMPORT_OK, f"heavy import unavailable ({_HEAVY_IMPORT_ERR})")
class TestDetectCompany(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(detect_company("What is Atlassian's revenue?"), "attlasian-10k-2026.pdf")

    def test_fuzzy_match_on_misspelling(self):
        self.assertEqual(detect_company("What is Atlasian revenue?"), "attlasian-10k-2026.pdf")

    def test_no_company_returns_none(self):
        # A generic query naming no company must NOT false-match to one — this
        # guards the cross-company contamination class of bug.
        self.assertIsNone(detect_company("What is the total addressable market?"))


@unittest.skipUnless(_HEAVY_IMPORT_OK, f"heavy import unavailable ({_HEAVY_IMPORT_ERR})")
class TestQueryRouting(unittest.TestCase):
    def test_single_company_routes_to_that_document_only(self):
        route = route_query("What is Atlassian's revenue?")
        self.assertEqual(route["scope"], "single")
        self.assertEqual(route["sources"], ["attlasian-10k-2026.pdf"])

    def test_two_named_companies_is_comparative(self):
        route = route_query("How do Salesforce and HubSpot growth strategies compare?")
        self.assertEqual(route["scope"], "comparative")
        self.assertIn("salesforce-10k-2026.pdf", route["sources"])
        self.assertIn("hubspot-10k-2026.pdf", route["sources"])

    def test_comparison_phrasing_with_no_names_spans_all_five(self):
        route = route_query("Which of the five companies has the strongest position?")
        self.assertEqual(route["scope"], "comparative")
        self.assertEqual(len(route["sources"]), 5)

    def test_no_company_and_not_comparative_is_ambiguous(self):
        route = route_query("What is the total addressable market?")
        self.assertEqual(route["scope"], "ambiguous")
        self.assertEqual(route["sources"], [])

    def test_detect_companies_finds_all_named(self):
        self.assertEqual(
            set(detect_companies("Compare Zoom, DocuSign and Atlassian")),
            {"zoom-10k-2026.pdf", "docusign-10k-2026.pdf", "attlasian-10k-2026.pdf"},
        )

    def test_is_comparative_flags(self):
        self.assertTrue(is_comparative("compare X and Y"))
        self.assertFalse(is_comparative("What is Zoom's revenue?"))


@unittest.skipUnless(_HEAVY_IMPORT_OK, f"heavy import unavailable ({_HEAVY_IMPORT_ERR})")
class TestComparativeMetadata(unittest.TestCase):
    def test_field_detection(self):
        self.assertIn("fiscal_year_end", _metadata_fields_in_query("compare fiscal year ends"))
        self.assertIn("ticker_symbol", _metadata_fields_in_query("compare the ticker symbols"))
        self.assertIn("irs_ein", _metadata_fields_in_query("how do their EINs differ"))

    def test_non_metadata_query_returns_none(self):
        # A narrative comparative must not be captured by the metadata path;
        # it should fall through (None) to retrieval fan-out.
        self.assertIsNone(
            comparative_metadata_answer(
                "compare their growth strategies",
                ["salesforce-10k-2026.pdf", "hubspot-10k-2026.pdf"],
            )
        )


@unittest.skipUnless(_HEAVY_IMPORT_OK, f"heavy import unavailable ({_HEAVY_IMPORT_ERR})")
class TestRetrievalGuard(unittest.TestCase):
    """Exercises the actual live ChromaDB index (embeddings only, no Ollama)."""

    def test_single_scope_never_reads_other_documents(self):
        # The core contamination guard: a single-company route must return
        # chunks ONLY from that company's filing.
        route = {"scope": "single", "sources": ["zoom-10k-2026.pdf"]}
        chunks = routed_dense_search("competition and market risks", route, k=5)
        self.assertTrue(chunks)
        self.assertTrue(all(c["source_file"] == "zoom-10k-2026.pdf" for c in chunks))

    def test_comparative_scope_spans_the_named_documents(self):
        route = {
            "scope": "comparative",
            "sources": ["zoom-10k-2026.pdf", "hubspot-10k-2026.pdf"],
        }
        chunks = routed_dense_search("competition risks", route, k=6)
        sources = {c["source_file"] for c in chunks}
        self.assertEqual(sources, {"zoom-10k-2026.pdf", "hubspot-10k-2026.pdf"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
