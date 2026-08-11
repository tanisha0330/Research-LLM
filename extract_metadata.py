import json
from pathlib import Path

import ollama
from pypdf import PdfReader

DOCUMENTS_DIR = Path(__file__).parent / "documents"
OUTPUT_PATH = Path(__file__).parent / "company_metadata.json"
GENERATION_MODEL = "llama3.1:8b"
PAGES_TO_READ = 2

FIELDS = [
    "company_name",
    "ticker_symbol",
    "exchange",
    "state_of_incorporation",
    "fiscal_year_end",
    "principal_office_address",
    "irs_ein",
]


def extract_cover_pages_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages[:PAGES_TO_READ]
    texts = [page.extract_text() or "" for page in pages]
    return "\n\n".join(texts)


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def build_prompt(cover_text: str, strict: bool) -> str:
    format_instruction = (
        "Respond with ONLY a single valid JSON object and nothing else — "
        "no markdown code fences, no commentary before or after the JSON."
        if strict
        else "Respond with a JSON object."
    )
    return f"""Extract the following fields from the SEC 10-K cover page text below.

Fields to extract:
- company_name: the full registrant company name
- ticker_symbol: the stock trading symbol
- exchange: the exchange the stock is listed on
- state_of_incorporation: the state or jurisdiction of incorporation
- fiscal_year_end: the date the fiscal year ends (e.g. "June 30")
- principal_office_address: the principal executive office address
- irs_ein: the I.R.S. Employer Identification Number

Rules:
- Use ONLY information explicitly present in the text below.
- If a field genuinely cannot be found in the text, use null for that field — do NOT guess or fabricate a value.

{format_instruction}
The JSON object must have exactly these keys: {json.dumps(FIELDS)}

Cover page text:
{cover_text}

JSON:"""


def extract_metadata_for_pdf(pdf_path: Path) -> dict:
    cover_text = extract_cover_pages_text(pdf_path)

    raw = ""
    for attempt, strict in enumerate((False, True)):
        prompt = build_prompt(cover_text, strict)
        response = ollama.chat(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _extract_json(response["message"]["content"])

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return {field: parsed.get(field) for field in FIELDS}

    return {field: None for field in FIELDS}


def main():
    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    all_metadata = {}
    for pdf_path in pdf_paths:
        print(f"Extracting metadata from {pdf_path.name}...")
        metadata = extract_metadata_for_pdf(pdf_path)
        all_metadata[pdf_path.name] = metadata

    OUTPUT_PATH.write_text(json.dumps(all_metadata, indent=2), encoding="utf-8")

    print("\n=== Extracted Company Metadata ===")
    print(json.dumps(all_metadata, indent=2))


if __name__ == "__main__":
    main()
