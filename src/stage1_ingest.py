import json
import re
import uuid
from collections import Counter
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"
OUTPUT_PATH = Path(__file__).parent / "chunks.json"

HEADER_FOOTER_MAX_LEN = 80
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def normalize_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"\d+", "#", line)
    return line.lower()


def extract_pages(reader: PdfReader) -> list[str]:
    return [page.extract_text() or "" for page in reader.pages]


def find_repeated_lines(pages: list[str]) -> set[str]:
    num_pages = len(pages)
    if num_pages < 2:
        return set()

    counts = Counter()
    for page_text in pages:
        seen_this_page = set()
        for raw_line in page_text.split("\n"):
            line = raw_line.strip()
            if not line or len(line) >= HEADER_FOOTER_MAX_LEN:
                continue
            norm = normalize_line(line)
            if norm and norm not in seen_this_page:
                seen_this_page.add(norm)
                counts[norm] += 1

    threshold = max(3, int(num_pages * 0.3))
    return {norm for norm, count in counts.items() if count >= threshold}


def strip_headers_footers(pages: list[str], repeated: set[str]) -> list[str]:
    cleaned_pages = []
    for page_text in pages:
        kept_lines = []
        for raw_line in page_text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if len(line) < HEADER_FOOTER_MAX_LEN and normalize_line(line) in repeated:
                continue
            kept_lines.append(line)
        cleaned_pages.append("\n".join(kept_lines))
    return cleaned_pages


def build_full_text_with_offsets(cleaned_pages: list[str]) -> tuple[str, list[tuple[int, int, int]]]:
    full_text_parts = []
    offsets = []  # (start, end, page_number) page_number is 1-indexed
    cursor = 0
    for i, page_text in enumerate(cleaned_pages, start=1):
        start = cursor
        full_text_parts.append(page_text)
        cursor += len(page_text)
        end = cursor
        offsets.append((start, end, i))
        # separator between pages
        full_text_parts.append("\n")
        cursor += 1
    return "".join(full_text_parts), offsets


def page_for_offset(offset: int, offsets: list[tuple[int, int, int]]) -> int:
    for start, end, page_number in offsets:
        if start <= offset < end:
            return page_number
    # fall back to nearest page if offset lands in a separator gap
    for start, end, page_number in offsets:
        if offset < end:
            return page_number
    return offsets[-1][2] if offsets else 1


def process_pdf(path: Path, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    reader = PdfReader(str(path))
    pages = extract_pages(reader)
    repeated = find_repeated_lines(pages)
    cleaned_pages = strip_headers_footers(pages, repeated)
    full_text, offsets = build_full_text_with_offsets(cleaned_pages)

    chunks = splitter.split_text(full_text)

    results = []
    cursor = 0
    for chunk_text in chunks:
        found = full_text.find(chunk_text, max(0, cursor - CHUNK_OVERLAP))
        if found == -1:
            found = full_text.find(chunk_text)
        if found == -1:
            found = cursor
        cursor = found + 1

        page_number = page_for_offset(found, offsets)
        results.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "source_file": path.name,
                "page_number": page_number,
                "text": chunk_text,
            }
        )
    return results


def main():
    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []
    for pdf_path in pdf_paths:
        all_chunks.extend(process_pdf(pdf_path, splitter))

    OUTPUT_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")

    total_documents = len(pdf_paths)
    total_chunks = len(all_chunks)
    lengths = [len(c["text"]) for c in all_chunks]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    print("=== Stage 1 Ingest Summary ===")
    print(f"Total documents processed: {total_documents}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Average chunk character length: {avg_len:.1f}")
    print(f"Min chunk length: {min_len}")
    print(f"Max chunk length: {max_len}")


if __name__ == "__main__":
    main()
