import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

chunks = json.loads((Path(__file__).parent.parent / "src" / "chunks.json").read_text(encoding="utf-8"))

atlassian_chunks = [c for c in chunks if c["source_file"] == "attlasian-10k-2026.pdf"]

team_chunk = None
for c in atlassian_chunks:
    if "TEAM Nasdaq Global Select Market" in c["text"]:
        team_chunk = c
        break

page84_chunks = [c for c in atlassian_chunks if c["page_number"] == 84]

print("=== TEAM/Nasdaq chunk (page 1) ===")
if team_chunk:
    print(f"chunk_id={team_chunk['chunk_id']} page={team_chunk['page_number']}")
    print(f"length={len(team_chunk['text'])} characters")
    print(f"text:\n{team_chunk['text']}\n")
else:
    print("NOT FOUND")

print("=== Page 84 chunk(s) (competing chunk) ===")
for c in page84_chunks:
    print(f"chunk_id={c['chunk_id']} page={c['page_number']}")
    print(f"length={len(c['text'])} characters")
    print(f"text:\n{c['text']}\n")
    print("-" * 80)

if team_chunk and page84_chunks:
    longest_84 = max(page84_chunks, key=lambda c: len(c["text"]))
    print("=== Comparison ===")
    print(f"TEAM/Nasdaq chunk (page 1): {len(team_chunk['text'])} characters")
    print(f"Page 84 chunk (longest of {len(page84_chunks)}): {len(longest_84['text'])} characters")
    diff = len(longest_84["text"]) - len(team_chunk["text"])
    print(f"Difference: page 84 chunk is {diff} characters longer")
