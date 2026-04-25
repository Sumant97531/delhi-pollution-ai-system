"""
rag.py
------
Retrieval logic over a local knowledge base text file.
"""

import os
import re

# ── Path ──────────────────────────────────────────────────────────────────────
KB_PATH = os.path.join(
    os.path.dirname(__file__),  # notebooks/src/
    "..", "..",                 # go up TWO levels to DELHI AQI STUDY/
    "data", "knowledge_base.txt"
)

# ── Parser ────────────────────────────────────────────────────────────────────

def _load_kb(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = []
    blocks = re.split(r"\n## SECTION:", raw)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines  = block.splitlines()
        header = lines[0].strip().lstrip("#").strip()
        body   = "\n".join(lines[1:]).strip()

        text_lower = (header + " " + body).lower()
        keywords   = set(re.findall(r"[a-z0-9][a-z0-9_.]{1,}", text_lower))

        chunks.append({
            "header"  : header,
            "content" : body,
            "keywords": keywords,
        })

    return chunks


_KB_CHUNKS = _load_kb(KB_PATH)

# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(chunk: dict, query_tokens: set) -> int:
    overlap       = len(chunk["keywords"] & query_tokens)
    header_tokens = set(chunk["header"].lower().split())
    bonus         = 5 if header_tokens & query_tokens else 0
    return overlap + bonus

# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3) -> list:
    query_tokens = set(re.findall(r"[a-z0-9][a-z0-9_.]{1,}", query.lower()))

    scored = [
        {
            "header" : chunk["header"],
            "content": chunk["content"],
            "score"  : _score(chunk, query_tokens),
        }
        for chunk in _KB_CHUNKS
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [s for s in scored[:top_k] if s["score"] > 0]


def retrieve_for_pollutants(pollutants: list) -> list:
    results = []
    for p in pollutants:
        query  = p.replace(".", "").lower()
        chunks = retrieve(query, top_k=1)
        if chunks:
            results.append(chunks[0])
    return results


def format_context(chunks: list) -> str:
    if not chunks:
        return "No relevant context found in knowledge base."

    parts = []
    for c in chunks:
        parts.append(f"[{c['header']}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = retrieve("what causes PM2.5 in Delhi and what are health effects?")
    for r in results:
        print(f"\n=== {r['header']} (score: {r['score']}) ===")
        print(r["content"][:300])