"""
rag.py
------
Retrieval logic over a local knowledge base text file.

Design:
  - Parses knowledge_base.txt into sections keyed by pollutant / topic
  - retrieve(query, top_k) returns the most relevant chunks
  - Retrieval strategy: keyword overlap scoring (deterministic, no external deps)
  - Called by ollama_app.py to ground LLM responses in factual content
"""
# %%
import os
import re
from typing import Optional

# ── Path ──────────────────────────────────────────────────────────────────────
# ── Path ──────────────────────────────────────────────────────────────────────

KB_PATH = os.path.join(
    os.path.dirname(__file__),   # notebooks/src/
    "..", "..",                  # ← go up TWO levels to DELHI AQI STUDY/
    "data", "knowledge_base.txt"
)
#%%


# ── Parser ────────────────────────────────────────────────────────────────────

def _load_kb(path: str) -> list[dict]:
    """
    Parses knowledge_base.txt into a list of section dicts.
    Each section = { "header": str, "content": str, "keywords": set[str] }
    Sections are delimited by '## SECTION:' lines.
    """
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

        # Build keyword set from header + all field values
        text_lower = (header + " " + body).lower()
        keywords   = set(re.findall(r"[a-z0-9][a-z0-9_.]{1,}", text_lower))

        chunks.append({
            "header"  : header,
            "content" : body,
            "keywords": keywords,
        })

    return chunks


# Load once at import time
_KB_CHUNKS: list[dict] = _load_kb(KB_PATH)

#%%
# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(chunk: dict, query_tokens: set[str]) -> int:
    """
    Keyword overlap score between query tokens and chunk keywords.
    Exact pollutant name match in header gets a 5x bonus.
    """
    overlap = len(chunk["keywords"] & query_tokens)
    header_tokens = set(chunk["header"].lower().split())
    bonus = 5 if header_tokens & query_tokens else 0
    return overlap + bonus

#%%
# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Returns top_k most relevant knowledge base chunks for the given query.

    Parameters
    ----------
    query  : free-text query (e.g. "what causes PM2.5 in Delhi?")
    top_k  : number of chunks to return

    Returns
    -------
    List of dicts with keys: header, content, score
    """
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
    top = [s for s in scored[:top_k] if s["score"] > 0]
    return top

#%%
def retrieve_for_pollutants(pollutants: list[str]) -> list[dict]:
    """
    Directly fetches knowledge base sections for a list of pollutant names.
    Used when we already know the top SHAP features.

    Parameters
    ----------
    pollutants : e.g. ["PM2.5", "PM10", "CO"]

    Returns
    -------
    List of matching knowledge base chunks
    """
    results = []
    for p in pollutants:
        query  = p.replace(".", "").lower()  # "pm25", "pm10", "no2" etc.
        chunks = retrieve(query, top_k=1)
        if chunks:
            results.append(chunks[0])
    return results

#%%
def format_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a single context string for the LLM prompt.
    """
    if not chunks:
        return "No relevant context found in knowledge base."

    parts = []
    for c in chunks:
        parts.append(f"[{c['header']}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)

#%%
# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = retrieve("what causes PM2.5 in Delhi and what are health effects?")
    for r in results:
        print(f"\n=== {r['header']} (score: {r['score']}) ===")
        print(r["content"][:300])
# %%
