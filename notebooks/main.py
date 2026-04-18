# ============================================================
# DELHI AQI — GenAI SYSTEM
# Complete notebook — run cells TOP TO BOTTOM in order
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1 — Install requests (run once, then can comment out)
# ─────────────────────────────────────────────────────────────
#%%
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "requests", "-q"], check=True)
print("requests ready ✅")

# ─────────────────────────────────────────────────────────────
# CELL 2 — All imports
# ─────────────────────────────────────────────────────────────
#%%
import re
import os
import json
import sys
import requests
import pandas as pd
import numpy as np
import joblib
import shap

print("All imports done ✅")

# ─────────────────────────────────────────────────────────────
# CELL 3 — Project paths (edit BASE_DIR if needed)
# ─────────────────────────────────────────────────────────────
#%%
BASE_DIR   = r"C:\Users\suman\OneDrive\Documents\Projects\DELHI AQI STUDY"
SRC_DIR    = os.path.join(BASE_DIR, "notebooks", "src")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "delhi_clean.csv")
KB_PATH    = os.path.join(BASE_DIR, "data", "knowledge_base.txt")

sys.path.insert(0, SRC_DIR)

print("BASE_DIR exists :", os.path.exists(BASE_DIR))
print("Model exists    :", os.path.exists(MODEL_PATH))
print("Data exists     :", os.path.exists(DATA_PATH))
print("KB exists       :", os.path.exists(KB_PATH))

# ─────────────────────────────────────────────────────────────
# CELL 4 — RAG system (knowledge base loader + retriever)
# ─────────────────────────────────────────────────────────────
#%%
def _load_kb(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = []
    blocks = re.split(r"\n## SECTION:", raw)

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines   = block.splitlines()
        header  = lines[0].strip().lstrip("#").strip()
        body    = "\n".join(lines[1:]).strip()
        text_lower = (header + " " + body).lower()
        keywords   = set(re.findall(r"[a-z0-9][a-z0-9_.]{1,}", text_lower))
        chunks.append({"header": header, "content": body, "keywords": keywords})

    return chunks

KB_CHUNKS = _load_kb(KB_PATH)
print(f"Knowledge base loaded ✅  ({len(KB_CHUNKS)} sections)")

def _score(chunk, query_tokens):
    overlap       = len(chunk["keywords"] & query_tokens)
    header_tokens = set(chunk["header"].lower().split())
    bonus         = 5 if header_tokens & query_tokens else 0
    return overlap + bonus

def retrieve(query: str, top_k: int = 3) -> list:
    query_tokens = set(re.findall(r"[a-z0-9][a-z0-9_.]{1,}", query.lower()))
    scored = [{"header": c["header"], "content": c["content"],
               "score": _score(c, query_tokens)} for c in KB_CHUNKS]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [s for s in scored[:top_k] if s["score"] > 0]

def retrieve_for_pollutants(pollutants: list) -> list:
    results = []
    for p in pollutants:
        chunks = retrieve(p.replace(".", "").lower(), top_k=1)
        if chunks:
            results.append(chunks[0])
    return results

def format_context(chunks: list) -> str:
    if not chunks:
        return "No relevant context found."
    return "\n\n---\n\n".join(f"[{c['header']}]\n{c['content']}" for c in chunks)

print("RAG functions ready ✅")

# ─────────────────────────────────────────────────────────────
# CELL 5 — Load model, data, SHAP explainer
# ─────────────────────────────────────────────────────────────
#%%
FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3']

_model = joblib.load(MODEL_PATH)
_df    = pd.read_csv(DATA_PATH)
_X     = _df[FEATURES].dropna().copy()

_background = shap.sample(_X, 100, random_state=42)
_explainer  = shap.Explainer(_model, _background)

print("Model loaded ✅")
print("Data shape  :", _X.shape)
print("SHAP ready  ✅")

# ─────────────────────────────────────────────────────────────
# CELL 6 — Simulation function
# ─────────────────────────────────────────────────────────────
#%%
def simulate(row: pd.Series, changes: dict) -> dict:
    """
    changes = {"PM2.5": 20, "NO2": 10}
    Reduces each feature by that percentage, returns AQI delta.
    """
    modified = row.copy()
    for feature, pct in changes.items():
        modified[feature] *= (1 - pct / 100)

    orig_df = pd.DataFrame([row],      columns=FEATURES)
    mod_df  = pd.DataFrame([modified], columns=FEATURES)

    orig_aqi = float(_model.predict(orig_df)[0])
    new_aqi  = float(_model.predict(mod_df)[0])

    return {
        "original_aqi": round(orig_aqi, 2),
        "new_aqi"     : round(new_aqi,  2),
        "delta"       : round(orig_aqi - new_aqi, 2),
    }

print("Simulation function ready ✅")

# ─────────────────────────────────────────────────────────────
# CELL 7 — SHAP explain function
# ─────────────────────────────────────────────────────────────
#%%
def explain(row: pd.Series, top_k: int = 3) -> dict:
    row_df    = pd.DataFrame([row], columns=FEATURES)
    shap_vals = _explainer(row_df)
    contribs  = pd.Series(shap_vals.values[0], index=FEATURES)
    top       = contribs[contribs > 0].sort_values(ascending=False).head(top_k)

    return {
        "top_contributors": top.index.tolist(),
        "shap_values"     : {f: round(float(v), 3) for f, v in contribs.items()},
        "top_shap_values" : {f: round(float(v), 3) for f, v in top.items()},
        "base_value"      : round(float(shap_vals.base_values[0]), 2),
    }

print("Explain function ready ✅")

# ─────────────────────────────────────────────────────────────
# CELL 8 — Intent parser
# ─────────────────────────────────────────────────────────────
#%%
ALIAS_MAP = {
    "pm2.5": "PM2.5", "pm 2.5": "PM2.5", "fine particles": "PM2.5",
    "pm10" : "PM10",  "pm 10" : "PM10",  "coarse particles": "PM10",
    "no2"  : "NO2",   "nitrogen dioxide": "NO2",
    "no"   : "NO",    "nitric oxide": "NO",
    "nh3"  : "NH3",   "ammonia": "NH3",
    "co"   : "CO",    "carbon monoxide": "CO",
    "so2"  : "SO2",   "sulphur dioxide": "SO2", "sulfur dioxide": "SO2",
    "o3"   : "O3",    "ozone": "O3",
}

def parse_intent(query: str) -> dict:
    q    = query.lower()
    changes = {}

    for alias, canonical in ALIAS_MAP.items():
        if alias in q:
            idx    = q.index(alias)
            window = q[max(0, idx - 20): idx + len(alias) + 60]
            nums   = re.findall(r"\b(\d+(?:\.\d+)?)\s*%?", window)
            if nums:
                pct = float(nums[0])
                if 0 < pct <= 100:
                    changes[canonical] = pct

    has_action = any(w in q for w in [
        "reduce", "cut", "drop", "decrease", "lower", "if", "simulate"
    ])

    intent_type = "simulate" if (changes and has_action) else (
                  "explain"  if not changes else "both")

    return {"changes": changes, "intent_type": intent_type}

print("Intent parser ready ✅")

# ─────────────────────────────────────────────────────────────
# CELL 9 — Ollama call + prompt builder
# ─────────────────────────────────────────────────────────────
#%%
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"  # change to "mistral" or "phi3" if preferred

def _build_prompt(row, user_query, intent, sim_result, shap_result, kb_context):
    top = shap_result["top_contributors"]
    top_vals = shap_result["top_shap_values"]

    sim_line = ""
    if sim_result:
        changes_str = ", ".join(f"{f} -{p}%" for f, p in intent["changes"].items())
        sim_line = f"Policy: {changes_str} | AQI: {sim_result['original_aqi']} → {sim_result['new_aqi']} (improvement: {sim_result['delta']} points)"

    # Keep KB context short — only first 300 chars
    kb_short = kb_context[:300] if kb_context else ""

    return f"""Delhi AQI analyst. Use only given data.

Question: {user_query}

Top pollutants driving AQI: {top} with SHAP values {top_vals}
{sim_line}

Context: {kb_short}

Answer briefly with:
1. What is driving AQI
2. Simulation result (if any)
3. Main sources
4. Health risk
5. Two policy actions
"""

def call_ollama(prompt: str) -> str:
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model"  : OLLAMA_MODEL,
            "prompt" : prompt,
            "stream" : False,
            "options": {"temperature": 0.1, "num_predict": 300}
        }, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "❌ Ollama not running. Open terminal and run: ollama serve"

print("Ollama functions ready ✅")

# ─────────────────────────────────────────────────────────────
# CELL 10 — Main answer_query function
# ─────────────────────────────────────────────────────────────
#%%
def answer_query(row: pd.Series, user_query: str) -> dict:
    print(f"\n[1/5] Parsing intent...")
    intent = parse_intent(user_query)
    print(f"      {intent}")

    print(f"[2/5] Simulation..." if intent["changes"] else "[2/5] No simulation needed")
    sim = simulate(row, intent["changes"]) if intent["changes"] else None

    print(f"[3/5] SHAP explanation...")
    shap_out = explain(row, top_k=3)

    print(f"[4/5] Retrieving knowledge base...")
    pollutants = list(dict.fromkeys(
        shap_out["top_contributors"] + list(intent["changes"].keys())
    ))
    kb_chunks  = retrieve_for_pollutants(pollutants)
    extra      = retrieve(user_query, top_k=2)
    seen       = {c["header"] for c in kb_chunks}
    kb_chunks += [c for c in extra if c["header"] not in seen]
    kb_ctx     = format_context(kb_chunks)

    print(f"[5/5] Calling Ollama ({OLLAMA_MODEL})...")
    prompt   = _build_prompt(row, user_query, intent, sim, shap_out, kb_ctx)
    response = call_ollama(prompt)

    structured = {
        "original_aqi"    : sim["original_aqi"]      if sim else None,
        "new_aqi"         : sim["new_aqi"]            if sim else None,
        "aqi_reduction"   : sim["delta"]              if sim else None,
        "policy_applied"  : intent["changes"],
        "top_contributors": shap_out["top_contributors"],
        "shap_values"     : shap_out["top_shap_values"],
        "kb_used"         : [c["header"] for c in kb_chunks],
    }

    return {
        "structured"  : structured,
        "llm_response": response,
    }

print("answer_query ready ✅")
print("\n✅ ALL CELLS LOADED — ready to query!")

# ─────────────────────────────────────────────────────────────
# CELL 11 — Check Ollama is running
# ─────────────────────────────────────────────────────────────
#%%
try:
    r = requests.get("http://localhost:11434", timeout=3)
    print("Ollama is running ✅")
except:
    print("❌ Ollama NOT running.")
    print("   Open a NEW terminal and run:")
    print("   ollama serve")
    print("   Then re-run this cell to confirm.")

# ─────────────────────────────────────────────────────────────
# CELL 12 — RUN YOUR QUERY HERE
# ─────────────────────────────────────────────────────────────

#%%
sample = _X.iloc[200]

result = answer_query(sample, "What happens if we reduce PM2.5 by 20%?")

print("\n── STRUCTURED OUTPUT ───────────────────────────────────")
print(json.dumps(result["structured"], indent=2))

print("\n── LLM RESPONSE ────────────────────────────────────────")
print(result["llm_response"])
# %%
import xgboost, shap
print(xgboost.__version__)
print(shap.__version__)
# %%
