#%%
"""
ollama_app.py
-------------
GenAI-powered explainable AQI decision system.

Flow for answer_query(row, user_query):
  1. Parse user intent  → extract pollutant names + % reductions from query
  2. Run simulation     → get new AQI and delta (calls simulate_change)
  3. Run SHAP explain   → get top contributing pollutants for this row
  4. Retrieve context   → fetch relevant knowledge base chunks via rag.py
  5. Build prompt       → assemble all real outputs into a grounded prompt
  6. Call Ollama        → stream response from local LLM (llama3)
  7. Return structured  → JSON + natural language response

No fake data. All numbers come from the real model and real SHAP values.
"""
import sys
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
#%%
import re
import json
import requests
import pandas as pd
import numpy as np
import joblib
import shap
import os
import sys

# ── Resolve src/ path without __file__ (works in Jupyter + .py both) ─────────
BASE_DIR = r"C:\Users\suman\OneDrive\Documents\Projects\DELHI AQI STUDY"
SRC_DIR  = os.path.join(BASE_DIR, "notebooks", "src")
sys.path.insert(0, SRC_DIR)

from rag import retrieve_for_pollutants, retrieve, format_context
#%%
# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"          # change to "mistral" or "phi3" if preferred

# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR      = r"C:\Users\suman\OneDrive\Documents\Projects\DELHI AQI STUDY"
MODEL_PATH    = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
DATA_PATH     = os.path.join(BASE_DIR, "data", "processed", "delhi_clean.csv")

# ── Feature list (must match model.py exactly) ────────────────────────────────
FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3']

# ── Load model + data once ────────────────────────────────────────────────────
_model = joblib.load(MODEL_PATH)
_df    = pd.read_csv(DATA_PATH)
_X     = _df[FEATURES].dropna().copy()

# Build SHAP explainer once (background sample for speed)
_background = shap.sample(_X, 100, random_state=42)
_explainer  = shap.Explainer(_model, _background)

#%%
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — INTENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

# Maps natural language aliases → canonical feature names
_ALIAS_MAP = {
    "particulate matter 2.5": "PM2.5", "fine particles": "PM2.5",
    "pm2.5": "PM2.5", "pm 2.5": "PM2.5",
    "particulate matter 10": "PM10",   "coarse particles": "PM10",
    "pm10": "PM10",   "pm 10": "PM10",
    "nitrogen dioxide": "NO2", "no2": "NO2",
    "nitric oxide": "NO",      "no": "NO",
    "ammonia": "NH3",          "nh3": "NH3",
    "carbon monoxide": "CO",   "co": "CO",
    "sulphur dioxide": "SO2",  "sulfur dioxide": "SO2", "so2": "SO2",
    "ozone": "O3",             "o3": "O3",
}


def parse_intent(user_query: str) -> dict:
    """
    Extracts pollutant → reduction_percent pairs from a natural language query.

    Examples handled:
      "reduce PM2.5 by 20%"              → {"PM2.5": 20}
      "cut NO2 by 15% and CO by 10%"     → {"NO2": 15, "CO": 10}
      "what happens if ozone drops 30?"  → {"O3": 30}
      "what is causing AQI?"             → {}  (explain-only intent)

    Returns
    -------
    dict with keys:
      "changes"     : {feature: percent, ...}  — empty if no reductions found
      "intent_type" : "simulate" | "explain" | "both"
    """
    q_lower = user_query.lower()
    changes = {}

    # Pattern: <pollutant_name> ... <number> ... %
    # Also handles: reduce X by N, cut X by N%, drop X N%
    for alias, canonical in _ALIAS_MAP.items():
        if alias in q_lower:
            # Look for a number near this alias mention (within 60 chars)
            idx   = q_lower.index(alias)
            window = q_lower[max(0, idx-20): idx + len(alias) + 60]
            nums  = re.findall(r"\b(\d+(?:\.\d+)?)\s*%?", window)
            if nums:
                pct = float(nums[0])
                if 0 < pct <= 100:
                    changes[canonical] = pct

    # Determine intent type
    has_action_verb = any(w in q_lower for w in [
        "reduce", "cut", "drop", "decrease", "lower",
        "if", "what if", "simulate", "what happens"
    ])

    if changes and has_action_verb:
        intent_type = "simulate"
    elif not changes:
        intent_type = "explain"
    else:
        intent_type = "both"

    return {"changes": changes, "intent_type": intent_type}

#%%
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SIMULATION (wraps simulate_change from simulation.py logic inline)
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate(row: pd.Series, changes: dict) -> dict:
    """
    Applies % reductions to specified features and returns AQI delta.
    Inline implementation — does NOT reimport simulation.py to avoid
    circular import issues, but uses identical logic.
    """
    modified = row.copy()
    for feature, pct in changes.items():
        if feature in modified.index:
            modified[feature] *= (1 - pct / 100)

    original_df = pd.DataFrame([row],      columns=FEATURES)
    modified_df = pd.DataFrame([modified], columns=FEATURES)

    original_aqi = float(_model.predict(original_df)[0])
    new_aqi      = float(_model.predict(modified_df)[0])

    return {
        "original_aqi": round(original_aqi, 2),
        "new_aqi"     : round(new_aqi,      2),
        "delta"       : round(original_aqi - new_aqi, 2),
        "modified_values": {
            f: round(float(modified[f]), 2) for f in changes
        }
    }

#%%
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SHAP EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════════

def _explain(row: pd.Series, top_k: int = 3) -> dict:
    """
    Returns SHAP-based top contributing pollutants for this specific row.
    Uses the real model and real SHAP values — no approximations.
    """
    row_df    = pd.DataFrame([row], columns=FEATURES)
    shap_vals = _explainer(row_df)

    contribs = pd.Series(shap_vals.values[0], index=FEATURES)

    top_positive = (
        contribs[contribs > 0]
        .sort_values(ascending=False)
        .head(top_k)
    )

    return {
        "top_contributors"    : top_positive.index.tolist(),
        "shap_values"         : {f: round(float(v), 3) for f, v in contribs.items()},
        "top_shap_values"     : {f: round(float(v), 3) for f, v in top_positive.items()},
        "base_value"          : round(float(shap_vals.base_values[0]), 2),
    }

#%%
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4+5 — PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_prompt(
    row         : pd.Series,
    user_query  : str,
    intent      : dict,
    sim_result  : dict | None,
    shap_result : dict,
    kb_context  : str,
) -> str:
    """
    Assembles a fully grounded prompt from real model outputs + KB context.
    The LLM is instructed to use ONLY this data — no hallucination.
    """

    # Current pollutant readings
    readings = "\n".join(
        f"  {f}: {round(float(row[f]), 2)}" for f in FEATURES
    )

    # SHAP section
    shap_lines = "\n".join(
        f"  {f}: SHAP = {v} (pushing AQI {'UP' if v > 0 else 'DOWN'})"
        for f, v in shap_result["shap_values"].items()
    )

    # Simulation section (only if a simulation was run)
    sim_section = ""
    if sim_result:
        changes_str = ", ".join(
            f"{f} reduced by {pct}%" for f, pct in intent["changes"].items()
        )
        sim_section = f"""
SIMULATION RESULT:
  Policy applied : {changes_str}
  Original AQI   : {sim_result['original_aqi']}
  New AQI        : {sim_result['new_aqi']}
  AQI reduction  : {sim_result['delta']} points
  Modified values: {json.dumps(sim_result['modified_values'])}
"""

    prompt = f"""You are an expert air quality analyst for Delhi, India.
You must answer ONLY using the data provided below. Do NOT invent numbers or facts.

USER QUESTION:
{user_query}

CURRENT POLLUTION READINGS (Delhi):
{readings}

MODEL OUTPUTS — SHAP FEATURE CONTRIBUTIONS:
  (positive = pushes AQI higher, negative = pushes AQI lower)
  Base AQI (model average): {shap_result['base_value']}
{shap_lines}

TOP POLLUTANTS DRIVING AQI TODAY:
  {', '.join(shap_result['top_contributors'])}
{sim_section}
KNOWLEDGE BASE CONTEXT (sources, health impacts, policies):
{kb_context}

INSTRUCTIONS:
Respond in this exact structure:

### AQI ASSESSMENT
[State current AQI situation using the SHAP numbers above. Mention which pollutants are driving it and by how much.]

### SIMULATION RESULT
[If a simulation was run, explain the numerical AQI change and what it means health-wise using the AQI threshold definitions from context. If no simulation, skip this section.]

### POLLUTANT SOURCES
[For each top contributor, state its real-world sources from the knowledge base context only.]

### HEALTH IMPLICATIONS
[State health impact for the current AQI level using knowledge base context.]

### ACTIONABLE POLICY SUGGESTIONS
[Give 3-5 specific, actionable policy recommendations from the knowledge base context. Be specific to the top pollutants identified by SHAP.]

Do NOT add any disclaimer or generic text. Be precise and data-driven.
"""
    return prompt

#%%
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — OLLAMA CALL
# ═══════════════════════════════════════════════════════════════════════════════

def _call_ollama(prompt: str, stream: bool = False) -> str:
    """
    Sends prompt to Ollama local LLM and returns response text.
    Raises ConnectionError with clear message if Ollama is not running.
    """
    payload = {
        "model" : OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.1,    # low temp for factual, deterministic output
            "num_predict": 800,
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot reach Ollama. Make sure it is running:\n"
            "  ollama serve\n"
            f"  ollama pull {OLLAMA_MODEL}"
        )

    data = response.json()
    return data.get("response", "").strip()

#%%
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — MAIN PUBLIC FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def answer_query(row: pd.Series, user_query: str) -> dict:
    """
    Main entry point. Takes a data row (one day's pollution readings) and
    a natural language user query, and returns a fully grounded response.

    Parameters
    ----------
    row        : pd.Series with index = FEATURES (one row from delhi_clean.csv)
    user_query : natural language question or policy request

    Returns
    -------
    dict with keys:
      "intent"          : parsed intent (changes dict + intent_type)
      "simulation"      : simulation result dict (or None)
      "shap_explanation": SHAP output dict
      "retrieved_chunks": list of KB chunks used
      "llm_response"    : natural language response from LLM
      "structured"      : JSON-serialisable summary of all numerical outputs
    """

    print(f"\n[1/5] Parsing intent...")
    intent = parse_intent(user_query)
    print(f"      Intent: {intent}")

    print(f"[2/5] Running simulation..." if intent["changes"] else "[2/5] No simulation (explain-only query)")
    sim_result = _simulate(row, intent["changes"]) if intent["changes"] else None

    print(f"[3/5] Computing SHAP explanation...")
    shap_result = _explain(row, top_k=3)

    print(f"[4/5] Retrieving knowledge base context...")
    # Retrieve for top SHAP contributors + any pollutants in the query
    query_pollutants = list(intent["changes"].keys()) if intent["changes"] else []
    all_pollutants   = list(dict.fromkeys(
        shap_result["top_contributors"] + query_pollutants
    ))
    kb_chunks   = retrieve_for_pollutants(all_pollutants)
    # Also do a free-text retrieve for any residual intent
    extra_chunks = retrieve(user_query, top_k=2)
    # Deduplicate by header
    seen     = {c["header"] for c in kb_chunks}
    kb_chunks += [c for c in extra_chunks if c["header"] not in seen]
    kb_context = format_context(kb_chunks)

    print(f"[5/5] Calling Ollama ({OLLAMA_MODEL})...")
    prompt       = _build_prompt(row, user_query, intent, sim_result, shap_result, kb_context)
    llm_response = _call_ollama(prompt)

    # Build structured summary
    structured = {
        "original_aqi"      : sim_result["original_aqi"] if sim_result else None,
        "new_aqi"           : sim_result["new_aqi"]       if sim_result else None,
        "aqi_reduction"     : sim_result["delta"]         if sim_result else None,
        "policy_applied"    : intent["changes"],
        "top_contributors"  : shap_result["top_contributors"],
        "shap_values"       : shap_result["top_shap_values"],
        "kb_sections_used"  : [c["header"] for c in kb_chunks],
    }

    return {
        "intent"           : intent,
        "simulation"       : sim_result,
        "shap_explanation" : shap_result,
        "retrieved_chunks" : kb_chunks,
        "llm_response"     : llm_response,
        "structured"       : structured,
    }

#%%
# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-POLLUTANT BATCH HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def compare_policies(row: pd.Series, policy_list: list[dict]) -> pd.DataFrame:
    """
    Runs multiple pollutant reduction scenarios and returns a ranked DataFrame.
    Does NOT call the LLM — purely numerical comparison.

    Parameters
    ----------
    row         : one day's pollution readings
    policy_list : list of change dicts, e.g. [{"PM2.5": 20}, {"PM10": 30, "CO": 10}]

    Returns
    -------
    DataFrame sorted by AQI reduction (best first)
    """
    rows = []
    for policy in policy_list:
        result = _simulate(row, policy)
        rows.append({
            "policy"      : str(policy),
            "original_aqi": result["original_aqi"],
            "new_aqi"     : result["new_aqi"],
            "improvement" : result["delta"],
        })
    return (
        pd.DataFrame(rows)
        .sort_values("improvement", ascending=False)
        .reset_index(drop=True)
    )
# %%
