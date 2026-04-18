"""
example_usage.py
----------------
Demonstrates all capabilities of the GenAI AQI decision system.
Run this after ensuring:
  1. ollama serve        (Ollama running locally)
  2. ollama pull llama3  (model downloaded)
  3. delhi_clean.csv     exists in data/processed/
  4. xgb_model.pkl       exists in models/

Usage:
  python app/example_usage.py
"""
#%%
import os
import sys
import json
import pandas as pd

# ── Resolve paths ─────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\suman\OneDrive\Documents\Projects\DELHI AQI STUDY"
APP_DIR  = os.path.join(BASE_DIR, "app")
SRC_DIR  = os.path.join(BASE_DIR, "notebooks", "src")

sys.path.insert(0, APP_DIR)
sys.path.insert(0, SRC_DIR)

from ollama_app import answer_query, compare_policies, _X, FEATURES

# ── Pick a sample row ─────────────────────────────────────────────────────────
# Using row index 200 — a real day from delhi_clean.csv
sample = _X.iloc[200]

print("=" * 65)
print("DELHI AQI — GenAI DECISION SYSTEM")
print("=" * 65)
print("\nToday's pollution readings:")
for f in FEATURES:
    print(f"  {f:8s}: {round(float(sample[f]), 2)}")
#%%
# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Single pollutant reduction query
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("QUERY 1: Single pollutant simulation")
print("─" * 65)

query1  = "What happens to AQI if we reduce PM2.5 by 30%?"
result1 = answer_query(sample, query1)

print("\n── STRUCTURED OUTPUT ──────────────────────────────────────")
print(json.dumps(result1["structured"], indent=2))

print("\n── LLM RESPONSE ───────────────────────────────────────────")
print(result1["llm_response"])
#%%
# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: Multi-pollutant reduction
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("QUERY 2: Multi-pollutant simulation")
print("─" * 65)

query2  = "What if we cut PM2.5 by 20% and NO2 by 15%? What would change?"
result2 = answer_query(sample, query2)

print("\n── STRUCTURED OUTPUT ──────────────────────────────────────")
print(json.dumps(result2["structured"], indent=2))

print("\n── LLM RESPONSE ───────────────────────────────────────────")
print(result2["llm_response"])
#%%
# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3: Explain-only (no simulation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("QUERY 3: Explain-only (SHAP + KB, no simulation)")
print("─" * 65)

query3  = "Why is AQI so high today and what are the main sources?"
result3 = answer_query(sample, query3)

print("\n── STRUCTURED OUTPUT ──────────────────────────────────────")
print(json.dumps(result3["structured"], indent=2))

print("\n── LLM RESPONSE ───────────────────────────────────────────")
print(result3["llm_response"])
#%%
# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 4: Policy comparison (no LLM — pure numerical)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("POLICY COMPARISON (numerical, no LLM call)")
print("─" * 65)

policies = [
    {"PM2.5": 10},
    {"PM2.5": 20},
    {"PM2.5": 30},
    {"PM10" : 20},
    {"NO2"  : 20},
    {"CO"   : 20},
    {"PM2.5": 20, "NO2": 10},
    {"PM2.5": 20, "PM10": 15, "CO": 10},
]

comparison_df = compare_policies(sample, policies)
print("\n" + comparison_df.to_string(index=False))
#%%
# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 5: Ozone query (tests alias mapping)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("QUERY 5: Alias test — 'ozone' → O3")
print("─" * 65)

query5  = "If ozone drops by 25%, what effect does it have on AQI?"
result5 = answer_query(sample, query5)

print("\n── STRUCTURED OUTPUT ──────────────────────────────────────")
print(json.dumps(result5["structured"], indent=2))

print("\n── LLM RESPONSE ───────────────────────────────────────────")
print(result5["llm_response"])

print("\n" + "=" * 65)
print("All examples completed.")
print("=" * 65)