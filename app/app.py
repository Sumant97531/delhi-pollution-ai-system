"""
app.py — Delhi AQI Intelligence System
Streamlit dashboard with XGBoost + SHAP + Ollama LLM
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import requests
import re
import os
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhi AQI Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\suman\OneDrive\Documents\Projects\DELHI AQI STUDY"
MODEL_PATH   = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
DATA_PATH    = os.path.join(BASE_DIR, "data", "processed", "delhi_clean.csv")
KB_PATH      = os.path.join(BASE_DIR, "data", "knowledge_base.txt")
FEATURES     = ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3']
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"
MAX_REDUCTION_PCT = 40

# ─────────────────────────────────────────────────────────────
# AQI HELPERS
# ─────────────────────────────────────────────────────────────
def aqi_category(aqi):
    aqi = max(0, aqi)
    if aqi <= 50:   return "Good",         "#2d6a4f", "#b7e4c7", "🌿"
    if aqi <= 100:  return "Satisfactory", "#52b788", "#d8f3dc", "🍃"
    if aqi <= 200:  return "Moderate",     "#e9c46a", "#fff3b0", "🌤️"
    if aqi <= 300:  return "Poor",         "#f4a261", "#ffe8d6", "⚠️"
    if aqi <= 400:  return "Very Poor",    "#e63946", "#ffd6d6", "🔴"
    return "Severe",                       "#6d0c0c", "#f5c0c0", "🔥"

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1117;
    color: #e8eaf0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

.aqi-header {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 2rem;
}
.aqi-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #a8edea, #fed6e3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.aqi-header p {
    color: rgba(255,255,255,0.45);
    font-size: 0.95rem;
    margin-top: 0.4rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 300;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-bottom: 0.8rem;
    margin-top: 0.2rem;
}
.aqi-card {
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.06);
}
.aqi-number {
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.2rem 0;
}
.aqi-label {
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.6;
    font-weight: 500;
}
.aqi-category { font-size: 1.4rem; font-weight: 600; margin-top: 0.5rem; }
.delta-card {
    border-radius: 14px;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.8rem;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.04);
}
.delta-label { font-size: 0.78rem; color: rgba(255,255,255,0.4); letter-spacing: 1px; text-transform: uppercase; }
.delta-value { font-size: 1.6rem; font-weight: 600; font-family: 'Playfair Display', serif; }
.insight-card {
    background: linear-gradient(135deg, rgba(168,237,234,0.06), rgba(254,214,227,0.06));
    border: 1px solid rgba(168,237,234,0.15);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
    line-height: 1.8;
    font-size: 0.95rem;
    color: rgba(255,255,255,0.82);
}
.insight-card .insight-header {
    font-size: 0.7rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #a8edea;
    font-weight: 600;
    margin-bottom: 0.8rem;
}
.warning-banner {
    background: rgba(246,173,85,0.1);
    border: 1px solid rgba(246,173,85,0.3);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-size: 0.78rem;
    color: rgba(246,173,85,0.9);
    margin-bottom: 0.8rem;
}
div.stButton > button {
    background: linear-gradient(135deg, #a8edea22, #fed6e322);
    border: 1px solid rgba(168,237,234,0.3);
    color: #a8edea;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 1px;
    padding: 0.5rem 1.2rem;
    width: 100%;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #a8edea33, #fed6e333);
    border-color: rgba(168,237,234,0.6);
}
.stSlider label { font-size: 0.82rem !important; color: rgba(255,255,255,0.55) !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
.stSelectbox label { font-size: 0.82rem !important; color: rgba(255,255,255,0.4) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD RESOURCES (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH)
    return raw[FEATURES].dropna().reset_index(drop=True)

@st.cache_resource
def load_explainer(_model, _X):
    return shap.TreeExplainer(_model)

@st.cache_data
def load_kb():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return f.read()

@st.cache_data
def build_day_options(model_path, data_path):
    m  = joblib.load(model_path)
    df = pd.read_csv(data_path)[FEATURES].dropna().reset_index(drop=True)
    options = {}
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        inp = pd.DataFrame([row], columns=FEATURES).astype(float)
        p   = float(np.clip(m.predict(inp)[0], 0, 999))
        if p > 50:
            cat   = aqi_category(p)[0]
            label = f"Row {i:>4d}  │  AQI {p:>5.0f}  │  {cat}"
            options[label] = i
    return options

# ─────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────
def predict_aqi(model, row_dict):
    input_df = pd.DataFrame([row_dict], columns=FEATURES).astype(float)
    return float(np.clip(model.predict(input_df)[0], 0, 999))

def get_shap(explainer, row_dict):
    input_df = pd.DataFrame([row_dict], columns=FEATURES).astype(float)
    vals     = explainer.shap_values(input_df)
    s        = pd.Series(vals[0], index=FEATURES)
    base     = float(explainer.expected_value)
    return s, base

def apply_reductions(baseline, reductions_dict):
    return {
        f: baseline[f] * (1 - min(reductions_dict.get(f, 0), MAX_REDUCTION_PCT) / 100)
        for f in FEATURES
    }

def compute_delta(orig_aqi, new_aqi):
    improvement = float(np.clip(orig_aqi - new_aqi, 0, orig_aqi))
    pct = float(np.clip(improvement / orig_aqi * 100, 0, 100)) if orig_aqi > 0 else 0.0
    return improvement, pct

# ─────────────────────────────────────────────────────────────
# RAG
# ─────────────────────────────────────────────────────────────
def retrieve_kb(kb_text, pollutants, max_chars=400):
    blocks  = re.split(r"\n## SECTION:", kb_text)
    matched = []
    for block in blocks:
        for p in pollutants:
            if p.replace(".", "").upper() in block.upper():
                lines = block.strip().splitlines()
                body  = "\n".join(lines[1:])[:200]
                matched.append(body)
                break
    return "\n---\n".join(matched)[:max_chars]

# ─────────────────────────────────────────────────────────────
# OLLAMA — FIX 3: plain text errors, no markdown fences
# ─────────────────────────────────────────────────────────────
def call_ollama(top_features, shap_vals_dict, sim_orig, sim_new, delta, kb_ctx):
    top_str  = ", ".join(top_features)
    shap_str = ", ".join(
        f"{f}={v:.1f}" for f, v in shap_vals_dict.items() if f in top_features
    )
    prompt = (
        f"Delhi AQI expert. Be concise (3-4 sentences max).\n\n"
        f"Top pollutants: {top_str}\n"
        f"SHAP values: {shap_str}\n"
        f"AQI change: {sim_orig:.0f} to {sim_new:.0f} (improvement {delta:.0f} pts)\n"
        f"Context: {kb_ctx}\n\n"
        f"Give: what drives AQI, health risk at current level, one key action."
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model"  : OLLAMA_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {"temperature": 0.1, "num_predict": 200},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "Ollama is not running. Open a terminal and run: ollama serve"
    except requests.exceptions.Timeout:
        return (
            f"Ollama timed out after 120 seconds. "
            f"Try a lighter model: run 'ollama pull phi3' in terminal, "
            f"then change OLLAMA_MODEL to 'phi3' in app.py."
        )
    except Exception as e:
        return f"Unexpected error: {e}"

# ─────────────────────────────────────────────────────────────
# SHAP CHART
# ─────────────────────────────────────────────────────────────
def shap_chart(shap_series):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    sorted_s = shap_series.sort_values()
    colors   = ["#e63946" if v > 0 else "#52b788" for v in sorted_s]
    ax.barh(sorted_s.index, sorted_s.values, color=colors, height=0.55, edgecolor="none")

    for i, (feat, val) in enumerate(sorted_s.items()):
        ax.text(
            val + (0.4 if val >= 0 else -0.4), i,
            f"{val:+.1f}", va="center",
            ha="left" if val >= 0 else "right",
            color="white", fontsize=7, alpha=0.75,
        )

    ax.axvline(0, color=(1, 1, 1, 0.15), linewidth=0.8)
    ax.tick_params(axis="both", colors=(1, 1, 1, 0.5), labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("SHAP value (AQI impact)", color=(1, 1, 1, 0.35), fontsize=7, labelpad=6)
    for lbl in ax.get_yticklabels():
        lbl.set_color((1, 1, 1, 0.65))
    ax.grid(axis="x", color=(1, 1, 1, 0.05), linewidth=0.5)
    fig.tight_layout(pad=0.6)
    return fig

# ─────────────────────────────────────────────────────────────
# SENSITIVITY CHART — FIX 1: dedicated function, explicit limits
# ─────────────────────────────────────────────────────────────
def sensitivity_chart(baseline, model):
    steps     = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    sens_vals = []
    for p in steps:
        mod = {f: baseline[f] * (1 - (p / 100 if f == "PM2.5" else 0)) for f in FEATURES}
        sens_vals.append(float(np.clip(predict_aqi(model, mod), 0, 999)))

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    ax.fill_between(steps, sens_vals, alpha=0.12, color="#a8edea")
    ax.plot(
        steps, sens_vals,
        color="#a8edea", linewidth=2, marker="o",
        markersize=4, markerfacecolor="#0f1117",
        markeredgecolor="#a8edea", markeredgewidth=1.5,
    )
    for x, y in zip(steps, sens_vals):
        ax.annotate(
            f"{int(y)}", (x, y),
            textcoords="offset points", xytext=(0, 7),
            ha="center", fontsize=7, color=(1, 1, 1, 0.55),
        )

    ax.set_xlabel("PM2.5 reduction (%)", color=(1, 1, 1, 0.4), fontsize=7)
    ax.set_ylabel("Predicted AQI",       color=(1, 1, 1, 0.4), fontsize=7)
    ax.tick_params(axis="both", colors=(1, 1, 1, 0.4), labelsize=7)
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.08))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=(1, 1, 1, 0.04), linewidth=0.5)
    ax.set_xlim(-1, 41)
    y_pad = max(sens_vals) * 0.12
    ax.set_ylim(max(0, min(sens_vals) - y_pad), max(sens_vals) + y_pad * 2)
    fig.tight_layout(pad=0.8)
    return fig

# ─────────────────────────────────────────────────────────────
# LOAD ALL
# ─────────────────────────────────────────────────────────────
model       = load_model()
X           = load_data()
explainer   = load_explainer(model, X)
kb_text     = load_kb()
day_options = build_day_options(MODEL_PATH, DATA_PATH)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="aqi-header">
    <h1>Delhi AQI Intelligence System</h1>
    <p>Understand &nbsp;•&nbsp; Simulate &nbsp;•&nbsp; Improve Air Quality</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DAY SELECTOR
# ─────────────────────────────────────────────────────────────
top_l, top_r = st.columns([3, 1])
with top_r:
    if not day_options:
        st.error("No rows with AQI > 50 found.")
        st.stop()
    day_labels   = list(day_options.keys())
    chosen_label = st.selectbox(
        "Select day",
        options=day_labels,
        index=len(day_labels) // 2,
        label_visibility="collapsed",
    )
    row_idx = day_options[chosen_label]

baseline = X.iloc[row_idx].to_dict()

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "reductions" not in st.session_state:
    st.session_state.reductions = {f: 0 for f in FEATURES}
if "reset_flag" not in st.session_state:
    st.session_state.reset_flag = False

if st.session_state.get("last_row_idx") != row_idx:
    st.session_state.last_row_idx = row_idx
    for f in FEATURES:
        st.session_state[f"slider_{f}"] = 0
    st.session_state.reductions = {f: 0 for f in FEATURES}

if st.session_state.reset_flag:
    for f in FEATURES:
        st.session_state[f"slider_{f}"] = 0
    st.session_state.reductions = {f: 0 for f in FEATURES}
    st.session_state.reset_flag = False

# ─────────────────────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────────────────────
modified          = apply_reductions(baseline, st.session_state.reductions)
orig_aqi          = predict_aqi(model, baseline)
new_aqi           = predict_aqi(model, modified)
delta, pct_change = compute_delta(orig_aqi, new_aqi)

shap_vals, base_val = get_shap(explainer, modified)
top3          = shap_vals[shap_vals > 0].sort_values(ascending=False).head(3)
top3_features = top3.index.tolist()

orig_cat, orig_text_col, orig_bg, orig_icon = aqi_category(orig_aqi)
new_cat,  new_text_col,  new_bg,  new_icon  = aqi_category(new_aqi)

# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

# ══════════════════════════════
# LEFT — SLIDERS
# ══════════════════════════════
with left_col:
    st.markdown('<div class="section-label">Pollutant Reduction Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="warning-banner">⚠️  Effective reduction capped at {MAX_REDUCTION_PCT}% '
        f'per pollutant (realistic policy bound).</div>',
        unsafe_allow_html=True,
    )

    btn_l, btn_r = st.columns(2)
    with btn_l:
        # FIX 2: st.rerun() forces recompute so AQI cards update immediately
        if st.button("⚡ Smart Recommend"):
            for f in FEATURES:
                val = 20 if f in top3_features else 0
                st.session_state[f"slider_{f}"] = val
                st.session_state.reductions[f]  = val
            st.rerun()
    with btn_r:
        if st.button("↺ Reset All"):
            st.session_state.reset_flag = True
            st.rerun()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:0.8rem">Particulates</div>', unsafe_allow_html=True)
    for f in ["PM2.5", "PM10"]:
        val = st.slider(
            f"{f}  — current: {baseline[f]:.1f} μg/m³",
            min_value=0, max_value=50, step=1,
            value=st.session_state.reductions.get(f, 0),
            key=f"slider_{f}", format="%d%%",
        )
        st.session_state.reductions[f] = val

    st.markdown('<div class="section-label" style="margin-top:0.8rem">Nitrogen Compounds</div>', unsafe_allow_html=True)
    for f in ["NO", "NO2", "NH3"]:
        val = st.slider(
            f"{f}  — current: {baseline[f]:.1f} μg/m³",
            min_value=0, max_value=50, step=1,
            value=st.session_state.reductions.get(f, 0),
            key=f"slider_{f}", format="%d%%",
        )
        st.session_state.reductions[f] = val

    st.markdown('<div class="section-label" style="margin-top:0.8rem">Other Gases</div>', unsafe_allow_html=True)
    for f in ["CO", "SO2", "O3"]:
        val = st.slider(
            f"{f}  — current: {baseline[f]:.1f} μg/m³",
            min_value=0, max_value=50, step=1,
            value=st.session_state.reductions.get(f, 0),
            key=f"slider_{f}", format="%d%%",
        )
        st.session_state.reductions[f] = val

# ══════════════════════════════
# RIGHT — AQI CARDS
# ══════════════════════════════
with right_col:
    st.markdown('<div class="section-label">Live AQI Forecast</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="aqi-card" style="background:linear-gradient(135deg,{orig_bg}22,{orig_bg}11);">
        <div class="aqi-label">Baseline AQI</div>
        <div class="aqi-number" style="color:{orig_text_col}">{orig_aqi:.0f}</div>
        <div class="aqi-category">{orig_icon} {orig_cat}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="aqi-card" style="background:linear-gradient(135deg,{new_bg}33,{new_bg}15);
         border:1px solid {new_text_col}44;">
        <div class="aqi-label">Simulated AQI</div>
        <div class="aqi-number" style="color:{new_text_col}">{new_aqi:.0f}</div>
        <div class="aqi-category">{new_icon} {new_cat}</div>
    </div>
    """, unsafe_allow_html=True)

    delta_color = "#52b788" if delta > 0 else ("#e63946" if delta < 0 else "#555")
    delta_arrow = "▼" if delta > 0 else ("▲" if delta < 0 else "—")

    st.markdown(f"""
    <div class="delta-card">
        <div>
            <div class="delta-label">AQI Improvement</div>
            <div class="delta-value" style="color:{delta_color}">
                {delta_arrow} {delta:.1f} pts
            </div>
        </div>
        <div style="text-align:right">
            <div class="delta-label">Reduction</div>
            <div style="font-size:1.3rem;font-weight:600;color:{delta_color}">
                {pct_change:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    rows_html = ""
    for f in top3_features:
        rows_html += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:0.35rem 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="font-size:0.85rem;color:rgba(255,255,255,0.7)">{f}</span>'
            f'<span style="font-size:0.85rem;font-weight:600;color:#e63946">+{shap_vals[f]:.1f}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:14px;padding:1rem 1.2rem;margin-top:0.5rem;">'
        f'<div class="section-label" style="margin-bottom:0.6rem">Top AQI Drivers Today</div>'
        f'{rows_html}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown("---")

viz_l, viz_r = st.columns(2, gap="large")

with viz_l:
    st.markdown('<div class="section-label">SHAP Feature Importance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.78rem;color:rgba(255,255,255,0.3);margin-bottom:0.8rem">'
        'Red = pushes AQI up &nbsp;|&nbsp; Green = pulls AQI down</div>',
        unsafe_allow_html=True,
    )
    fig = shap_chart(shap_vals)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with viz_r:
    st.markdown('<div class="section-label">Sensitivity — PM2.5 Reduction</div>', unsafe_allow_html=True)
    # FIX 1: dedicated function, no inline matplotlib that can silently fail
    fig2 = sensitivity_chart(baseline, model)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ─────────────────────────────────────────────────────────────
# AI INSIGHTS
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">AI Insights</div>', unsafe_allow_html=True)
st.markdown(
    f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.25);margin-bottom:0.5rem">'
    f'Model: <code style="color:rgba(168,237,234,0.5)">{OLLAMA_MODEL}</code> &nbsp;·&nbsp; '
    f'Requires <code style="color:rgba(168,237,234,0.5)">ollama serve</code> running in a terminal'
    f'</div>',
    unsafe_allow_html=True,
)

ai_l, _ = st.columns([0.2, 0.8])
with ai_l:
    if st.button("✦ Generate Insights"):
        kb_ctx = retrieve_kb(kb_text, top3_features)
        with st.spinner("Thinking..."):
            response = call_ollama(
                top3_features, shap_vals.to_dict(),
                orig_aqi, new_aqi, delta, kb_ctx,
            )
        st.session_state.llm_response = response

if st.session_state.get("llm_response"):
    resp = st.session_state.llm_response
    st.markdown(
        f'<div class="insight-card">'
        f'<div class="insight-header">✦ &nbsp; AI Analysis &nbsp;·&nbsp; {OLLAMA_MODEL}</div>'
        f'{resp.replace(chr(10), "<br>")}'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.15);font-size:0.72rem;
     letter-spacing:1.5px;text-transform:uppercase;padding:1rem 0;">
    Delhi AQI Intelligence &nbsp;·&nbsp; XGBoost + SHAP + Ollama &nbsp;·&nbsp; Local AI, Zero Cloud
</div>
""", unsafe_allow_html=True)