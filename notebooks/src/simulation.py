#%%
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR   = r"C:\Users\suman\OneDrive\Documents\Projects\DELHI AQI STUDY"
data_path  = os.path.join(BASE_DIR, "data", "processed", "delhi_clean.csv")
model_path = os.path.join(BASE_DIR, "models", "xgb_model.pkl")

# ============================================================
# 2. LOAD DATA + MODEL
# ============================================================

df       = pd.read_csv(data_path)
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3']

X     = df[features].dropna().copy()
model = joblib.load(model_path)

print(f"Data loaded: {X.shape}")
print("Model loaded ✅")

#%%
# ============================================================
# 3. SIMULATION FUNCTION
# ============================================================

def simulate_change(row: pd.Series, changes: dict) -> pd.DataFrame:
    """
    Applies percentage reductions to specified features.
    Returns a single-row DataFrame (required by XGBoost predict).

    changes = {"PM2.5": 20, "NO2": 10}  → reduce each by that %
    """
    modified = row.copy()
    for feature, percent in changes.items():
        if feature not in modified.index:
            raise ValueError(f"'{feature}' not in features list")
        modified[feature] *= (1 - percent / 100)

    # FIX: wrap in DataFrame so XGBoost gets correct feature names
    return pd.DataFrame([modified], columns=features)

#%%
# ============================================================
# 4. SINGLE SCENARIO TEST
# ============================================================

sample   = X.iloc[100]
scenario = {"PM2.5": 20, "NO2": 10}

original_input = pd.DataFrame([sample], columns=features)
modified_input = simulate_change(sample, scenario)

original_pred = model.predict(original_input)[0]
new_pred      = model.predict(modified_input)[0]

print("\n--- Single Scenario Result ---")
print(f"Original AQI   : {original_pred:.2f}")
print(f"New AQI        : {new_pred:.2f}")
print(f"Improvement    : {original_pred - new_pred:.2f}")

#%%
# ============================================================
# 5. MULTI-SCENARIO ENGINE
# ============================================================

def run_scenarios(sample: pd.Series, scenarios: list) -> pd.DataFrame:
    """
    Runs multiple policy scenarios against a single sample row.
    Returns a DataFrame sorted by AQI improvement (best first).
    """
    base_input = pd.DataFrame([sample], columns=features)
    base_pred  = model.predict(base_input)[0]

    results = []
    for i, scenario in enumerate(scenarios):
        modified = simulate_change(sample, scenario)
        new_pred = model.predict(modified)[0]
        results.append({
            "scenario_id" : i + 1,
            "changes"     : str(scenario),
            "base_aqi"    : round(base_pred,  2),
            "new_aqi"     : round(new_pred,   2),
            "improvement" : round(base_pred - new_pred, 2)
        })

    return pd.DataFrame(results).sort_values("improvement", ascending=False).reset_index(drop=True)

#%%
# ============================================================
# 6. TEST MULTIPLE POLICIES
# ============================================================

scenarios = [
    {"PM2.5": 10},
    {"PM2.5": 20},
    {"PM2.5": 30},
    {"NO2"  : 20},
    {"PM2.5": 20, "NO2": 10},
    {"PM10" : 15, "CO" : 10},
]

results_df = run_scenarios(sample, scenarios)

print("\n--- Policy Comparison ---")
print(results_df.to_string(index=False))

#%%
# ============================================================
# 7. SENSITIVITY ANALYSIS
# ============================================================

def sensitivity_analysis(
    sample: pd.Series,
    feature: str,
    steps: list = [0, 10, 20, 30, 40, 50]
) -> pd.DataFrame:
    """
    Shows how AQI changes as a single feature is reduced
    by increasing percentages.
    """
    base_input = pd.DataFrame([sample], columns=features)
    base_pred  = model.predict(base_input)[0]

    results = []
    for p in steps:
        modified = simulate_change(sample, {feature: p})
        new_pred = model.predict(modified)[0]
        results.append({
            "reduction_%"   : p,
            "new_aqi"       : round(new_pred,          2),
            "improvement"   : round(base_pred - new_pred, 2)
        })

    return pd.DataFrame(results)

#%%
# Run sensitivity for the two dominant features from SHAP
pm25_sens = sensitivity_analysis(sample, "PM2.5")
pm10_sens = sensitivity_analysis(sample, "PM10")

print("\n--- PM2.5 Sensitivity ---")
print(pm25_sens.to_string(index=False))

print("\n--- PM10 Sensitivity ---")
print(pm10_sens.to_string(index=False))

# %%
#%%
# ============================================================
# 8. SMART POLICY (SHAP-GUIDED AUTO SELECTION)
# ============================================================

import shap

background = shap.sample(X, 100, random_state=42)
explainer  = shap.Explainer(model, background)

def smart_policy(sample: pd.Series, top_k: int = 3, reduction: float = 20) -> dict:
    """
    Uses per-sample SHAP values to identify which pollutants
    are driving AQI the most on that specific day, then
    simulates reducing only those by `reduction` percent.

    This is smarter than fixed scenarios — the policy adapts
    to the actual pollution profile of the day.
    """
    sample_df = pd.DataFrame([sample], columns=features)
    shap_vals = explainer(sample_df)

    contrib = pd.Series(
        shap_vals.values[0],
        index=features
    ).sort_values(ascending=False)

    # Only take features with positive SHAP (actively pushing AQI up)
    top_features = contrib[contrib > 0].head(top_k).index.tolist()

    if not top_features:
        return {"message": "No positive SHAP contributors found for this sample."}

    scenario  = {f: reduction for f in top_features}
    modified  = simulate_change(sample, scenario)

    base_pred = model.predict(sample_df)[0]
    new_pred  = model.predict(modified)[0]

    return {
        "top_features"  : top_features,
        "scenario"      : scenario,
        "original_aqi"  : round(base_pred,             2),
        "new_aqi"       : round(new_pred,               2),
        "improvement"   : round(base_pred - new_pred,   2),
        "shap_contribs" : contrib[top_features].round(2).to_dict()
    }

#%%
# --- Single sample test ---
result = smart_policy(sample, top_k=3, reduction=20)
print("\n--- SMART POLICY (single sample) ---")
for k, v in result.items():
    print(f"  {k}: {v}")

#%%
# --- Run smart policy across multiple days ---
def run_smart_policy_bulk(X: pd.DataFrame, top_k: int = 3, reduction: float = 20, n: int = 20) -> pd.DataFrame:
    """
    Applies smart_policy to the first n rows and compares results.
    Useful for seeing which days benefit most from targeted intervention.
    """
    rows = []
    for i in range(n):
        r = smart_policy(X.iloc[i], top_k=top_k, reduction=reduction)
        if "message" in r:
            continue
        rows.append({
            "day_index"     : i,
            "original_aqi"  : r["original_aqi"],
            "new_aqi"       : r["new_aqi"],
            "improvement"   : r["improvement"],
            "top_features"  : ", ".join(r["top_features"])
        })
    return pd.DataFrame(rows).sort_values("improvement", ascending=False).reset_index(drop=True)

bulk_results = run_smart_policy_bulk(X, top_k=3, reduction=20, n=50)
print("\n--- SMART POLICY (bulk, top 10 days) ---")
print(bulk_results.head(10).to_string(index=False))

# %%