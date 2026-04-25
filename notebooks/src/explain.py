#%%
import pandas as pd
import shap
import joblib
import os

#%%
BASE_DIR = r"C:\Users\suman\OneDrive\Documents\Projects\DELHI AQI STUDY"

data_path = os.path.join(BASE_DIR, "data", "processed", "delhi_clean.csv")
df = pd.read_csv(data_path)

features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3']

X = df[features].dropna().copy()

model_path = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
model = joblib.load(model_path)

print(f"Data loaded: {X.shape} | Model loaded ✅")

#%%
# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

background  = shap.sample(X, 100, random_state=42)
explainer   = shap.Explainer(model, background)
shap_values = explainer(X)

# 1️⃣ Global Feature Importance
shap.plots.bar(shap_values)

# 2️⃣ Detailed Distribution
shap.plots.beeswarm(shap_values)

# 3️⃣ Force Plot (single prediction)
shap.plots.waterfall(shap_values[0])

# %%