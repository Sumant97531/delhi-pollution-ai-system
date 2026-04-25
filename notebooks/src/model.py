#%%
import numpy as np
import pandas as pd
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import shap

#%%
# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

BASE_DIR      = r"C:\Users\suman\OneDrive\Documents\Projects\DELHI AQI STUDY"
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(RAW_DIR,       exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Raw dir exists      :", os.path.exists(RAW_DIR))
print("Processed dir exists:", os.path.exists(PROCESSED_DIR))

#%%
# ============================================================
# 2. LOAD DATA — read from raw/
# ============================================================

file_path = os.path.join(RAW_DIR, "city_day.csv")

df = pd.read_csv(file_path)
print("Data loaded:", df.shape)

#%%
# ============================================================
# 3. FILTER DELHI
# ============================================================

df['City']        = df['City'].str.strip().str.lower()
delhi_data        = df[df['City'] == 'delhi'].copy()

delhi_data['Date']        = pd.to_datetime(delhi_data['Date'])
delhi_data                = delhi_data.sort_values('Date').reset_index(drop=True)
delhi_data['month']       = delhi_data['Date'].dt.month
delhi_data['day_of_year'] = delhi_data['Date'].dt.dayofyear

print("Delhi rows:", len(delhi_data))

#%%
# ============================================================
# 4. MISSING VALUES
# ============================================================

numeric_cols = delhi_data.select_dtypes(include=['number']).columns

delhi_data[numeric_cols] = delhi_data[numeric_cols].interpolate(
    limit_direction="both"
)

def get_aqi_bucket(aqi):
    if pd.isna(aqi):
        return None
    elif aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

mask = delhi_data['AQI_Bucket'].isna()
delhi_data.loc[mask, 'AQI_Bucket'] = delhi_data.loc[mask, 'AQI'].apply(get_aqi_bucket)

print("Remaining missing AQI_Bucket:", delhi_data['AQI_Bucket'].isna().sum())
print("Remaining missing (numeric)  :", delhi_data[numeric_cols].isna().sum().sum())

#%%
# ============================================================
# 5. SAVE PROCESSED DATA
# ============================================================

processed_path = os.path.join(PROCESSED_DIR, "delhi_clean.csv")
delhi_data.to_csv(processed_path, index=False)

print("\n✅ PROCESSED DATA SAVED")
print("Path       :", processed_path)
print("File exists:", os.path.exists(processed_path))
print("Shape      :", delhi_data.shape)

#%%
# ============================================================
# 6. MODEL TRAINING
# ============================================================

features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3']

model_data = delhi_data.dropna(subset=features + ['AQI']).copy()

X = model_data[features]
y = model_data['AQI']

split_idx          = int(len(model_data) * 0.8)
X_train, X_test    = X.iloc[:split_idx],  X.iloc[split_idx:]
y_train, y_test    = y.iloc[:split_idx],  y.iloc[split_idx:]

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

xgb_model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
r2     = xgb_model.score(X_test, y_test)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = root_mean_squared_error(y_test, y_pred)

print(f"\n🎯 XGB R²  : {r2:.4f}")
print(f"📉 MAE     : {mae:.2f}")
print(f"📉 RMSE    : {rmse:.2f}")

#%%
# ============================================================
# 7. SHAP
# ============================================================

background  = shap.sample(X_train, 100, random_state=42)
explainer   = shap.Explainer(xgb_model, background)
shap_values = explainer(X_test)

print("SHAP ready ✅")

#%%
# ============================================================
# 8. SAVE MODEL
# ============================================================

model_path = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)

joblib.dump(xgb_model, model_path)
print("Model saved at:", model_path)

# %%