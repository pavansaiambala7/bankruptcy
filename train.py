# train.py – ML pipeline for Bankruptcy Forecasting
# Run once: python train.py

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# =============================================================
# STEP 1 – Parse the ARFF file (no scipy needed)
# =============================================================
def load_arff(path):
    attr_names, data_lines, in_data = [], [], False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                attr_names.append(line.split()[1])
            elif low.startswith("@data"):
                in_data = True
            elif in_data:
                data_lines.append(line)
    df = pd.read_csv(StringIO("\n".join(data_lines)),
                     header=None, names=attr_names, na_values="?")
    # class column arrives as b'0' or b'1' – extract the digit
    df["class"] = df["class"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    return df

df = load_arff("data/3year.arff")
print(f"[1] Loaded {df.shape[0]} rows, {df.shape[1]} columns")
print(f"    Bankruptcy rate: {df['class'].mean()*100:.1f}%")

# =============================================================
# STEP 2 – Data cleaning
# =============================================================
# Replace inf with NaN (division-by-zero in financial ratios)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

FEATURE_COLS = [c for c in df.columns if c != "class"]
X_raw = df[FEATURE_COLS].values
y     = df["class"].values
print(f"[2a] Inf replaced, features: {len(FEATURE_COLS)}")

# Fill NaN with the column median (robust against outliers)
imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X_raw)
print(f"[2] NaN after imputation: {np.isnan(X_imp).sum()}")

# =============================================================
# STEP 3 – Feature engineering
# =============================================================
# 3-a: Clip to 1st-99th percentile so one bad row can't dominate
p1  = np.percentile(X_imp, 1,  axis=0)
p99 = np.percentile(X_imp, 99, axis=0)
X_clipped = np.clip(X_imp, p1, p99)
print("[3a] Outliers clipped")

# 3-b: Net-profit margin proxy = Attr2 / (Attr1 + eps)
margin_proxy = (X_clipped[:, 1] / (X_clipped[:, 0] + 1e-9)).reshape(-1, 1)
print("[3b] margin_proxy added")
