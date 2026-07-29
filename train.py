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

# 3-c: Leverage ratio = Attr4 / (Attr5 + eps) – high value → risk
leverage = (X_clipped[:, 3] / (X_clipped[:, 4] + 1e-9)).reshape(-1, 1)
X_eng = np.hstack([X_clipped, margin_proxy, leverage])
FINAL_FEATURE_NAMES = FEATURE_COLS + ["margin_proxy", "leverage_ratio"]
print(f"[3] Total features: {X_eng.shape[1]}")

# =============================================================
# STEP 4 – Stratified train/test split (80/20)
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y, test_size=0.2, stratify=y, random_state=42)
print(f"[4] Train={X_train.shape[0]}  Test={X_test.shape[0]}")

# =============================================================
# STEP 5 – Scale features (fit on train ONLY – no data leakage)
# =============================================================
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("[5] StandardScaler applied")

# =============================================================
# STEP 6 – Model A: XGBoost with GridSearchCV
# =============================================================
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos
print(f"[6] scale_pos_weight = {spw:.1f}")

xgb_base = XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                          random_state=42, n_jobs=-1)
param_grid = {
    "max_depth":     [3, 5],
    "learning_rate": [0.05, 0.1],
    "n_estimators":  [100, 200],
}
# cv=3 and roc_auc scoring suit imbalanced classification
grid = GridSearchCV(xgb_base, param_grid, cv=3,
                    scoring="roc_auc", n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)
xgb_model = grid.best_estimator_
print(f"[6] Best XGB params: {grid.best_params_}")

# =============================================================
# STEP 7 – Model B: Keras ANN (complete)
# =============================================================
class_weight = {0: 1.0, 1: spw}   # penalise bankrupt misses more

ann = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1,  activation="sigmoid"),
])
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=5,
                           restore_best_weights=True)
ann.fit(X_train, y_train,
        validation_split=0.1, epochs=50, batch_size=256,
        class_weight=class_weight, callbacks=[early_stop], verbose=0)
print("[7] ANN training complete")
