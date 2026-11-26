# =============================================================================
# train.py  –  One-shot ML pipeline for Bankruptcy Forecasting
# Run once:  python train.py
# =============================================================================

import os, json, pickle, warnings
import numpy as np
import pandas as pd
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

# =============================================================================
# STEP 1 – Load the ARFF file WITHOUT scipy
# =============================================================================
def load_arff(path):
    """
    Read a .arff file manually.
    @attribute lines tell us column names.
    @data marks where the actual numbers begin.
    We hand the data section straight to pandas.
    """
    attr_names = []
    data_lines = []
    in_data = False

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):   # skip blank / comments
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                # e.g.  @attribute Attr1 NUMERIC
                parts = line.split()
                attr_names.append(parts[1])        # second token = name
            elif low.startswith("@data"):
                in_data = True
            elif in_data:
                data_lines.append(line)

    from io import StringIO
    csv_text = "\n".join(data_lines)
    df = pd.read_csv(StringIO(csv_text), header=None,
                     names=attr_names, na_values="?")

    # The 'class' column arrives as b'0'/b'1' or '0'/'1' – normalise to int
    df["class"] = df["class"].astype(str).str.strip("b'\" ").astype(int)
    return df

df = load_arff("data/3year.arff")
print(f"[1] Loaded  {df.shape[0]} rows, {df.shape[1]} columns")
print(f"    Bankruptcy rate: {df['class'].mean()*100:.1f}%")

# =============================================================================
# STEP 2 – Data cleaning
# =============================================================================
# Replace infinite values with NaN so the imputer can handle them uniformly
df.replace([np.inf, -np.inf], np.nan, inplace=True)

FEATURE_COLS = [c for c in df.columns if c != "class"]
X_raw = df[FEATURE_COLS].values
y     = df["class"].values

# SimpleImputer fills each NaN with that column's median
# (median is more robust than mean when there are extreme outliers)
imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(X_raw)

print(f"[2] Missing values after imputation: {np.isnan(X_imp).sum()}")

# =============================================================================
# STEP 3 – Feature engineering  (3 simple, explainable steps)
# =============================================================================
# 3-a: Clip extreme outliers to the 1st–99th percentile so a single bad row
#      cannot dominate the model's learning.
p1  = np.percentile(X_imp, 1,  axis=0)
p99 = np.percentile(X_imp, 99, axis=0)
X_clipped = np.clip(X_imp, p1, p99)

# 3-b: Net-profit margin proxy  =  Attr2 / (Attr1 + 1e-9)
#      (Attr1 ≈ net profit / total assets; Attr2 ≈ total liabilities / total assets)
#      Dividing gives a quick solvency signal.
margin_proxy = X_clipped[:, 1] / (X_clipped[:, 0] + 1e-9)
margin_proxy = margin_proxy.reshape(-1, 1)

# 3-c: Leverage ratio  =  Attr4 / (Attr5 + 1e-9)
#      (Attr4 ≈ short-term liabilities / total assets; Attr5 ≈ equity)
#      High leverage → higher bankruptcy risk.
leverage = X_clipped[:, 3] / (X_clipped[:, 4] + 1e-9)
leverage = leverage.reshape(-1, 1)

# Stack original + 2 new features
X_eng = np.hstack([X_clipped, margin_proxy, leverage])
FINAL_FEATURE_NAMES = FEATURE_COLS + ["margin_proxy", "leverage_ratio"]
print(f"[3] Features after engineering: {X_eng.shape[1]}")

# =============================================================================
# STEP 4 – Train / test split  (80/20, stratified so class ratios are preserved)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y, test_size=0.2, stratify=y, random_state=42
)
print(f"[4] Train={X_train.shape[0]}  Test={X_test.shape[0]}")

# =============================================================================
# STEP 5 – Scaling  (fit ONLY on train to avoid data leakage)
# =============================================================================
# StandardScaler: (x - mean) / std  → all features on the same scale.
# Neural networks and distance-based models need this; XGBoost doesn't strictly
# need it but it makes hyperparameter ranges more predictable.
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# =============================================================================
# STEP 6 – Model A: XGBoost with GridSearchCV
# =============================================================================
# scale_pos_weight = negative_samples / positive_samples
# tells XGBoost to treat each bankrupt company as ~N× more important,
# counteracting the class imbalance.
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos
print(f"[6] Class imbalance ratio (scale_pos_weight): {spw:.1f}")

xgb_base = XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                          random_state=42, n_jobs=-1)

param_grid = {
    "max_depth":      [3, 5],
    "learning_rate":  [0.05, 0.1],
    "n_estimators":   [100, 200],
}
# cv=3 keeps training fast; scoring on ROC-AUC suits imbalanced data
grid = GridSearchCV(xgb_base, param_grid, cv=3,
                    scoring="roc_auc", n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)
xgb_model = grid.best_estimator_
print(f"[6] Best XGB params: {grid.best_params_}")

# =============================================================================
# STEP 7 – Model B: Keras ANN
# =============================================================================
# class_weight tells Keras to penalise misclassifying the minority class more.
class_weight = {0: 1.0, 1: spw}

ann = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),          # randomly zero 30% of neurons → prevents overfitting
    Dense(32, activation="relu"),
    Dense(1,  activation="sigmoid"),  # output = probability of bankruptcy
])
ann.compile(optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=5,
                           restore_best_weights=True)
ann.fit(X_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=256,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0)
print("[7] ANN training complete")

# =============================================================================
# STEP 8 – Hybrid model: soft voting (average of two probabilities)
# =============================================================================
# Neither model is perfect alone. Averaging their probabilities smooths out
# individual errors – same idea as asking two experts and averaging their
# confidence scores.
xgb_prob  = xgb_model.predict_proba(X_test)[:, 1]
ann_prob  = ann.predict(X_test, verbose=0).flatten()
hybrid_prob = (xgb_prob + ann_prob) / 2.0

# Classify as bankrupt if probability ≥ 0.5
xgb_pred    = (xgb_prob    >= 0.5).astype(int)
ann_pred    = (ann_prob    >= 0.5).astype(int)
hybrid_pred = (hybrid_prob >= 0.5).astype(int)

# =============================================================================
# STEP 9 – Evaluation
# =============================================================================
def metrics(name, y_true, y_pred, y_prob):
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob), 4),
    }

results = [
    metrics("XGBoost", y_test, xgb_pred, xgb_prob),
    metrics("ANN",     y_test, ann_pred,  ann_prob),
    metrics("Hybrid",  y_test, hybrid_pred, hybrid_prob),
]

print("\n[9] ── Evaluation Results ──────────────────────────────")
print(pd.DataFrame(results).to_string(index=False))

for name, pred in [("XGBoost", xgb_pred), ("ANN", ann_pred), ("Hybrid", hybrid_pred)]:
    cm = confusion_matrix(y_test, pred)
    print(f"\n    {name} confusion matrix:\n{cm}")

# =============================================================================
# STEP 10 – Export artifacts
# =============================================================================
ann.save("models/ann_model.keras")

with open("models/xgb_model.pkl",  "wb") as f: pickle.dump(xgb_model, f)
with open("models/imputer.pkl",    "wb") as f: pickle.dump(imputer,   f)
with open("models/scaler.pkl",     "wb") as f: pickle.dump(scaler,    f)
with open("models/features.pkl",   "wb") as f: pickle.dump(FEATURE_COLS, f)
with open("models/clip_bounds.pkl","wb") as f: pickle.dump((p1, p99), f)

# Save 5 real test samples for the demo button (before scaling, after imputation)
# We need to find the original indices that ended up in X_test
# X_test at this point is already scaled; let's store raw imputed+clipped values
_, X_test_raw, _, y_test_raw = train_test_split(
    X_eng, y, test_size=0.2, stratify=y, random_state=42)

samples = []
for i in range(5):
    row = X_test_raw[i, :len(FEATURE_COLS)].tolist()   # only original 64 features
    samples.append({"features": row, "true_label": int(y_test_raw[i])})

with open("models/samples.json", "w") as f:
    json.dump(samples, f, indent=2)

print("\n[10] Models saved to models/")
print("     Done! Now run:  python app.py")
