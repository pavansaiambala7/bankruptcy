# =============================================================================
# app.py  –  Flask web app for real-time bankruptcy prediction
# Run:  python app.py
# =============================================================================

import os, json, pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

# =============================================================================
# Load all models ONCE at startup (not per request – saves time & memory)
# =============================================================================
with open("models/xgb_model.pkl",  "rb") as f: xgb_model  = pickle.load(f)
with open("models/imputer.pkl",    "rb") as f: imputer    = pickle.load(f)
with open("models/scaler.pkl",     "rb") as f: scaler     = pickle.load(f)
with open("models/features.pkl",   "rb") as f: FEATURE_COLS = pickle.load(f)
with open("models/clip_bounds.pkl","rb") as f: p1, p99    = pickle.load(f)

ann_model = tf.keras.models.load_model("models/ann_model.keras")

with open("models/samples.json") as f:
    SAMPLES = json.load(f)

# =============================================================================
# Shared preprocessing – EXACT same steps as train.py so there's no mismatch
# =============================================================================
def preprocess(raw_values: list) -> np.ndarray:
    """
    raw_values : list of 64 floats (Attr1 .. Attr64), possibly containing NaN.
    Returns    : scaled numpy array ready for model.predict()
    """
    # 1. Impute missing values using the median learned during training
    arr = np.array(raw_values, dtype=float).reshape(1, -1)
    arr = imputer.transform(arr)

    # 2. Clip to the 1st-99th percentile bounds learned during training
    arr = np.clip(arr, p1[:len(FEATURE_COLS)], p99[:len(FEATURE_COLS)])

    # 3. Recreate the two engineered features (must match train.py exactly)
    margin_proxy = arr[:, 1] / (arr[:, 0] + 1e-9)
    leverage     = arr[:, 3] / (arr[:, 4] + 1e-9)
    arr = np.hstack([arr, margin_proxy.reshape(-1,1), leverage.reshape(-1,1)])

    # 4. Scale with the fitted StandardScaler
    arr = scaler.transform(arr)
    return arr

def predict_all(arr: np.ndarray) -> dict:
    """Run all three models and return a dict of probabilities."""
    xgb_p    = float(xgb_model.predict_proba(arr)[0, 1])
    ann_p    = float(ann_model.predict(arr, verbose=0)[0, 0])
    hybrid_p = (xgb_p + ann_p) / 2.0
    return {"xgb": round(xgb_p,4), "ann": round(ann_p,4),
            "hybrid": round(hybrid_p,4)}

# =============================================================================
# Routes
# =============================================================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",
                           features=FEATURE_COLS,
                           samples=SAMPLES)

@app.route("/predict", methods=["POST"])
def predict():
    # Pull one float per feature from the form; default to 0 if blank
    raw = [float(request.form.get(f, 0) or 0) for f in FEATURE_COLS]
    arr  = preprocess(raw)
    probs = predict_all(arr)
    label = "🚨 BANKRUPT" if probs["hybrid"] >= 0.5 else "✅ NOT Bankrupt"
    return render_template("index.html",
                           features=FEATURE_COLS,
                           samples=SAMPLES,
                           result=label,
                           probs=probs,
                           values=raw)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint: POST {"features": [v1, v2, ..., v64]} → probabilities."""
    data = request.get_json(force=True)
    raw  = data.get("features", [0]*len(FEATURE_COLS))
    arr  = preprocess(raw)
    probs = predict_all(arr)
    probs["bankrupt"] = probs["hybrid"] >= 0.5
    return jsonify(probs)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
