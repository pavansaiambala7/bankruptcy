# app.py – Flask app with preprocessing
import os, json, pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

with open("models/xgb_model.pkl",  "rb") as f: xgb_model  = pickle.load(f)
with open("models/imputer.pkl",    "rb") as f: imputer    = pickle.load(f)
with open("models/scaler.pkl",     "rb") as f: scaler     = pickle.load(f)
with open("models/features.pkl",   "rb") as f: FEATURE_COLS = pickle.load(f)
with open("models/clip_bounds.pkl","rb") as f: p1, p99    = pickle.load(f)
ann_model = tf.keras.models.load_model("models/ann_model.keras")
with open("models/samples.json") as f: SAMPLES = json.load(f)

def preprocess(raw_values):
    """Same steps as train.py – prevents train/serve mismatch."""
    arr = np.array(raw_values, dtype=float).reshape(1, -1)
    arr = imputer.transform(arr)
    arr = np.clip(arr, p1[:len(FEATURE_COLS)], p99[:len(FEATURE_COLS)])
    margin = arr[:, 1] / (arr[:, 0] + 1e-9)
    lev    = arr[:, 3] / (arr[:, 4] + 1e-9)
    arr = np.hstack([arr, margin.reshape(-1,1), lev.reshape(-1,1)])
    return scaler.transform(arr)

@app.route("/")
def index():
    return render_template("index.html", features=FEATURE_COLS, samples=SAMPLES)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
