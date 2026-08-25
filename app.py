# app.py – Flask app with model loading
import pickle
import numpy as np
from flask import Flask, render_template
import tensorflow as tf

app = Flask(__name__)

# Load all models ONCE at startup (not per request)
with open("models/xgb_model.pkl",  "rb") as f: xgb_model  = pickle.load(f)
with open("models/imputer.pkl",    "rb") as f: imputer    = pickle.load(f)
with open("models/scaler.pkl",     "rb") as f: scaler     = pickle.load(f)
with open("models/features.pkl",   "rb") as f: FEATURE_COLS = pickle.load(f)
with open("models/clip_bounds.pkl","rb") as f: p1, p99    = pickle.load(f)
ann_model = tf.keras.models.load_model("models/ann_model.keras")
print(f"Models loaded. Features: {len(FEATURE_COLS)}")

@app.route("/")
def index():
    return f"Bankruptcy Forecaster – {len(FEATURE_COLS)} features ready"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
