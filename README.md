# Bankruptcy Forecasting Web App

Predicts company bankruptcy using financial ratio data from Polish companies.
Three models are trained — XGBoost, a Keras ANN, and a soft-voting Hybrid — and served through a Flask web interface.

---

## Dataset

[UCI Polish Companies Bankruptcy Data](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data) — `data/3year.arff`

- 10,503 companies, 64 numeric financial-ratio features (`Attr1` – `Attr64`)
- Target: `class` (0 = healthy, 1 = bankrupt)
- ~95% healthy / ~5% bankrupt (heavily imbalanced)

---

## Project Structure

```
bankruptcy-app/
├── data/
│   └── 3year.arff          # UCI dataset (download separately)
├── models/                 # Saved artifacts (created by train.py)
│   ├── xgb_model.pkl
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   ├── clip_bounds.pkl
│   ├── ann_model.keras
│   └── samples.json
├── templates/
│   └── index.html
├── train.py
├── app.py
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from the UCI link above and place `3year.arff` in `data/`.

---

## Usage

**Step 1 — Train the models** (run once, takes ~2–5 minutes):

```bash
python train.py
```

This runs the full pipeline — data cleaning, feature engineering, model training, evaluation — and saves all artifacts to `models/`.

**Step 2 — Start the web app:**

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

**API endpoint** (JSON in / JSON out):

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, 0.5, ...]}'
```

---

## Pipeline

| Step | Description |
|------|-------------|
| Load | Custom ARFF parser — reads `@attribute` headers, loads data with `pandas.read_csv` |
| Clean | Replace `inf` with `NaN`, fill missing values with column median (`SimpleImputer`) |
| Feature engineering | Clip to 1st–99th percentile; add `margin_proxy` and `leverage_ratio` ratio features |
| Split | 80/20 train/test, stratified by class label (`random_state=42`) |
| Scale | `StandardScaler` fitted on training data only |
| XGBoost | `GridSearchCV` over `max_depth`, `learning_rate`, `n_estimators`; `scale_pos_weight` for imbalance |
| Keras ANN | `Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(1, sigmoid)`; `EarlyStopping` on `val_loss` |
| Hybrid | Soft voting — average of XGBoost and ANN probabilities, threshold 0.5 |
| Evaluate | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix for all three models |
| Export | ANN → `ann_model.keras`; rest → `.pkl` via `pickle` |

---

## Results (3-year dataset)

| Model   | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------|----------|-----------|--------|--------|---------|
| XGBoost | 0.9653   | 0.6912    | 0.4747 | 0.5629 | 0.9142  |
| ANN     | 0.6854   | 0.1042    | 0.7475 | 0.1829 | 0.7642  |
| Hybrid  | 0.9567   | 0.5444    | 0.4949 | 0.5185 | 0.8795  |

> Accuracy is a poor metric here due to class imbalance (~95% healthy). ROC-AUC and Recall are more meaningful for a bankruptcy detection task.

---

## Tech Stack

- **Python 3.10+**
- [Flask](https://flask.palletsprojects.com/) — web serving
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosted trees
- [TensorFlow / Keras](https://www.tensorflow.org/) — neural network
- [scikit-learn](https://scikit-learn.org/) — preprocessing, evaluation
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
