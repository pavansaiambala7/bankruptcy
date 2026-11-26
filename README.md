# Bankruptcy Forecasting Web App

A beginner-friendly machine-learning project that predicts company bankruptcy using three models — XGBoost, a Keras ANN, and a soft-voting Hybrid — served through a simple Flask web interface.

---

## How to Run

```bash
# 1. Install dependencies (one-time)
pip install -r requirements.txt

# 2. Put the dataset file here:
#    bankruptcy-app/data/3year.arff
#    Download from: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

# 3. Train all models (one-time, ~2-5 minutes)
python train.py

# 4. Start the web app
python app.py
# Open: http://127.0.0.1:5000
```

---

## Learn This Project in 30 Minutes

Work through these steps in order. Each one is a short, commented section in `train.py`.

### Step 1 — Load the ARFF file
The UCI dataset comes in ARFF format (a text file with metadata headers and a data block). We parse it with a small custom function: read column names from `@attribute` lines, skip to `@data`, then load the rest with `pandas.read_csv`. No external ARFF library needed.

**WHY?** It keeps the dependency list minimal and shows you exactly what the file contains.

---

### Step 2 — Data Cleaning
Two things:
- Replace `inf` / `-inf` with `NaN` (some financial ratios divide by zero).
- Fill every `NaN` with the column's **median** using scikit-learn's `SimpleImputer`.

**WHY median, not mean?** Financial ratios can have extreme outliers. The median is not pulled by them, so it gives a safer estimate of a "typical" value.

---

### Step 3 — Feature Engineering
Three simple operations:
1. **Clip outliers** to the 1st–99th percentile. A single data-entry error (e.g., ratio = 10,000) can dominate a model's weights. Clipping limits the damage.
2. **Margin proxy** = Attr2 / (Attr1 + ε). Combining two correlated ratios into one can capture information neither captures alone.
3. **Leverage ratio** = Attr4 / (Attr5 + ε). High leverage is a classic bankruptcy warning sign.

**WHY only 3?** The goal is interpretability. You can explain each feature in one sentence.

---

### Step 4 — Train / Test Split
We hold out 20% of rows for final evaluation. `stratify=y` ensures both splits have the same share of bankrupt companies (~6.8%). Without stratification you might get a test set with zero bankrupt cases.

---

### Step 5 — Scaling
`StandardScaler` transforms every feature to mean=0, std=1. Neural networks need this because their gradient steps are sensitive to feature magnitudes. XGBoost doesn't strictly need it, but it makes the shared scaler simpler to manage.

**Key rule:** fit the scaler **only on training data**, then apply it to test data. Fitting on test data would leak future information ("data leakage").

---

### Step 6 — XGBoost with GridSearchCV
XGBoost is a fast, tree-based ensemble model. We set `scale_pos_weight = negative_count / positive_count` so the model treats each bankrupt sample as ~14× more important, correcting for the class imbalance. `GridSearchCV` tries every combination of `max_depth`, `learning_rate`, and `n_estimators` with 3-fold cross-validation and picks the best by ROC-AUC.

---

### Step 7 — Keras ANN
A simple 3-layer neural network:
```
Input (66 features) → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(1, sigmoid)
```
- **relu** introduces non-linearity so the network can learn complex patterns.
- **Dropout(0.3)** randomly zeroes 30% of neurons each step, preventing over-reliance on any single neuron (regularisation).
- **sigmoid** outputs a probability between 0 and 1.
- **EarlyStopping** halts training when validation loss stops improving, preventing overfitting.

---

### Step 8 — Hybrid (Soft Voting)
Average the XGBoost probability and ANN probability. If one model is wrong in a particular region of feature space, the other model often compensates. This is the simplest ensemble technique and frequently outperforms either model alone.

---

### Step 9 — Evaluation
We compute five metrics for all three models on the held-out test set.

---

### Step 10 — Export
- **ANN → `.keras`**: TensorFlow's native format preserves the full graph, weights, and optimizer state.
- **XGBoost, imputer, scaler → `.pkl`**: Python's `pickle` serialises any Python object to bytes. It's the standard for scikit-learn and XGBoost artifacts.
- **samples.json**: 5 real test-set rows pre-loaded for the browser demo button.

---

## Why Accuracy Alone Is Misleading

This dataset is **imbalanced**: ~93% of companies are healthy, only ~7% go bankrupt. A model that always predicts "not bankrupt" would have **93% accuracy** but **0% recall** — it catches zero actual bankruptcies. That is useless for a risk tool.

| Metric | What it tells you |
|---|---|
| **Accuracy** | % of all predictions correct. Misleading when classes are unequal. |
| **Precision** | Of companies we flagged bankrupt, how many really were? (False-alarm rate) |
| **Recall** | Of all companies that did go bankrupt, how many did we catch? (Miss rate) |
| **F1** | Harmonic mean of precision and recall. Balanced single-number summary. |
| **ROC-AUC** | How well the model *ranks* bankrupt above healthy regardless of threshold. 0.5 = random, 1.0 = perfect. |

For a bankruptcy detector, **recall** is usually the most important metric because missing a real bankruptcy (false negative) is far more costly than a false alarm.

---

## Why the Hybrid Helps

Individually, XGBoost excels at structured tabular data with complex interactions; the ANN captures smooth non-linear patterns differently. When their errors are not perfectly correlated, averaging their probabilities reduces variance — the ensemble is more stable than either alone. This is called **soft voting**.

---

## Why `.keras` vs `.pkl`

| | `.keras` | `.pkl` |
|---|---|---|
| **Used for** | TensorFlow/Keras models | Any Python object |
| **Why native format** | Preserves graph + weights + optimizer; version-safe | Faster and simpler for scikit-learn/XGBoost |
| **Can you load in another language?** | Yes (TF serving, TF.js) | Python only |

---

## 10 Likely Interview Questions

**Q1. Why do you impute with the median and not the mean?**
Financial ratios have heavy outliers; the median is resistant to them and gives a more representative "typical" value.

**Q2. What is data leakage, and how did you avoid it?**
Leakage is when test-set information influences training. We avoided it by fitting the imputer, scaler, and clip bounds *only* on training data, then applying them to test data.

**Q3. What is `scale_pos_weight` in XGBoost?**
It tells XGBoost to weight the minority class (bankrupt=1) more heavily during training, compensating for class imbalance. Set to negative_count / positive_count.

**Q4. Why use Dropout in the ANN?**
Dropout randomly deactivates neurons during training, preventing co-adaptation. This is a regularisation technique that reduces overfitting.

**Q5. Why is ROC-AUC a better metric than accuracy here?**
ROC-AUC measures ranking quality and is unaffected by class imbalance. A naïve all-negative classifier has 93% accuracy but 0.5 ROC-AUC (random).

**Q6. What is soft voting and when would you use hard voting instead?**
Soft voting averages probabilities (requires models to output calibrated probabilities). Hard voting takes the majority class vote. Soft voting generally performs better when models output well-calibrated probabilities.

**Q7. Why do you fit the scaler only on training data?**
Fitting on test data would reveal test-set statistics to the model, an information leak. The scaler must only know what training data looks like.

**Q8. How does EarlyStopping work?**
It monitors validation loss each epoch. If it doesn't improve for `patience` consecutive epochs, training stops and the best weights are restored.

**Q9. How would you deploy this to production?**
Swap Flask's dev server for Gunicorn + Nginx, containerise with Docker, store models in a blob store (S3/GCS), and version-pin dependencies. Add model monitoring for data drift.

**Q10. Why save 5 sample rows in `samples.json`?**
So the demo button in the UI can populate the form with real test-set values without needing a live database connection. It makes the demo self-contained and reproducible.
