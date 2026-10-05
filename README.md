# Bankruptcy Forecasting Web App

Predicts company bankruptcy using financial ratio data from Polish companies.

## Goal
Build a Flask web app using XGBoost + Keras to predict if a company
will go bankrupt within 3 years, based on 64 financial ratios.

## Dataset
UCI Polish Companies Bankruptcy Data — `data/3year.arff`
- 10,503 rows, 64 numeric features (Attr1..Attr64)
- Target: `class` (0 = healthy, 1 = bankrupt)
- Heavily imbalanced: ~95% healthy, ~5% bankrupt

## Project Structure
```
bankruptcy-app/
├── data/3year.arff
├── train.py
├── app.py
├── models/
├── templates/index.html
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python train.py
python app.py
```

## Pipeline Overview
1. Load — Parse ARFF file without scipy
2. Clean — Replace inf, impute NaN with median
3. Engineer — Clip outliers, add ratio features
4. Split — 80/20 train/test, stratified
5. Scale — StandardScaler (fit on train only)
6. XGBoost — GridSearchCV, scale_pos_weight for imbalance
7. ANN — Dense(64)→Dropout→Dense(32)→Dense(1), EarlyStopping
8. Hybrid — Soft voting (average XGB + ANN probabilities)
9. Evaluate — Accuracy, Precision, Recall, F1, ROC-AUC
10. Export — .keras for ANN, .pkl for everything else

## Why Accuracy Is Misleading
With 95% healthy companies, a dummy classifier scores 95% accuracy catching ZERO bankruptcies.
Prefer Recall and ROC-AUC for imbalanced datasets.
