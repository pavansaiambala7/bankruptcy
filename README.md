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

## Class Imbalance
Only ~5% bankrupt. scale_pos_weight = neg/pos (~20) corrects this.
