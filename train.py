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
    # Read @attribute lines to get column names
    attr_names = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                attr_names.append(line.split()[1])
            elif line.lower().startswith("@data"):
                break
    print(f"Attributes: {attr_names[:5]} ...")
    # TODO: load data section
