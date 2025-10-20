
import joblib
import numpy as np
from pathlib import Path

SCALER_PATH = Path("data/processed/scaler.pkl")

def load_preprocessor():
    return joblib.load(SCALER_PATH)

def transform_features(pre, arr2d):
    # arr2d: List[List[float]] in model-space (already feature-engineered)
    # If your serving takes raw feature dict, replicate same steps used in training.
    return pre.transform(arr2d)
