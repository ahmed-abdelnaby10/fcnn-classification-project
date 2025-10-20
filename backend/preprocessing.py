import joblib
import numpy as np
from pathlib import Path
import pandas as pd

SCALER_PATH = Path("data/processed/scaler.pkl")

def load_preprocessor():
    return joblib.load(SCALER_PATH)

def get_expected_feature_names(pre):
    # numeric transformer is first; its column list is at transformers_[0][2]
    return list(pre.transformers_[0][2])

def transform_features(pre, arr2d):
    return pre.transform(arr2d)

def prepare_from_raw(pre, raw_dict):
    """
    Build a single-row DataFrame in the training column order from a raw dict, then transform.
    """
    cols = get_expected_feature_names(pre)
    missing = [c for c in cols if c not in raw_dict]
    if missing:
        raise ValueError(f"Missing keys in 'raw': {missing}")
    df = pd.DataFrame([[raw_dict[c] for c in cols]], columns=cols)
    return transform_features(pre, df)
