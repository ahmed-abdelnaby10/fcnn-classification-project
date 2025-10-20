"""
EDA & Preprocessing for the Pima Indians Diabetes dataset (binary classification).
- Loads a raw CSV (default: data/raw/diabetes.csv)
- Replaces zeros with NaN in medical columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
- Imputes numeric features (median), scales with StandardScaler
- Splits into train/val/test = 70% / 15% / 15% (stratified)
- Saves processed CSVs + fitted preprocessing pipeline (scaler.pkl)

Notes:
- Target column is 'Outcome' (0/1).
- All features are numeric; no categorical encoding is applied.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


# ---------- paths ----------
# Resolve project root whether the script is run from repo root or elsewhere
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "data" / "processed"


# ---------- helpers ----------
def load_and_clean_diabetes(csv_path: Path) -> pd.DataFrame:
    """Load diabetes CSV and convert known 'zero means missing' columns to NaN."""
    df = pd.read_csv(csv_path)

    # Columns where 0 is physiologically implausible and usually encodes missing
    zero_as_na_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in zero_as_na_cols:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)

    # Strip spaces for any object columns (defensive; diabetes is numeric though)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).strip()

    return df


def build_numeric_preprocessor(num_cols):
    """Median impute + StandardScaler for numeric features."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        ],
        remainder="drop",
    )


def to_dense_df(arr, prefix="f"):
    """Convert (possibly sparse) array to a dense pandas DataFrame with f0..fN columns."""
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return pd.DataFrame(arr, columns=[f"{prefix}{i}" for i in range(arr.shape[1])])


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to raw diabetes CSV (default: data/raw/diabetes.csv)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help="Output dir for processed splits and scaler.pkl (default: data/processed)",
    )
    args = ap.parse_args()

    csv_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    assert csv_path.exists(), f"CSV not found: {csv_path}"

    # Load and clean
    df = load_and_clean_diabetes(csv_path)
    assert "Outcome" in df.columns, "Target column 'Outcome' not found."

    y = df["Outcome"].astype(int)
    X = df.drop(columns=["Outcome"])

    # All columns are numeric in this dataset
    num_cols = list(X.columns)
    pre = build_numeric_preprocessor(num_cols)

    # Split 70/15/15 (stratified)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
        )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
        )

    # Fit/transform
    X_train_p = pre.fit_transform(X_train)
    X_val_p = pre.transform(X_val)
    X_test_p = pre.transform(X_test)

    # Save splits
    train_df = to_dense_df(X_train_p, "f"); train_df["target"] = y_train.values
    val_df   = to_dense_df(X_val_p,   "f"); val_df["target"]   = y_val.values
    test_df  = to_dense_df(X_test_p,  "f"); test_df["target"]  = y_test.values

    train_df.to_csv(outdir / "train.csv", index=False)
    val_df.to_csv(outdir / "val.csv", index=False)
    test_df.to_csv(outdir / "test.csv", index=False)

    # Save preprocessing pipeline
    joblib.dump(pre, outdir / "scaler.pkl")

    print("Saved:")
    print(" -", outdir / "train.csv")
    print(" -", outdir / "val.csv")
    print(" -", outdir / "test.csv")
    print(" -", outdir / "scaler.pkl")


if __name__ == "__main__":
    main()
