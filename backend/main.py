from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from pathlib import Path
from .models import PredictRequest, PredictBatchRequest
from .preprocessing import load_preprocessor, transform_features, get_expected_feature_names, prepare_from_raw

app = FastAPI(title="FCNN Classifier API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("models/best_model.h5")
PRE = None
MODEL = None
EXPECTED_COLS = []

@app.on_event("startup")
def startup_event():
    global PRE, MODEL, EXPECTED_COLS
    try:
        PRE = load_preprocessor()
        EXPECTED_COLS = get_expected_feature_names(PRE)
        MODEL = load_model(MODEL_PATH)
        print("Loaded model and preprocessor. Expected columns:", EXPECTED_COLS)
    except Exception as e:
        print("Startup warning:", e)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": bool(MODEL is not None),
        "expected_n_features": len(EXPECTED_COLS) if EXPECTED_COLS else None,
        "expected_feature_names": EXPECTED_COLS or None,
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None or PRE is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded")

    try:
        if req.raw is not None:
            x_scaled = prepare_from_raw(PRE, req.raw)  # 1 x n
        else:
            feats = req.features or []
            n_exp = len(EXPECTED_COLS) if EXPECTED_COLS else None
            if n_exp is not None and len(feats) != n_exp:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {n_exp} features {EXPECTED_COLS}, got {len(feats)}."
                )
            x = np.array(feats, dtype=float).reshape(1, -1)
            # If your preprocessor was fit on a DataFrame (with named columns),
            # passing a numpy array is fine because the ColumnTransformer numeric step
            # uses column positions. If you want strict safety, build a DataFrame with EXPECTED_COLS.
            import pandas as pd
            x_df = pd.DataFrame(x, columns=EXPECTED_COLS) if EXPECTED_COLS else x
            x_scaled = transform_features(PRE, x_df)

        prob = float(MODEL.predict(x_scaled).ravel()[0])
        pred = int(prob >= 0.5)
        return {"prediction": pred, "probability": prob, "class_name": "Positive" if pred==1 else "Negative"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    if MODEL is None or PRE is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded")
    try:
        n_exp = len(EXPECTED_COLS) if EXPECTED_COLS else None
        for i, row in enumerate(req.batch):
            if n_exp is not None and len(row) != n_exp:
                raise HTTPException(
                    status_code=400,
                    detail=f"Row {i}: Expected {n_exp} features {EXPECTED_COLS}, got {len(row)}."
                )
        import pandas as pd
        X = np.array(req.batch, dtype=float)
        X_df = pd.DataFrame(X, columns=EXPECTED_COLS) if EXPECTED_COLS else X
        Xs = transform_features(PRE, X_df)
        probs = MODEL.predict(Xs).ravel()
        preds = (probs >= 0.5).astype(int).tolist()
        return {"predictions": preds, "probabilities": [float(p) for p in probs]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid batch payload: {e}")
