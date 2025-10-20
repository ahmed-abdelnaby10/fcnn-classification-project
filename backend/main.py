
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from pathlib import Path
from .models import PredictRequest, PredictBatchRequest
from .preprocessing import load_preprocessor, transform_features

app = FastAPI(title="FCNN Classifier API", version="1.0.0")

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

@app.on_event("startup")
def startup_event():
    global PRE, MODEL
    try:
        PRE = load_preprocessor()
        MODEL = load_model(MODEL_PATH)
    except Exception as e:
        print("Startup warning:", e)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": bool(MODEL is not None)}

@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None or PRE is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded")
    x = np.array(req.features, dtype=float).reshape(1, -1)
    x_scaled = transform_features(PRE, x)
    prob = float(MODEL.predict(x_scaled).ravel()[0])
    pred = int(prob >= 0.5)
    return {"prediction": pred, "probability": prob, "class_name": "Positive" if pred==1 else "Negative"}

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    if MODEL is None or PRE is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded")
    X = np.array(req.batch, dtype=float)
    Xs = transform_features(PRE, X)
    probs = MODEL.predict(Xs).ravel()
    preds = (probs >= 0.5).astype(int).tolist()
    return {"predictions": preds, "probabilities": [float(p) for p in probs]}
