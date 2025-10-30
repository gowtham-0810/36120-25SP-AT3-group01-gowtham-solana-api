from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd

app = FastAPI(
    title="36120-25SP-AT3 – Solana Next-Day HIGH API",
    description="FastAPI service to serve Solana (SOL) next-day HIGH price prediction.",
    version="0.1.0",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "tuned_elasticnet_model.joblib"

model = None


def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")


@app.on_event("startup")
def startup_event():
    load_model()


class SolanaFeatures(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    marketCap: float | None = None


@app.get("/")
def read_root():
    return {
        "project": "36120-25SP-AT3 – Data Product with ML",
        "token": "SOLANA",
        "objective": "Predict next-day HIGH price (day +1)",
        "endpoints": {
            "/": "this message",
            "/health/": "GET – health check",
            "/predict/solana": "GET – prediction with sample data",
            "/docs": "Swagger UI",
        },
        "expected_input_for_model": {
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float (optional)",
            "marketCap": "float (optional)",
        },
        "github_repo": "https://github.com/gowtham-0810/36120-25SP-AT3-group01-gowtham-solana-api",
    }


@app.get("/health/")
def health():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "ok", "message": "Solana API is alive 💚", "model_loaded": True}


@app.get("/predict/solana")
def predict_solana():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    sample_row = {
        "open": 142.35,
        "high": 145.10,
        "low": 140.80,
        "close": 143.50,
        "volume": 58234567.0,
        "marketCap": 65000000000.0,
    }

    df = pd.DataFrame([sample_row])

    try:
        y_pred = model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "token": "SOL",
        "prediction": float(y_pred),
        "features_used": sample_row,
        "model_version": "v1",
    }


@app.post("/predict/solana")
def predict_solana_post(features: SolanaFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([features.model_dump()])
    try:
        y_pred = model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "token": "SOL",
        "prediction": float(y_pred),
        "input": features.model_dump(),
        "model_version": "v1",
    }
