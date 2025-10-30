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

# paths
BASE_DIR = Path(__file__).resolve().parent      # /app/app
PROJECT_ROOT = BASE_DIR.parent                  # /app
MODEL_PATH = PROJECT_ROOT / "models" / "tuned_elasticnet_model.joblib"

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


def build_features(payload: dict) -> pd.DataFrame:
    """
    Take the raw incoming dict and build ALL features
    that the model was trained on.
    """
    open_ = payload["open"]
    high_ = payload["high"]
    low_ = payload["low"]
    close_ = payload["close"]
    volume_ = payload.get("volume", 0.0) or 0.0
    mc_ = payload.get("marketCap", 0.0) or 0.0

    # derived features (simple, safe versions)
    price_momentum = close_ - open_
    if low_ != 0:
        price_range_pct = (high_ - low_) / low_
    else:
        price_range_pct = 0.0

    if mc_ != 0:
        volume_to_mc_ratio = volume_ / mc_
    else:
        volume_to_mc_ratio = 0.0

    # IMPORTANT: order + names must match training
    data = {
        "open": open_,
        "high": high_,
        "low": low_,
        "close": close_,
        "volume": volume_,
        "marketCap": mc_,
        "price_momentum": price_momentum,
        "price_range_pct": price_range_pct,
        "volume_to_mc_ratio": volume_to_mc_ratio,
    }

    return pd.DataFrame([data])


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

    # this is just a dummy row
    sample_row = {
        "open": 142.35,
        "high": 145.10,
        "low": 140.80,
        "close": 143.50,
        "volume": 58_234_567.0,
        "marketCap": 65_000_000_000.0,
    }

    df = build_features(sample_row)

    try:
        y_pred = model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "token": "SOL",
        "prediction": float(y_pred),
        "features_used": df.to_dict(orient="records")[0],
        "model_version": "v1",
    }


@app.post("/predict/solana")
def predict_solana_post(features: SolanaFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = build_features(features.model_dump())
    try:
        y_pred = model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "token": "SOL",
        "prediction": float(y_pred),
        "input": df.to_dict(orient="records")[0],
        "model_version": "v1",
    }
