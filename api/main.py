"""
EV Battery Health AI — FastAPI REST API
========================================
Production-grade prediction endpoint for battery SoH, RUL, and risk score.

Endpoints:
  POST /predict          — Single-cycle prediction
  POST /predict/batch    — Batch CSV prediction
  GET  /health           — Service health check
  GET  /model/info       — Model metadata

Usage:
  uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import io
import sys

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.risk_scoring import assess_risk
from utils.feature_engineering import FEATURE_COLS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EV Battery Health AI API",
    description=(
        "Real-time battery State of Health (SoH), Remaining Useful Life (RUL), "
        "and Risk Score prediction powered by ML + LSTM deep learning."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models/saved"

# ---------------------------------------------------------------------------
# Model registry (lazy loaded)
# ---------------------------------------------------------------------------

_models: dict = {}


def load_models() -> None:
    """Load all trained models into memory at startup."""
    global _models
    for name in ["rf_soh", "rf_rul", "xgb_soh", "xgb_rul"]:
        pkl_path = MODEL_DIR / f"{name}.pkl"
        if pkl_path.exists():
            _models[name] = joblib.load(pkl_path)
            print(f"[API] Loaded {name}")
        else:
            print(f"[API] WARNING: {name} not found at {pkl_path}")

    scaler_path = MODEL_DIR / "scaler.pkl"
    if scaler_path.exists():
        _models["scaler"] = joblib.load(scaler_path)
        print("[API] Scaler loaded")


@app.on_event("startup")
async def startup_event():
    load_models()
    print("[API] EV Battery Health AI API ready 🔋")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CycleInput(BaseModel):
    """Single battery cycle feature input."""
    cycle: int = Field(..., ge=0, le=2000, description="Charge/discharge cycle number")
    dis_v_mean: float = Field(..., description="Mean discharge voltage (V)")
    dis_v_std: float = Field(..., description="Std discharge voltage (V)")
    dis_v_min: float = Field(..., description="Min discharge voltage (V)")
    dis_v_max: float = Field(..., description="Max discharge voltage (V)")
    dis_v_skew: float = 0.0
    dis_v_kurt: float = 0.0
    dis_i_mean: float = Field(..., description="Mean discharge current (A)")
    dis_i_std: float = 0.05
    dis_t_mean: float = Field(..., description="Mean discharge temperature (°C)")
    dis_t_max: float = Field(..., description="Max discharge temperature (°C)")
    dis_t_std: float = 0.5
    dis_duration_s: float = Field(..., description="Discharge duration (s)")
    dis_charge_throughput_ah: float = Field(..., description="Discharge throughput (Ah)")
    v_drop_rate: float = 0.0
    chg_v_mean: float = 3.8
    chg_v_max: float = 4.2
    chg_i_mean: float = 1.5
    chg_t_mean: float = 25.0
    chg_duration_s: float = 7200.0
    time_to_4v_s: float = 3600.0
    ir_mean: float = Field(..., description="Mean internal resistance (Ω)")
    ir_max: float = Field(..., description="Max internal resistance (Ω)")
    delta_dis_v_mean: float = 0.0
    delta_dis_t_mean: float = 0.0
    delta_ir_mean: float = 0.0
    delta_dis_charge_throughput_ah: float = 0.0
    cumulative_throughput_ah: float = Field(..., description="Lifetime Ah throughput")

    model: Optional[str] = Field("xgb", description="Model: 'xgb' or 'rf'")

    @validator("dis_v_min")
    def v_min_reasonable(cls, v):
        if v < 2.0 or v > 4.5:
            raise ValueError("Voltage out of Li-ion range [2.0, 4.5]V")
        return v


class PredictionResponse(BaseModel):
    battery_model:    str
    soh:              float
    soh_pct:          str
    rul_cycles:       int
    risk_label:       str
    risk_score:       float
    risk_color:       str
    recommendation:   str
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _feature_vector(inp: CycleInput) -> np.ndarray:
    """Convert Pydantic input → numpy feature array."""
    row = {col: getattr(inp, col, 0.0) for col in FEATURE_COLS}
    X = np.array([[row[c] for c in FEATURE_COLS]], dtype=np.float32)
    if "scaler" in _models:
        X = _models["scaler"].transform(X)
    return X


def _predict(inp: CycleInput) -> PredictionResponse:
    model_key = inp.model if inp.model in ("xgb", "rf") else "xgb"

    soh_model = _models.get(f"{model_key}_soh")
    rul_model = _models.get(f"{model_key}_rul")

    if soh_model is None or rul_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run training/run_all.py first.",
        )

    t0 = time.perf_counter()
    X = _feature_vector(inp)
    soh = float(np.clip(soh_model.predict(X)[0], 0.0, 1.0))
    rul = int(np.clip(rul_model.predict(X)[0], 0, 9999))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    risk = assess_risk(soh, rul)
    return PredictionResponse(
        battery_model=model_key.upper(),
        soh=round(soh, 4),
        soh_pct=f"{soh*100:.1f}%",
        rul_cycles=rul,
        risk_label=risk.risk_label,
        risk_score=risk.risk_score,
        risk_color=risk.risk_color,
        recommendation=risk.recommendation,
        inference_time_ms=round(elapsed_ms, 3),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    loaded = list(_models.keys())
    return {
        "status": "healthy",
        "models_loaded": loaded,
        "ready": len(loaded) >= 3,
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "EV Battery Health AI API",
        "status": "running",
        "endpoints": ["/docs", "/redoc", "/health", "/model/info", "/predict"],
    }


@app.get("/model/info", tags=["System"])
async def model_info():
    metrics_path = MODEL_DIR / "all_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    return {
        "available_models": ["xgb", "rf", "lstm"],
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "targets": ["soh", "rul"],
        "metrics": metrics,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(inp: CycleInput):
    """
    Predict battery SoH, RUL, and risk score from a single cycle's features.
    """
    return _predict(inp)


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction from uploaded CSV file.
    CSV must have columns matching FEATURE_COLS.
    Returns JSON array of predictions.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files accepted")

    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing columns: {missing[:5]}...",
        )

    X = df[FEATURE_COLS].values.astype(np.float32)
    if "scaler" in _models:
        X = _models["scaler"].transform(X)

    results = []
    soh_model = _models.get("xgb_soh")
    rul_model = _models.get("xgb_rul")

    if soh_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    t0 = time.perf_counter()
    sohs = np.clip(soh_model.predict(X), 0.0, 1.0)
    ruls = np.clip(rul_model.predict(X), 0, 9999).astype(int)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    for i, (soh, rul) in enumerate(zip(sohs, ruls)):
        risk = assess_risk(float(soh), int(rul))
        results.append({
            "row": i,
            "soh": round(float(soh), 4),
            "soh_pct": f"{soh*100:.1f}%",
            "rul_cycles": int(rul),
            "risk_label": risk.risk_label,
            "risk_score": risk.risk_score,
        })

    return JSONResponse(content={
        "n_predictions": len(results),
        "inference_time_ms": round(elapsed_ms, 3),
        "predictions": results,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
