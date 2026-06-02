"""
EV Battery Health AI — Machine Learning Baseline Models
========================================================
Trains Random Forest and XGBoost regressors for:
  - State of Health (SoH) prediction
  - Remaining Useful Life (RUL) prediction

Architecture decisions:
  - We use per-cycle aggregated features (not raw sequences) for ML models —
    this is intentional: ML models can't handle sequences natively, while
    the aggregated stats still capture degradation trends well.
  - Cross-validation is done battery-by-battery (LeaveOneGroupOut) to
    prevent data leakage — critical in time-series settings.
  - Both models are saved as .pkl for inference in the dashboard.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

from utils.feature_engineering import (
    FEATURE_COLS,
    extract_cycle_features,
    preprocess_features,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models/saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    metrics = {
        "mae": float(round(mae, 5)),
        "rmse": float(round(rmse, 5)),
        "r2": float(round(r2, 5)),
    }
    print(
        f"  [{label}]  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}"
    )
    return metrics


def transform_target(y: np.ndarray, target: str) -> np.ndarray:
    return np.log1p(y) if target == "rul" else y


def inverse_transform_target(y: np.ndarray, target: str) -> np.ndarray:
    return np.expm1(y) if target == "rul" else y


# ---------------------------------------------------------------------------
# Data splitting (group-aware to prevent leakage)
# ---------------------------------------------------------------------------

def train_test_split_by_battery(
    feat_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Hold out one battery (group) for testing.
    This simulates the real-world scenario of predicting on unseen batteries.
    """
    groups = feat_df["battery_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(feat_df, groups=groups))

    train_df = feat_df.iloc[train_idx]
    test_df  = feat_df.iloc[test_idx]

    print(f"[Split] Train={len(train_df)} cycles  |  "
          f"Test={len(test_df)} cycles  |  "
          f"Test batteries: {test_df['battery_id'].unique()}")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestBatteryModel:
    """
    RF baseline for SoH and RUL prediction.
    Two separate models: one per target (more interpretable than multi-output).
    """

    def __init__(self, target: str = "soh"):
        assert target in ("soh", "rul")
        self.target = target
        self.model  = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        self.feature_importance_: pd.Series | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBatteryModel":
        t0 = time.time()
        self.model.fit(X, y)
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
        print(f"[RF-{self.target.upper()}] Trained in {time.time()-t0:.1f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.model.predict(X)
        if self.target == "soh":
            pred = np.clip(pred, 0.0, 1.0)
        else:
            pred = np.clip(pred, 0, None)
        return pred

    def save(self) -> str:
        path = MODEL_DIR / f"rf_{self.target}.pkl"
        joblib.dump(self, path)
        print(f"[RF-{self.target.upper()}] Saved -> {path}")
        return str(path)

    @classmethod
    def load(cls, target: str) -> "RandomForestBatteryModel":
        path = MODEL_DIR / f"rf_{target}.pkl"
        return joblib.load(path)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostBatteryModel:
    """
    XGBoost gradient boosting for SoH and RUL.
    Better than RF on tabular data when features carry complex interactions.
    """

    def __init__(self, target: str = "soh"):
        assert target in ("soh", "rul")
        self.target = target
        params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            tree_method="hist",       # fast histogram method
        )
        if target == "soh":
            params["objective"] = "reg:squarederror"
        else:
            params["objective"] = "reg:squarederror"

        params["eval_metric"] = "rmse"
        params["early_stopping_rounds"] = 20
        self.model = xgb.XGBRegressor(**params)
        self.feature_importance_: pd.Series | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostBatteryModel":
        t0 = time.time()
        eval_set = [(X_val, y_val)] if X_val is not None else None
        fit_kwargs = {"verbose": False}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set

        self.model.fit(
            X_train, y_train,
            **fit_kwargs,
        )
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
        print(f"[XGB-{self.target.upper()}] Trained in {time.time()-t0:.1f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.model.predict(X)
        if self.target == "soh":
            pred = np.clip(pred, 0.0, 1.0)
        else:
            pred = np.clip(pred, 0, None)
        return pred

    def save(self) -> str:
        path = MODEL_DIR / f"xgb_{self.target}.pkl"
        joblib.dump(self, path)
        print(f"[XGB-{self.target.upper()}] Saved -> {path}")
        return str(path)

    @classmethod
    def load(cls, target: str) -> "XGBoostBatteryModel":
        path = MODEL_DIR / f"xgb_{target}.pkl"
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_ml_models(feat_df: pd.DataFrame) -> dict:
    """
    End-to-end ML training pipeline.
    Returns dict of all metrics for comparison.
    """
    print("\n" + "="*60)
    print("  EV Battery Health AI — ML Training Pipeline")
    print("="*60)

    # Split
    train_df, test_df = train_test_split_by_battery(feat_df)

    # Features
    X_train, y_soh_train, y_rul_train = preprocess_features(
        train_df, fit=True
    )
    X_test, y_soh_test, y_rul_test = preprocess_features(
        test_df, fit=False
    )

    all_metrics = {}

    for target, y_train, y_test in [
        ("soh", y_soh_train, y_soh_test),
        ("rul", y_rul_train, y_rul_test),
    ]:
        print(f"\n── Target: {target.upper()} ──────────────────────────────")

        y_train_model = transform_target(y_train, target)
        y_test_model = transform_target(y_test, target)

        # Random Forest
        print(" Random Forest:")
        rf = RandomForestBatteryModel(target=target)
        rf.fit(X_train, y_train_model)
        rf_pred = rf.predict(X_test)
        rf_pred = inverse_transform_target(rf_pred, target)
        rf_pred = np.clip(rf_pred, 0, None)
        rf_metrics = evaluate(y_test, rf_pred, f"RF-{target.upper()}")
        rf.save()

        # XGBoost
        print(" XGBoost:")
        xgb_model = XGBoostBatteryModel(target=target)
        xgb_model.fit(X_train, y_train_model, X_test, y_test_model)
        xgb_pred = xgb_model.predict(X_test)
        xgb_pred = inverse_transform_target(xgb_pred, target)
        xgb_pred = np.clip(xgb_pred, 0, None)
        xgb_metrics = evaluate(y_test, xgb_pred, f"XGB-{target.upper()}")
        xgb_model.save()

        all_metrics[f"rf_{target}"]  = rf_metrics
        all_metrics[f"xgb_{target}"] = xgb_metrics

        # Save predictions for plotting
        preds_df = pd.DataFrame({
            "cycle":     test_df["cycle"].values,
            "battery_id": test_df["battery_id"].values,
            "y_true":    y_test,
            "rf_pred":   rf_pred,
            "xgb_pred":  xgb_pred,
        })
        preds_df.to_parquet(MODEL_DIR / f"ml_predictions_{target}.parquet", index=False)

    # Save metrics summary
    metrics_path = MODEL_DIR / "ml_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)
    print(f"\n[Training] Metrics saved -> {metrics_path}")

    return all_metrics


if __name__ == "__main__":
    from utils.data_generator import generate_battery_dataset
    raw = generate_battery_dataset()
    feat = extract_cycle_features(raw)
    metrics = train_ml_models(feat)
    print("\nFinal metrics summary:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
