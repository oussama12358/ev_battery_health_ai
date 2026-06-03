"""
EV Battery Health AI — Master Training Pipeline
================================================
Run this once to generate data, train all models, and produce artefacts.

Usage:
    cd ev_battery_health_ai
    python training/run_all.py
"""

import sys
import json
import time
import warnings
from pathlib import Path
import importlib

import numpy as np
import pandas as pd
import matplotlib

warnings.filterwarnings("ignore")

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")

_training_train_ml = importlib.import_module("training.train_ml")
try:
    _training_train_dl = importlib.import_module("training.train_dl")
    train_dl_pipeline = _training_train_dl.train_dl_pipeline
    SEQ_LEN = _training_train_dl.SEQ_LEN
except Exception as e:
    _training_train_dl = None
    train_dl_pipeline = None
    SEQ_LEN = 10
    print(f"[Warning] Deep learning module unavailable, skipping DL pipeline: {e}")

_utils_data_loader = importlib.import_module("utils.data_loader")
_utils_feature_engineering = importlib.import_module("utils.feature_engineering")
_utils_risk_scoring = importlib.import_module("utils.risk_scoring")

generate_battery_dataset = _utils_data_loader.generate_battery_dataset
extract_cycle_features = _utils_feature_engineering.extract_cycle_features
BatteryAnomalyDetector = _utils_risk_scoring.BatteryAnomalyDetector
compute_degradation_rate = _utils_risk_scoring.compute_degradation_rate
train_ml_models = _training_train_ml.train_ml_models

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models/saved"
PROCESSED_DIR = ROOT_DIR / "data/processed"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def run_eda(raw_df: pd.DataFrame, feat_df: pd.DataFrame) -> None:
    """Save EDA plots for the dashboard / README."""
    print("\n── EDA Plots ─────────────────────────────────────────────────")

    # 1. SoH degradation per battery
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (bat_id, grp) in zip(
        axes, feat_df.groupby("battery_id")
    ):
        cycle_grp = grp.sort_values("cycle")
        ax.plot(cycle_grp["cycle"], cycle_grp["soh"],
                color="#2980b9", linewidth=1.8, alpha=0.9)
        ax.axhline(0.80, color="#e74c3c", linestyle="--",
                   linewidth=1.2, label="EOL threshold (80%)")
        ax.set_title(f"Battery {bat_id}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("State of Health")
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("SoH Degradation Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "soh_degradation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: soh_degradation.png")

    # 2. Temperature vs Cycle
    fig, ax = plt.subplots(figsize=(10, 4))
    for bat_id, grp in feat_df.groupby("battery_id"):
        ax.scatter(grp["cycle"], grp["dis_t_mean"], s=6, alpha=0.5, label=bat_id)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Mean Discharge Temperature (°C)")
    ax.set_title("Temperature Rise with Aging", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "temperature_vs_cycle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: temperature_vs_cycle.png")

    # 3. Internal resistance growth
    fig, ax = plt.subplots(figsize=(10, 4))
    for bat_id, grp in feat_df.groupby("battery_id"):
        g = grp.sort_values("cycle")
        ax.plot(g["cycle"], g["ir_mean"], linewidth=1.8, label=bat_id)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Mean Internal Resistance (Ω)")
    ax.set_title("Internal Resistance Growth with Aging", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "internal_resistance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: internal_resistance.png")

    # 4. Correlation heatmap
    num_cols = [c for c in feat_df.columns if feat_df[c].dtype in [np.float64, np.float32, float, int]]
    corr = feat_df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: correlation_heatmap.png")

    # 5. RUL distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(feat_df["rul"], bins=30, color="#2980b9", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Remaining Useful Life (cycles)")
    ax.set_ylabel("Count")
    ax.set_title("RUL Distribution", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "rul_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: rul_distribution.png")


def run_explainability(feat_df: pd.DataFrame) -> None:
    """SHAP analysis on XGBoost SoH model."""
    print("\n── SHAP Explainability ───────────────────────────────────────")
    from training.train_ml import train_test_split_by_battery
    from utils.feature_engineering import preprocess_features
    from utils.explainability import BatteryExplainer

    _, test_df = train_test_split_by_battery(feat_df)
    X_test, _, _ = preprocess_features(test_df, fit=False)

    try:
        from training.train_ml import XGBoostBatteryModel

        xgb_soh = XGBoostBatteryModel.load("soh")
        explainer = BatteryExplainer(xgb_soh, model_type="xgb")
        explainer.plot_bar_importance(X_test, target="soh", save=True)
        explainer.plot_summary(X_test, target="soh", save=True)
        explainer.plot_waterfall_single(X_test, sample_idx=5, target="soh", save=True)

        top_feats = explainer.get_top_features(X_test, top_n=10)
        top_feats.to_csv(PROCESSED_DIR / "top_shap_features.csv", index=False)
        print("  Saved: top_shap_features.csv")
    except Exception as e:
        print(f"  [SHAP] Skipped: {e}")

    try:
        from training.train_dl import load_lstm_model
        from utils.feature_engineering import build_lstm_sequences
        from utils.explainability import lstm_gradient_saliency

        X_seq, _, _, _ = build_lstm_sequences(feat_df, seq_len=SEQ_LEN, fit_scaler=False)
        lstm_soh = load_lstm_model(
            "soh",
            n_features=X_seq.shape[2],
            seq_len=X_seq.shape[1],
        )
        lstm_gradient_saliency(lstm_soh, X_seq, sample_idx=0, target="soh", save=True)
    except Exception as saliency_err:
        print(f"  [Saliency] Skipped: {saliency_err}")
        placeholder_path = PROCESSED_DIR / "lstm_saliency_soh.png"
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(
            0.5, 0.5,
            "LSTM saliency unavailable\nSee console output for details",
            va="center",
            ha="center",
            fontsize=14,
            color="#333333",
        )
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(placeholder_path, dpi=150, bbox_inches="tight")
        print(f"  Saved placeholder: {placeholder_path}")
        plt.close(fig)


def run_anomaly_detection(feat_df: pd.DataFrame) -> None:
    """Train and apply anomaly detector."""
    print("\n── Anomaly Detection ─────────────────────────────────────────")
    detector = BatteryAnomalyDetector()
    detector.fit(feat_df)
    detector.save()

    annotated = detector.annotate(feat_df)
    n_anomalies = annotated["is_anomaly"].sum()
    print(f"  Detected {n_anomalies} anomalous cycles "
          f"({n_anomalies/len(feat_df)*100:.1f}%)")
    annotated.to_parquet(PROCESSED_DIR / "annotated_features.parquet", index=False)
    print("  Saved: annotated_features.parquet")


def save_combined_metrics(ml_metrics: dict, dl_metrics: dict) -> None:
    """Merge and save all metrics for dashboard comparison table."""
    combined = {**ml_metrics, **dl_metrics}
    path = MODEL_DIR / "all_metrics.json"
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[Metrics] Combined metrics saved -> {path}")

    # Pretty print
    print("\n" + "="*55)
    print(f"  {'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("="*55)
    for model_key, m in combined.items():
        print(f"  {model_key:<20} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")
    print("="*55)


def main():
    t_start = time.time()
    print("\n" + "█"*60)
    print("  EV BATTERY HEALTH AI — FULL TRAINING PIPELINE")
    print("█"*60)

    # 1. Load data
    print("\n[1/6] Loading battery telemetry dataset...")
    raw_df = generate_battery_dataset(save_path="data/raw")

    # 2. Feature engineering
    print("\n[2/6] Engineering cycle-level features...")
    feat_df = extract_cycle_features(raw_df)
    feat_df.to_parquet(PROCESSED_DIR / "cycle_features.parquet", index=False)
    print(f"  Feature matrix: {feat_df.shape[0]} cycles × {feat_df.shape[1]} columns")

    # 3. EDA
    print("\n[3/6] Generating EDA plots...")
    run_eda(raw_df, feat_df)

    # 4. ML models
    print("\n[4/6] Training ML baselines (RF + XGBoost)...")
    ml_metrics = train_ml_models(feat_df)

    # 5. DL model
    if train_dl_pipeline is not None:
        print("\n[5/6] Training LSTM deep learning model...")
        dl_metrics = train_dl_pipeline(feat_df)
    else:
        print("\n[5/6] Skipping LSTM deep learning model because TensorFlow is unavailable.")
        dl_metrics = {}
    # 6. Explainability + Anomaly Detection
    print("\n[6/6] Running explainability and anomaly detection...")
    run_explainability(feat_df)
    run_anomaly_detection(feat_df)

    save_combined_metrics(ml_metrics, dl_metrics)

    elapsed = time.time() - t_start
    print(f"\n✅ Full pipeline completed in {elapsed/60:.1f} minutes")
    print(f"   Models -> {MODEL_DIR}/")
    print(f"   Plots  -> {PROCESSED_DIR}/")
    print("\n   Next: run the dashboard with:")
    print("   streamlit run app/dashboard.py\n")


if __name__ == "__main__":
    main()
