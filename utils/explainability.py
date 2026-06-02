"""
EV Battery Health AI — Explainable AI (SHAP)
=============================================
Generates SHAP explanations for both RF and XGBoost models.

Why SHAP?
  - Model-agnostic unified framework (TreeSHAP for tree models = fast exact values)
  - Returns per-feature contributions for every prediction
  - Summary plots show global feature importance
  - Waterfall/force plots show individual prediction breakdown

Engineering note:
  For the LSTM, we use gradient-based saliency (GradientTape) instead of SHAP
  because SHAP's DeepExplainer has limitations with custom Keras layers.
  The output is a per-timestep, per-feature importance heatmap.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from utils.feature_engineering import FEATURE_COLS

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = ROOT_DIR / "data/processed"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# TreeSHAP explainer (RF + XGBoost)
# ---------------------------------------------------------------------------

class BatteryExplainer:
    """
    SHAP-based explainer for tree models.
    """

    def __init__(self, model, model_type: str = "xgb"):
        """
        Parameters
        ----------
        model      : fitted RF or XGB wrapper (has .model attribute)
        model_type : "xgb" | "rf"
        """
        self.model_type = model_type
        self.feature_names = FEATURE_COLS

        underlying = model.model   # unwrap our wrapper class
        self.explainer = shap.TreeExplainer(underlying)
        print(f"[SHAP] TreeExplainer initialised for {model_type.upper()}")

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for given feature matrix.
        Returns array of shape (n_samples, n_features).
        """
        shap_values = self.explainer.shap_values(X)
        # For multi-output RF, shap_values is a list; take first output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return shap_values

    def plot_summary(
        self,
        X: np.ndarray,
        target: str = "soh",
        save: bool = True,
    ) -> plt.Figure:
        """
        Beeswarm summary plot — shows feature impact distribution.
        """
        shap_values = self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            show=False,
            plot_type="dot",
            max_display=15,
        )
        plt.title(
            f"SHAP Feature Importance — {self.model_type.upper()} ({target.upper()})",
            fontsize=14, fontweight="bold", pad=15,
        )
        plt.tight_layout()

        if save:
            path = PLOT_DIR / f"shap_summary_{self.model_type}_{target}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[SHAP] Summary plot saved -> {path}")

        return fig

    def plot_bar_importance(
        self,
        X: np.ndarray,
        target: str = "soh",
        top_n: int = 12,
        save: bool = True,
    ) -> plt.Figure:
        """
        Bar chart of mean |SHAP| values — clean, presentation-ready.
        """
        shap_values = self.compute_shap_values(X)
        mean_abs    = np.abs(shap_values).mean(axis=0)
        importance  = pd.Series(mean_abs, index=self.feature_names)
        importance  = importance.sort_values(ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, top_n))
        bars = ax.barh(importance.index, importance.values, color=colors, edgecolor="white")
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title(
            f"Top {top_n} Features — {self.model_type.upper()} ({target.upper()})",
            fontsize=13, fontweight="bold",
        )

        # Annotate bars
        for bar, val in zip(bars, importance.values):
            ax.text(
                val + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="#333333"
            )

        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("white")
        plt.tight_layout()

        if save:
            path = PLOT_DIR / f"shap_bar_{self.model_type}_{target}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[SHAP] Bar plot saved -> {path}")

        return fig

    def plot_waterfall_single(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        target: str = "soh",
        save: bool = True,
    ) -> plt.Figure:
        """
        Waterfall plot for a single prediction — shows how each feature
        pushes the prediction above/below the base value.
        """
        shap_values = self.compute_shap_values(X)
        base_value  = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]

        sv = shap_values[sample_idx]
        features = X[sample_idx]
        feature_names = [
            f"{n}={v:.3f}" for n, v in zip(self.feature_names, features)
        ]

        # Sort by abs impact
        order = np.argsort(np.abs(sv))[::-1][:10]
        sv_top = sv[order]
        fn_top = [feature_names[i] for i in order]

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#e74c3c" if v > 0 else "#27ae60" for v in sv_top]
        ax.barh(fn_top[::-1], sv_top[::-1], color=colors[::-1], edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on model output)", fontsize=11)
        ax.set_title(
            f"Prediction Breakdown — Sample #{sample_idx} ({target.upper()})",
            fontsize=13, fontweight="bold",
        )
        red_patch   = mpatches.Patch(color="#e74c3c", label="Increases prediction")
        green_patch = mpatches.Patch(color="#27ae60", label="Decreases prediction")
        ax.legend(handles=[red_patch, green_patch], loc="lower right", fontsize=9)
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("white")
        plt.tight_layout()

        if save:
            path = PLOT_DIR / f"shap_waterfall_{self.model_type}_{target}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[SHAP] Waterfall plot saved -> {path}")

        return fig

    def get_top_features(self, X: np.ndarray, top_n: int = 10) -> pd.DataFrame:
        """Return top N features by mean |SHAP| as DataFrame."""
        shap_values = self.compute_shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# LSTM saliency (gradient-based pseudo-explanation)
# ---------------------------------------------------------------------------

def lstm_gradient_saliency(
    model,
    X_seq: np.ndarray,
    sample_idx: int = 0,
    target: str = "soh",
    save: bool = True,
) -> np.ndarray:
    """
    Compute gradient-based saliency for LSTM:
      saliency[t, f] = |∂output / ∂input[t, f]|

    Highlights which timesteps and features the LSTM focused on.
    """
    import tensorflow as tf

    x_sample = tf.Variable(
        X_seq[sample_idx : sample_idx + 1], dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        tape.watch(x_sample)
        output = model(x_sample, training=False)

    grads = tape.gradient(output, x_sample).numpy()[0]   # (seq_len, n_features)
    saliency = np.abs(grads)
    saliency = np.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)
    display_saliency = saliency

    # Normalize for display so low-magnitude gradients remain visible.
    if saliency.size and not np.allclose(saliency, 0.0):
        display_saliency = saliency / (np.max(saliency) + 1e-12)

    feature_names = FEATURE_COLS[: display_saliency.shape[1]]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        display_saliency.T,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(range(display_saliency.shape[0]))
    ax.set_xticklabels(range(display_saliency.shape[0]), fontsize=9)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel("Time Step (cycle lookback)", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.set_title(
        f"LSTM Gradient Saliency — Sample #{sample_idx} ({target.upper()})",
        fontsize=14, fontweight="bold",
    )
    cbar = plt.colorbar(im, ax=ax, label="Normalized |∂output/∂input|")
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()

    if save:
        path = PLOT_DIR / f"lstm_saliency_{target}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Saliency] Saved -> {path}")

    plt.close(fig)
    return saliency


if __name__ == "__main__":
    print("[SHAP] Module loaded — run via training/run_explainability.py")
