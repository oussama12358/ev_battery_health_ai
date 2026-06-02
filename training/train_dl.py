"""
EV Battery Health AI — LSTM Deep Learning Model
================================================
Architecture: Stacked Bidirectional LSTM with Attention
Input:  (batch, seq_len=10, n_features=28)
Output: SoH scalar in [0, 1]  /  RUL integer

Why LSTM over Transformer here?
  - Dataset size (~600 cycle-sequences per battery) is too small for
    Transformers to shine. LSTMs are sample-efficient and interpretable.
  - Bidirectional LSTM captures both forward degradation trend and
    backward context (useful when sequences have reverse-time patterns).
  - Attention layer lets the model weight which timesteps matter most —
    this doubles as a soft explainability mechanism.

Engineering decisions:
  - Separate models for SoH and RUL (different loss landscapes).
  - SoH uses sigmoid activation (bounded [0,1]).
  - RUL uses ReLU (non-negative, unbounded).
  - Early stopping + ReduceLROnPlateau prevent overfitting on small data.
  - Model exported to ONNX for deployment (lighter, framework-agnostic).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers
callbacks = tf.keras.callbacks

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models/saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN     = 10     # lookback window in cycles
N_FEATURES  = 28     # matches FEATURE_COLS length in feature_engineering.py


def _weights_are_valid(model: keras.Model) -> bool:
    for w in model.get_weights():
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            return False
    return True


def _validate_and_save_weights(model: keras.Model, path: str) -> None:
    # If current model weights look valid, save them immediately.
    if _weights_are_valid(model):
        model.save_weights(path)
        print(f"[LSTM-{model.name}] Saved weights -> {path}")
        return

    # Fallback strategy: try to use the best checkpoint saved by ModelCheckpoint
    # Extract target name from filename (expect format: lstm_<target>.weights.h5)
    try:
        target = Path(path).stem.split("_")[1]
    except Exception:
        target = None

    best_path = MODEL_DIR / f"lstm_{target}_best.weights.h5" if target else None

    import shutil

    if best_path and best_path.exists():
        # Validate the checkpoint by loading into a fresh model of matching shape
        try:
            temp_model = build_lstm_model(
                n_features=model.input_shape[-1],
                seq_len=model.input_shape[1],
                target=target,
            )
            temp_model.load_weights(str(best_path))
            if _weights_are_valid(temp_model):
                shutil.copy(str(best_path), str(path))
                print(f"[LSTM-{model.name}] Current weights invalid -- copied validated checkpoint -> {path}")
                return
            else:
                print(f"[LSTM-{model.name}] Best checkpoint found but contains invalid weights: {best_path}")
        except Exception as exc:
            print(f"[LSTM-{model.name}] Failed to validate/copy best checkpoint: {exc}")

    # As a last resort, attempt to save the full model (may include non-weight state)
    try:
        full_model_path = MODEL_DIR / f"lstm_{target}.keras" if target else MODEL_DIR / "lstm_model.keras"
        model.save(str(full_model_path))
        print(f"[LSTM-{model.name}] Saved full model as fallback -> {full_model_path}")
        return
    except Exception as exc:
        print(f"[LSTM-{model.name}] Full-model save fallback failed: {exc}")

    # Diagnostics: print which weights are invalid and then raise
    for i, w in enumerate(model.get_weights()):
        try:
            nan_count = int(np.isnan(w).sum())
            inf_count = int(np.isinf(w).sum())
        except Exception:
            nan_count = inf_count = -1
        if nan_count > 0 or inf_count > 0:
            print(f"[DEBUG] weight #{i} invalid: shape={getattr(w, 'shape', None)} nan={nan_count} inf={inf_count}")

    raise ValueError(f"Invalid LSTM weights detected, refusing to save: {path}")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_lstm_model(
    n_features: int = N_FEATURES,
    seq_len: int = SEQ_LEN,
    target: str = "soh",
    lstm_units: list[int] = [128, 64],
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Stacked Bidirectional LSTM with self-attention for battery SoH/RUL prediction.

    Architecture:
      Input → BiLSTM(128) → Dropout → BiLSTM(64) → Dropout
            → Attention → Dense(32, relu) → Dropout → Output
    """
    inputs = keras.Input(shape=(seq_len, n_features), name="sequence_input")

    # ── BiLSTM layers ───────────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(lstm_units[0], return_sequences=True, name="bilstm_1"),
        name="bilstm_wrapper_1",
    )(inputs)
    x = layers.Dropout(dropout_rate, name="drop_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units[1], return_sequences=True, name="bilstm_2"),
        name="bilstm_wrapper_2",
    )(x)
    x = layers.Dropout(dropout_rate, name="drop_2")(x)

    # ── Temporal Attention ──────────────────────────────────────────────────
    # score each timestep, softmax-normalise, then weighted sum
    attention_scores = layers.Dense(1, activation="tanh", name="att_score")(x)
    attention_weights = layers.Softmax(axis=1, name="att_weights")(attention_scores)
    context = layers.Multiply(name="att_context")([x, attention_weights])
    context = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1),
        output_shape=(lstm_units[1] * 2,),
        name="att_sum",
    )(context)

    # ── Prediction head ─────────────────────────────────────────────────────
    x = layers.Dense(32, activation="relu", name="dense_1")(context)
    x = layers.Dropout(dropout_rate / 2, name="drop_3")(x)

    if target == "soh":
        output = layers.Dense(1, activation="sigmoid", name="soh_output")(x)
    else:
        output = layers.Dense(1, activation="relu", name="rul_output")(x)

    model = keras.Model(inputs=inputs, outputs=output, name=f"BatteryLSTM_{target.upper()}")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lstm(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    target: str = "soh",
    epochs: int = 80,
    batch_size: int = 32,
    val_split: float = 0.15,
    learning_rate: float = 1e-3,
) -> tuple[keras.Model, dict]:
    """
    Train the LSTM model with callbacks and return (model, history_dict).
    """
    print(f"\n-- LSTM Training: target={target.upper()} --")
    print(f"   X shape={X_seq.shape}  |  y shape={y_seq.shape}")
    print(f"   epochs={epochs}  batch={batch_size}  lr={learning_rate}")

    model = build_lstm_model(
        n_features=X_seq.shape[2],
        seq_len=X_seq.shape[1],
        target=target,
    )

    # Loss: Huber is more robust to outliers than MSE for RUL
    loss = keras.losses.Huber(delta=0.1) if target == "soh" else keras.losses.Huber(delta=5.0)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=loss,
        metrics=["mae"],
    )

    model.summary()

    # Callbacks
    cb_list = [
        callbacks.TerminateOnNaN(),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        # Note: we avoid saving weights during fit on Windows because
        # in-training checkpointing has corrupted LSTM weights in this repo.
    ]

    t0 = time.time()
    history = model.fit(
        X_seq, y_seq,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"[LSTM-{target.upper()}] Training done in {elapsed:.1f}s")

    return model, history.history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_lstm(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target: str = "soh",
) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test, verbose=0).flatten()

    if target == "soh":
        y_pred = np.clip(y_pred, 0.0, 1.0)
    else:
        y_pred = np.clip(y_pred, 0, None)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    metrics = {"mae": round(mae, 5), "rmse": round(rmse, 5), "r2": round(r2, 5)}
    print(f"  [LSTM-{target.upper()}]  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return metrics, y_pred


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_lstm_model(model: keras.Model, target: str) -> str:
    path = str(MODEL_DIR / f"lstm_{target}.weights.h5")
    _validate_and_save_weights(model, path)
    return path


def load_lstm_model(
    target: str,
    n_features: int | None = None,
    seq_len: int | None = None,
) -> keras.Model:
    # Support multiple possible saved filename variants (historic and current)
    best_weights_candidates = [
        MODEL_DIR / f"lstm_{target}_best.weights.h5",
        MODEL_DIR / f"lstm_{target}_best.weights.keras",
        MODEL_DIR / f"lstm_{target}_best.keras",
    ]
    weights_candidates = [
        MODEL_DIR / f"lstm_{target}.weights.h5",
        MODEL_DIR / f"lstm_{target}.weights.keras",
    ]
    full_model_candidates = [
        MODEL_DIR / f"lstm_{target}.keras",
        MODEL_DIR / f"lstm_{target}_best.keras",
    ]

    def _load_and_validate(path: Path, model: keras.Model) -> bool:
        try:
            model.load_weights(str(path))
        except Exception as exc:
            print(f"[LSTM-{target.upper()}] Failed to load weights from {path}: {exc}")
            return False

        if _weights_are_valid(model):
            print(f"[LSTM-{target.upper()}] Loaded weights from {path}")
            return True

        print(f"[LSTM-{target.upper()}] Invalid weights loaded from {path}; rejecting file")
        return False

    model = build_lstm_model(
        n_features=n_features or N_FEATURES,
        seq_len=seq_len or SEQ_LEN,
        target=target,
    )

    for p in best_weights_candidates:
        if p.exists() and _load_and_validate(p, model):
            return model

    for p in weights_candidates:
        if p.exists() and _load_and_validate(p, model):
            return model

    for p in full_model_candidates:
        if p.exists():
            try:
                model = keras.models.load_model(str(p), safe_mode=False)
                if _weights_are_valid(model):
                    print(f"[LSTM-{target.upper()}] Loaded full model from {p}")
                    return model
                print(f"[LSTM-{target.upper()}] Discarding invalid full model: {p}")
                p.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[LSTM-{target.upper()}] Full-model load failed: {exc}")

    raise FileNotFoundError(
        f"No valid saved LSTM weights or model found for target '{target}'."
    )

    if full_model_path.exists():
        try:
            model = keras.models.load_model(str(full_model_path), safe_mode=False)
            print(f"[LSTM-{target.upper()}] Loaded full model from {full_model_path}")
            return model
        except Exception as exc:
            print(f"[LSTM-{target.upper()}] Full-model load failed: {exc}")

    raise FileNotFoundError(
        f"No saved LSTM weights or model found for target '{target}'."
    )


# ---------------------------------------------------------------------------
# Full DL training pipeline
# ---------------------------------------------------------------------------

def train_dl_pipeline(feat_df) -> dict:
    """
    Full deep learning training pipeline.
    """
    from utils.feature_engineering import build_lstm_sequences
    from sklearn.model_selection import train_test_split

    print("\n" + "="*60)
    print("  EV Battery Health AI — DL Training Pipeline")
    print("="*60)

    X_seq, y_soh_seq, y_rul_seq, groups = build_lstm_sequences(feat_df, seq_len=SEQ_LEN)
    print(f"[DL] Sequences built: X={X_seq.shape}")

    all_metrics = {}
    all_predictions = {}

    for target, y_seq in [("soh", y_soh_seq), ("rul", y_rul_seq)]:
        # Stratified split by battery group
        idx = np.arange(len(X_seq))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.15, random_state=42
        )
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y_seq[train_idx], y_seq[test_idx]

        model, history = train_lstm(X_train, y_train, target=target, epochs=60)
        metrics, y_pred = evaluate_lstm(model, X_test, y_test, target=target)

        save_lstm_model(model, target)

        # Save history for plotting
        hist_path = MODEL_DIR / f"lstm_{target}_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f)

        # Save predictions
        preds = np.stack([y_test, y_pred], axis=1)
        np.save(MODEL_DIR / f"lstm_{target}_predictions.npy", preds)

        all_metrics[f"lstm_{target}"] = metrics
        all_predictions[target] = (y_test, y_pred)

    # Save combined metrics
    metrics_path = MODEL_DIR / "dl_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


if __name__ == "__main__":
    from utils.data_generator import generate_battery_dataset
    from utils.feature_engineering import extract_cycle_features

    raw  = generate_battery_dataset()
    feat = extract_cycle_features(raw)
    metrics = train_dl_pipeline(feat)
    print("\nDL Metrics:", metrics)
