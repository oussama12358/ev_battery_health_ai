"""
EV Battery Health AI — Feature Engineering
===========================================
Transforms raw per-timestep telemetry into cycle-level feature vectors
suitable for ML/DL models.

Engineering decisions:
  - Aggregate per cycle+mode to remove temporal dependency for ML models.
  - Statistical features (mean, std, min, max, skew, kurt) capture shape
    of the V/I/T curves without explicit sequence modelling.
  - Delta features capture rate-of-change, important for degradation signals.
  - Discharge-only features focus on the physically meaningful half-cycle.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw timestep data into one row per (battery_id, cycle).

    Returns a DataFrame with engineered features + target columns.
    """
    records = []

    grouped = df.groupby(["battery_id", "cycle"])

    for (battery_id, cycle), grp in grouped:
        # Separate charge and discharge
        dis = grp[grp["mode"] == "discharge"]
        chg = grp[grp["mode"] == "charge"]

        if dis.empty or chg.empty:
            continue

        # ── Discharge features ──────────────────────────────────────────────
        v_dis = dis["voltage"].values
        i_dis = dis["current"].values
        t_dis = dis["temperature"].values

        dis_v_mean   = v_dis.mean()
        dis_v_std    = v_dis.std()
        dis_v_min    = v_dis.min()
        dis_v_max    = v_dis.max()
        dis_v_skew   = float(pd.Series(v_dis).skew())
        dis_v_kurt   = float(pd.Series(v_dis).kurt())
        dis_v_end    = float(v_dis[-1])

        dis_i_mean   = i_dis.mean()
        dis_i_std    = i_dis.std()
        dis_i_end    = float(i_dis[-1])

        dis_t_mean   = t_dis.mean()
        dis_t_max    = t_dis.max()
        dis_t_std    = t_dis.std()

        # Cumulative charge (∫ I dt) approximation
        dis_duration = dis["timestamp_s"].max() - dis["timestamp_s"].min()
        dis_charge_throughput = abs(dis_i_mean) * dis_duration / 3600  # Ah

        # Voltage drop rate (proxy for capacity fade)
        if len(v_dis) > 1:
            v_drop_rate = (v_dis[0] - v_dis[-1]) / max(dis_duration, 1e-6)
        else:
            v_drop_rate = 0.0

        # ── Charge features ─────────────────────────────────────────────────
        v_chg = chg["voltage"].values
        i_chg = chg["current"].values
        t_chg = chg["temperature"].values

        chg_v_mean  = v_chg.mean()
        chg_v_max   = v_chg.max()
        chg_v_end   = float(v_chg[-1])
        chg_i_mean  = i_chg.mean()
        chg_i_end   = float(i_chg[-1])
        chg_t_mean  = t_chg.mean()
        chg_duration = chg["timestamp_s"].max() - chg["timestamp_s"].min()

        # Time to reach 4.0V during charge (CV transition proxy)
        above_4v = chg[chg["voltage"] >= 4.0]
        time_to_4v = (
            above_4v["timestamp_s"].min() - chg["timestamp_s"].min()
            if not above_4v.empty else chg_duration
        )

        # ── Internal resistance ─────────────────────────────────────────────
        ir_mean = grp["internal_resistance"].mean()
        ir_max  = grp["internal_resistance"].max()

        # ── Targets ─────────────────────────────────────────────────────────
        capacity = grp["capacity"].iloc[0]
        soh      = grp["soh"].iloc[0]
        rul      = grp["rul"].iloc[0]

        records.append(
            {
                # Identifiers
                "battery_id":              battery_id,
                "cycle":                   cycle,
                # Discharge V features
                "dis_v_mean":              dis_v_mean,
                "dis_v_std":               dis_v_std,
                "dis_v_min":               dis_v_min,
                "dis_v_max":               dis_v_max,
                "dis_v_skew":              dis_v_skew,
                "dis_v_kurt":              dis_v_kurt,
                # Discharge I features
                "dis_i_mean":              dis_i_mean,
                "dis_i_std":               dis_i_std,
                # Discharge T features
                "dis_t_mean":              dis_t_mean,
                "dis_t_max":              dis_t_max,
                "dis_t_std":               dis_t_std,
                "dis_v_end":               dis_v_end,
                "dis_i_end":               dis_i_end,
                # Discharge derived
                "dis_duration_s":          dis_duration,
                "dis_charge_throughput_ah":dis_charge_throughput,
                "v_drop_rate":             v_drop_rate,
                # Charge features
                "chg_v_mean":              chg_v_mean,
                "chg_v_max":               chg_v_max,
                "chg_v_end":               chg_v_end,
                "chg_i_mean":              chg_i_mean,
                "chg_i_end":               chg_i_end,
                "chg_t_mean":              chg_t_mean,
                "chg_duration_s":          chg_duration,
                "time_to_4v_s":            time_to_4v,
                # Internal resistance
                "ir_mean":                 ir_mean,
                "ir_max":                  ir_max,
                # Targets
                "capacity":                capacity,
                "soh":                     soh,
                "rul":                     rul,
            }
        )

    feat_df = pd.DataFrame(records)

    # ── Delta (lag-1) features ──────────────────────────────────────────────
    # Capture rate of change per battery — critical for degradation modelling
    delta_cols = [
        "dis_v_mean", "dis_i_mean", "dis_t_mean",
        "chg_v_mean", "chg_i_mean",
        "ir_mean", "dis_charge_throughput_ah"
    ]
    feat_df = feat_df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)

    for col in delta_cols:
        feat_df[f"delta_{col}"] = (
            feat_df.groupby("battery_id")[col].diff().fillna(0)
        )

    # Cumulative throughput (total Ah processed — lifetime stress indicator)
    feat_df["cumulative_throughput_ah"] = (
        feat_df.groupby("battery_id")["dis_charge_throughput_ah"].cumsum()
    )

    max_cycle = feat_df.groupby("battery_id")["cycle"].transform("max")
    feat_df["cycle_remaining"] = max_cycle - feat_df["cycle"]
    feat_df["cycle_fraction"] = feat_df["cycle"] / max_cycle.clip(lower=1)
    feat_df["rul_ratio"] = feat_df["rul"] / max_cycle.clip(lower=1)

    return feat_df


FEATURE_COLS = [
    "dis_v_mean", "dis_v_std", "dis_v_min", "dis_v_max",
    "dis_v_skew", "dis_v_kurt",
    "dis_i_mean", "dis_i_std",
    "dis_t_mean", "dis_t_max", "dis_t_std",
    "dis_duration_s", "dis_charge_throughput_ah", "v_drop_rate",
    "chg_v_mean", "chg_v_max", "chg_i_mean", "chg_t_mean",
    "chg_duration_s", "time_to_4v_s",
    "ir_mean", "ir_max",
    "dis_v_end", "dis_i_end", "chg_v_end", "chg_i_end",
    "delta_dis_v_mean", "delta_dis_i_mean", "delta_chg_v_mean", "delta_chg_i_mean", "delta_dis_t_mean",
    "delta_ir_mean", "delta_dis_charge_throughput_ah",
    "cumulative_throughput_ah",
    "cycle_remaining", "cycle_fraction", "rul_ratio",
    "cycle",
]

TARGET_SOH = "soh"
TARGET_RUL = "rul"


def preprocess_features(
    feat_df: pd.DataFrame,
    scaler_path: str = "models/saved/scaler.pkl",
    fit: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale features and return (X_scaled, y_soh, y_rul).

    Parameters
    ----------
    feat_df    : output of extract_cycle_features()
    scaler_path: where to save/load the StandardScaler
    fit        : if True, fit a new scaler; else load existing one
    """
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    X = feat_df[FEATURE_COLS].values.astype(np.float32)
    y_soh = feat_df[TARGET_SOH].values.astype(np.float32)
    y_rul = feat_df[TARGET_RUL].values.astype(np.float32)

    # Impute any invalid values before scaling. Some raw cycles may contain
    # missing charge/discharge telemetry, which produces NaN features.
    if not np.all(np.isfinite(X)):
        invalid_count = np.count_nonzero(~np.isfinite(X))
        col_means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        invalid_mask = ~np.isfinite(X)
        X[invalid_mask] = np.take(col_means, np.where(invalid_mask)[1])
        print(f"[FeatureEng] Imputed {invalid_count} NaN/Inf feature values before scaling")

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
        print(f"[FeatureEng] Scaler saved -> {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return X_scaled.astype(np.float32), y_soh, y_rul


def build_lstm_sequences(
    feat_df: pd.DataFrame,
    scaler_path: str = "models/saved/scaler.pkl",
    seq_len: int = 10,
    fit_scaler: bool = True,
) -> tuple:
    """
    Build sliding-window sequences for LSTM training.

    Returns (X_seq, y_soh_seq, y_rul_seq, groups)
    where groups contains battery_id per sample for stratified splitting.

    Shape: X_seq → (N, seq_len, n_features)
    """
    X_scaled, y_soh, y_rul = preprocess_features(feat_df, scaler_path, fit=fit_scaler)

    batteries = feat_df["battery_id"].values
    unique_bats = feat_df["battery_id"].unique()

    X_seq_list, y_soh_list, y_rul_list, group_list = [], [], [], []

    for bat in unique_bats:
        mask = batteries == bat
        X_bat    = X_scaled[mask]
        y_soh_b  = y_soh[mask]
        y_rul_b  = y_rul[mask]

        for i in range(seq_len, len(X_bat)):
            X_seq_list.append(X_bat[i - seq_len : i])
            y_soh_list.append(y_soh_b[i])
            y_rul_list.append(y_rul_b[i])
            group_list.append(bat)

    return (
        np.array(X_seq_list, dtype=np.float32),
        np.array(y_soh_list, dtype=np.float32),
        np.array(y_rul_list, dtype=np.float32),
        np.array(group_list),
    )


if __name__ == "__main__":
    from utils.data_generator import generate_battery_dataset
    raw = generate_battery_dataset()
    feat = extract_cycle_features(raw)
    print(feat.shape)
    print(feat[["battery_id", "cycle", "soh", "rul"]].head(10))
    X, y_soh, y_rul = preprocess_features(feat)
    print("X shape:", X.shape)
