"""
EV Battery Health AI — Synthetic Dataset Generator
====================================================
Mimics the NASA Battery Dataset (PCoE) structure:
- B0005, B0006, B0007, B0018 cell chemistries
- Li-ion 18650 cells, 2Ah nominal capacity
- Charge / Discharge / Impedance cycles

Engineering note:
  We generate 3 batteries (IDs) with different degradation rates to simulate
  real-world variance in cell manufacturing quality and thermal history.
  The degradation model follows a capacity-fade curve:
      C(n) = C0 * exp(-α * n) + ε
  which matches empirical lithium-ion fade closely.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Physics-inspired constants (tuned to match NASA dataset statistics)
# ---------------------------------------------------------------------------
NOMINAL_CAPACITY = 2.0          # Ah
VOLTAGE_MIN      = 2.7          # V  (discharge cut-off)
VOLTAGE_MAX      = 4.2          # V  (charge cut-off)
TEMP_AMBIENT     = 24.0         # °C
MAX_CYCLES       = 200          # per battery

BATTERY_CONFIGS = {
    "B0005": {"alpha": 0.0028, "noise_scale": 0.01, "temp_bias":  0.0},
    "B0006": {"alpha": 0.0035, "noise_scale": 0.012, "temp_bias": 3.0},
    "B0018": {"alpha": 0.0022, "noise_scale": 0.008, "temp_bias": -2.0},
}

EOL_THRESHOLD = 0.80   # State of Health < 80% → end of life


def _capacity_fade(cycle: int, alpha: float, noise_scale: float) -> float:
    """Exponential capacity fade with Gaussian noise."""
    fade = NOMINAL_CAPACITY * np.exp(-alpha * cycle)
    noise = np.random.normal(0, noise_scale)
    return float(np.clip(fade + noise, 0.5, NOMINAL_CAPACITY))


def _generate_voltage_curve(n_points: int, mode: str, soh: float) -> np.ndarray:
    """
    Synthesise a voltage-time curve for charge or discharge.
    Shape matches real Li-ion plateau behaviour.
    """
    t = np.linspace(0, 1, n_points)
    if mode == "discharge":
        # S-shaped discharge: high plateau → knee → rapid drop
        v_start = VOLTAGE_MAX * soh
        v_end   = VOLTAGE_MIN
        plateau = v_start - (v_start - v_end) * (1 / (1 + np.exp(-10 * (t - 0.75))))
        noise   = np.random.normal(0, 0.003, n_points)
        return np.clip(plateau + noise, VOLTAGE_MIN, VOLTAGE_MAX)
    else:
        # Charge: rapid rise → CC/CV plateau
        v_start = VOLTAGE_MIN + 0.1
        plateau = VOLTAGE_MAX - (VOLTAGE_MAX - v_start) * np.exp(-5 * t)
        noise   = np.random.normal(0, 0.002, n_points)
        return np.clip(plateau + noise, VOLTAGE_MIN, VOLTAGE_MAX)


def _generate_current_curve(n_points: int, mode: str, soh: float) -> np.ndarray:
    """Constant-current discharge / CC-CV charge profile."""
    t = np.linspace(0, 1, n_points)
    if mode == "discharge":
        current = -1.5 * soh * np.ones(n_points)           # negative = discharge
        noise   = np.random.normal(0, 0.02, n_points)
        return current + noise
    else:
        # CC phase then taper in CV
        cc   = 1.5 * np.ones(n_points)
        cv   = 1.5 * np.exp(-6 * (t - 0.6)) * (t > 0.6)
        taper = np.where(t > 0.6, cv, cc)
        noise = np.random.normal(0, 0.015, n_points)
        return np.clip(taper + noise, 0.0, 2.0)


def _generate_temperature_curve(
    n_points: int, cycle: int, mode: str, temp_bias: float
) -> np.ndarray:
    """
    Temperature rises during operation (Joule heating), decays at rest.
    Older cells run hotter due to increased internal resistance.
    """
    t = np.linspace(0, 1, n_points)
    ir_factor = 1 + 0.002 * cycle      # internal resistance grows with age
    delta_t   = 8.0 * ir_factor if mode == "discharge" else 5.0 * ir_factor
    temp      = TEMP_AMBIENT + temp_bias + delta_t * np.sin(np.pi * t)
    noise     = np.random.normal(0, 0.4, n_points)
    return temp + noise


# Attempt to delegate to a real-data loader if available. This allows the
# pipeline to keep calling `utils.data_generator.generate_battery_dataset()`
# while supporting real datasets (local CSV/Parquet or Kaggle) via
# `utils.data_loader`.
_real_loader = None
try:
    from . import data_loader as _dl  # type: ignore
    _real_loader = _dl
except Exception:
    _real_loader = None


def _generate_synthetic_dataset(save_path: str = "data/raw") -> pd.DataFrame:
    """Internal: original synthetic generator (renamed)."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    all_records = []

    for battery_id, cfg in BATTERY_CONFIGS.items():
        alpha = cfg["alpha"]
        noise_sc = cfg["noise_scale"]
        temp_bias = cfg["temp_bias"]

        capacities = [_capacity_fade(c, alpha, noise_sc) for c in range(MAX_CYCLES)]

        eol_cycle = next(
            (i for i, cap in enumerate(capacities) if cap / NOMINAL_CAPACITY < EOL_THRESHOLD),
            MAX_CYCLES,
        )

        for cycle in range(MAX_CYCLES):
            cap = capacities[cycle]
            soh = cap / NOMINAL_CAPACITY
            rul = max(0, eol_cycle - cycle)
            ir = 0.1 + 0.0015 * cycle + np.random.normal(0, 0.002)

            for mode in ["charge", "discharge"]:
                n_pts = np.random.randint(80, 120)
                voltage = _generate_voltage_curve(n_pts, mode, soh)
                current = _generate_current_curve(n_pts, mode, soh)
                temperature = _generate_temperature_curve(n_pts, cycle, mode, temp_bias)
                timestamps = np.linspace(0, 3600 * (2 if mode == "charge" else 1.5), n_pts)

                for i in range(n_pts):
                    all_records.append(
                        {
                            "battery_id": battery_id,
                            "cycle": cycle,
                            "mode": mode,
                            "timestamp_s": round(timestamps[i], 2),
                            "voltage": round(voltage[i], 4),
                            "current": round(current[i], 4),
                            "temperature": round(temperature[i], 4),
                            "capacity": round(cap, 5),
                            "soh": round(soh, 5),
                            "rul": rul,
                            "internal_resistance": round(ir, 5),
                        }
                    )

    df = pd.DataFrame(all_records)
    out_path = Path(save_path) / "battery_telemetry_raw.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[DataGenerator] Saved {len(df):,} rows -> {out_path}")
    return df


def generate_battery_dataset(save_path: str = "data/raw", source: str | None = None) -> pd.DataFrame:
    """Public API used by the pipeline.

    Behavior:
    - If `source` is provided (path or Kaggle dataset id) or a local
      `data/raw/battery_telemetry_raw.parquet` or `.csv` exists, try to load
      real data via `utils.data_loader` (if available) or pandas.
    - Otherwise, fall back to the original synthetic generator.
    """
    # Prefer explicit source
    if source and _real_loader:
        return _real_loader.generate_battery_dataset(source)

    # If a real file already exists in save_path, try to load it
    p_parquet = Path(save_path) / "battery_telemetry_raw.parquet"
    p_csv = Path(save_path) / "battery_telemetry_raw.csv"
    if p_parquet.exists() or p_csv.exists():
        try:
            if _real_loader:
                return _real_loader.load_raw(str(p_parquet if p_parquet.exists() else p_csv))
            # Fallback simple load via pandas
            df = pd.read_parquet(p_parquet) if p_parquet.exists() else pd.read_csv(p_csv)
            print(f"[DataGenerator] Loaded real data from {p_parquet if p_parquet.exists() else p_csv}")
            return df
        except Exception as e:
            print(f"[DataGenerator] Failed to load existing raw file: {e}; falling back to synthetic generator.")

    # No real data found → synthetic
    return _generate_synthetic_dataset(save_path)


if __name__ == "__main__":
    df = generate_battery_dataset()
    print(df.head())
    print(df.dtypes)
    print(df["battery_id"].value_counts())
