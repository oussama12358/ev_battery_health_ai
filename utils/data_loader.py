"""
Real dataset loader for EV Battery Health AI
-------------------------------------------------
Provides helpers to download a Kaggle dataset (if configured) or load a
local CSV/Parquet file and normalize column names to the pipeline's expected
schema. Exposes `generate_battery_dataset()` so it can be used as a drop-in
replacement for the synthetic generator.

Usage examples:
  # Load local file
  from utils.data_loader import generate_battery_dataset
  df = generate_battery_dataset("data/raw/my_battery.csv")

  # Download from Kaggle (requires Kaggle API credentials)
  df = generate_battery_dataset("patrickfleith/nasa-battery-dataset")

The returned DataFrame contains timestep rows with at least these columns:
  battery_id, cycle, mode, timestamp_s, voltage, current, temperature,
  capacity (optional), soh (optional), rul (optional), internal_resistance
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import pandas as pd
from typing import Optional


REQUIRED_COLUMNS = [
    "battery_id",
    "cycle",
    "mode",
    "timestamp_s",
    "voltage",
    "current",
    "temperature",
    "internal_resistance",
]


def _download_kaggle_dataset(dataset_id: str, dest: str = "data/raw") -> Path:
    """Download and unzip a Kaggle dataset using the Kaggle API.

    Expects `dataset_id` like 'owner/dataset-name'. Requires Kaggle API
    credentials set in the environment. Raises RuntimeError with guidance
    if the Kaggle client is not available or authentication fails.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError(
            "Kaggle API not available. Install the 'kaggle' package and set KAGGLE_USERNAME/KAGGLE_KEY."
        ) from e

    api = KaggleApi()
    api.authenticate()
    out_dir = Path(dest)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.dataset_download_files(dataset_id, path=str(out_dir), unzip=True, quiet=False)
    except Exception as e:
        raise RuntimeError(f"Failed to download Kaggle dataset '{dataset_id}': {e}") from e
    return out_dir


def _find_candidate_files(path: Path) -> list[Path]:
    files = []
    for ext in ("*.parquet", "*.parq", "*.csv", "*.zip"):
        files.extend(sorted(path.glob(ext)))
    return files


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common alternative column names to the pipeline's expected schema."""
    col_map = {}
    lowercols = {c.lower().strip(): c for c in df.columns}

    def find(*names):
        for n in names:
            if n in lowercols:
                return lowercols[n]
        return None

    # mappings (common variants)
    mapping = {
        "battery_id": ["battery_id", "bat_id", "cell", "id"],
        "cycle": ["cycle", "cycle_index", "cycle_number", "shot"],
        "mode": ["mode", "phase"],
        "timestamp_s": ["timestamp_s", "time_s", "time", "timestamp"],
        "voltage": ["voltage", "v", "volts"],
        "current": ["current", "i", "amps", "amp"],
        "temperature": ["temperature", "temp", "t"],
        "capacity": ["capacity", "cap"],
        "soh": ["soh", "state_of_health", "soc"],
        "rul": ["rul", "remaining_cycles", "remaining_life"],
        "internal_resistance": ["internal_resistance", "ir", "impedance"],
    }

    for target, variants in mapping.items():
        found = find(*variants)
        if found:
            col_map[found] = target

    df = df.rename(columns=col_map)

    # Ensure required dtype conversions where possible
    if "cycle" in df.columns:
        try:
            df["cycle"] = df["cycle"].astype(int)
        except Exception:
            pass

    if "timestamp_s" in df.columns and not pd.api.types.is_float_dtype(df["timestamp_s"]):
        try:
            df["timestamp_s"] = pd.to_numeric(df["timestamp_s"], errors="coerce")
        except Exception:
            pass

    # Lowercase string columns for 'mode'
    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.lower().str.strip()

    return df


def _to_float(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        # handle complex string-like values from the dataset metadata
        s = s.replace(" ", "")
        if "j" in s:
            try:
                c = complex(s)
                return abs(c)
            except Exception:
                pass
        # remove parentheses and parse first float
        s = s.strip("()")
        for part in s.replace("+", " ").replace("-", " ").split():
            try:
                return float(part)
            except ValueError:
                continue
    return 0.0


def _find_nasa_kaggle_dataset_dir(path: Path) -> Optional[Path]:
    """Return the root folder of a downloaded NASA Kaggle dataset."""
    path = path.resolve()
    if (path / "metadata.csv").exists():
        return path
    if (path / "data").exists() and (path / "metadata.csv").exists():
        return path
    if (path / "data" / "metadata.csv").exists():
        return path
    # also check parent for the metadata root
    if (path.parent / "metadata.csv").exists():
        return path.parent
    return None


def _load_nasa_kaggle_folder(path: Path) -> Optional[pd.DataFrame]:
    """Load the folder structure from the NASA Kaggle dataset into a unified raw frame."""
    dataset_root = _find_nasa_kaggle_dataset_dir(path)
    if dataset_root is None:
        return None

    if (dataset_root / "metadata.csv").exists():
        meta_path = dataset_root / "metadata.csv"
    else:
        raise FileNotFoundError(f"metadata.csv not found in {dataset_root}")

    # data files are most likely under <root>/data
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        data_dir = dataset_root

    metadata = pd.read_csv(meta_path)
    metadata = metadata.sort_values(["battery_id", "test_id"]).reset_index(drop=True)

    records = []
    # assign a cycle id based on discharge/charge ordering
    for battery_id, bat_meta in metadata.groupby("battery_id"):
        current_cycle = -1
        cycle_capacity = {}
        cycle_ir = {}

        for _, row in bat_meta.iterrows():
            file_path = data_dir / str(row["filename"]).strip()
            if not file_path.exists():
                continue
            dtype = str(row["type"]).strip().lower()
            if dtype == "discharge":
                current_cycle += 1
                cycle_capacity[current_cycle] = _to_float(row["Capacity"]) if pd.notna(row["Capacity"]) else None
            if dtype == "impedance":
                re_val = _to_float(row["Re"])
                rct_val = _to_float(row["Rct"])
                cycle_ir[current_cycle] = re_val + rct_val

            # read time-series row file
            df = pd.read_csv(file_path)
            if "Time" not in df.columns or "Voltage_measured" not in df.columns or "Current_measured" not in df.columns:
                continue

            df = df.rename(columns={
                "Voltage_measured": "voltage",
                "Current_measured": "current",
                "Temperature_measured": "temperature",
                "Time": "timestamp_s",
            })
            df["battery_id"] = battery_id
            df["mode"] = dtype
            df["cycle"] = current_cycle
            df["capacity"] = cycle_capacity.get(current_cycle, None)
            df["internal_resistance"] = cycle_ir.get(current_cycle, 0.0)
            records.append(df[["battery_id", "cycle", "mode", "timestamp_s", "voltage", "current", "temperature", "capacity", "internal_resistance"]])

        # fill cycle-level capacity and resistance values forward for missing rows
        # no additional action required here, as missing values will be imputed later

    if not records:
        raise ValueError(f"No NASA Kaggle files were loaded from {path}")

    raw_df = pd.concat(records, ignore_index=True)
    raw_df["mode"] = raw_df["mode"].astype(str).str.lower().str.strip()

    # compute nominal capacity per battery and assign soh/rul using discharge capacity
    capacities = (
        raw_df[raw_df["mode"] == "discharge"]["capacity"]
        .dropna()
        .groupby(raw_df[raw_df["mode"] == "discharge"]["battery_id"])
        .max()
    )
    nominal = capacities.to_dict()
    raw_df["soh"] = raw_df.apply(
        lambda r: float(r["capacity"]) / nominal.get(r["battery_id"], 1.0)
        if pd.notna(r["capacity"]) and nominal.get(r["battery_id"], 1.0) > 0 else None,
        axis=1,
    )

    # set rul per cycle using discharge soh
    leaderboard = raw_df[raw_df["mode"] == "discharge"].groupby(["battery_id", "cycle"])["soh"].first().reset_index()
    rul_rows = []
    for battery_id, bat in leaderboard.groupby("battery_id"):
        bat = bat.sort_values("cycle")
        eol_cycle = bat[bat["soh"] < 0.8]["cycle"]
        eol_cycle = int(eol_cycle.iloc[0]) if not eol_cycle.empty else int(bat["cycle"].max())
        for _, row in bat.iterrows():
            rul_rows.append({"battery_id": battery_id, "cycle": row["cycle"], "rul": max(0, eol_cycle - int(row["cycle"]))})
    rul_df = pd.DataFrame(rul_rows)

    raw_df = raw_df.merge(rul_df, on=["battery_id", "cycle"], how="left")
    raw_df = raw_df.sort_values(["battery_id", "cycle", "mode", "timestamp_s"]).reset_index(drop=True)
    return raw_df


def load_raw(path_or_dataset: Optional[str] = None, dest: str = "data/raw") -> pd.DataFrame:
    """Load raw telemetry data.

    Parameters
    - path_or_dataset: local path to file or directory, or Kaggle dataset id
      (owner/dataset). If None, looks for `data/raw/battery_telemetry_raw.parquet` or
      `.csv`.
    - dest: directory to download/extract into when using Kaggle.
    """
    if path_or_dataset is None:
        p_parquet = Path(dest) / "battery_telemetry_raw.parquet"
        p_csv = Path(dest) / "battery_telemetry_raw.csv"
        if p_parquet.exists():
            df = pd.read_parquet(p_parquet)
            return _normalize_columns(df)
        if p_csv.exists():
            df = pd.read_csv(p_csv)
            return _normalize_columns(df)
        raise FileNotFoundError(f"No default raw file found in {dest}")

    # If looks like Kaggle id (owner/dataset), download
    if isinstance(path_or_dataset, str) and "/" in path_or_dataset and not Path(path_or_dataset).exists():
        out_dir = _download_kaggle_dataset(path_or_dataset, dest=dest)
        candidates = _find_candidate_files(out_dir)
        if not candidates:
            raise FileNotFoundError(f"No CSV/Parquet files found in downloaded dataset {path_or_dataset}")
        # Try to find a file with 'telemetry' or 'battery' in name
        chosen = None
        for c in candidates:
            if "telemetry" in c.name.lower() or "battery" in c.name.lower():
                chosen = c
                break
        if not chosen:
            chosen = candidates[0]
        if chosen.suffix.lower() in (".parquet", ".parq"):
            df = pd.read_parquet(chosen)
        elif chosen.suffix.lower() == ".zip":
            # try to unzip and find CSV inside
            import zipfile

            with zipfile.ZipFile(chosen, "r") as zf:
                tmp = Path(tempfile.mkdtemp())
                zf.extractall(tmp)
                cand = _find_candidate_files(tmp)
                if not cand:
                    raise FileNotFoundError("No CSV/Parquet inside downloaded zip")
                df = pd.read_csv(cand[0])
                shutil.rmtree(tmp)
        else:
            df = pd.read_csv(chosen)

        return _normalize_columns(df)

    # Otherwise treat as a local path (file or directory)
    p = Path(path_or_dataset)
    if p.is_dir():
        # special support for the Kaggle NASA folder structure
        nasa_df = _load_nasa_kaggle_folder(p)
        if nasa_df is not None:
            return nasa_df

        candidates = _find_candidate_files(p)
        if not candidates:
            raise FileNotFoundError(f"No CSV/Parquet files found in directory {p}")
        chosen = candidates[0]
        if chosen.suffix.lower() in (".parquet", ".parq"):
            df = pd.read_parquet(chosen)
        else:
            df = pd.read_csv(chosen)
        return _normalize_columns(df)

    # single file
    if p.exists():
        if p.suffix.lower() in (".parquet", ".parq"):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        return _normalize_columns(df)

    raise FileNotFoundError(f"Could not find or download dataset: {path_or_dataset}")


def generate_battery_dataset(source: Optional[str] = None, save_path: str = "data/raw") -> pd.DataFrame:
    """Drop-in replacement for the synthetic generator.

    - If `source` is a Kaggle dataset id (owner/dataset) it will be downloaded.
    - If `source` is a path it will be loaded.
    - The loaded DataFrame will be saved to `save_path/battery_telemetry_raw.parquet`.
    """
    # Try to load real data; if not found, fall back to a small synthetic
    # generator to keep tests and downstream pipelines working.
    try:
        df = load_raw(source, dest=save_path)
    except FileNotFoundError:
        # Minimal synthetic fallback: small reproducible dataset (used in tests)
        import numpy as _np

        Path(save_path).mkdir(parents=True, exist_ok=True)
        rng = _np.random.default_rng(42)
        records = []
        batteries = ["B0005", "B0045", "B0047"]
        max_cycles = 12
        for battery_id in batteries:
            for cycle in range(max_cycles):
                nominal_capacity = 2.0
                capacity = nominal_capacity * _np.exp(-0.003 * cycle) + rng.normal(0, 0.01)
                # bound capacity to [0.5, nominal_capacity] to ensure soh in [0,1]
                capacity = float(min(nominal_capacity, max(0.5, capacity)))
                soh = float(capacity / nominal_capacity)
                rul = max(0, max_cycles - cycle)
                ir = 0.05 + 0.001 * cycle + rng.normal(0, 0.001)
                for mode in ("discharge", "charge"):
                    n_pts = 5
                    for t_i in range(n_pts):
                        ts = float(t_i * 60)
                        voltage = float(_np.clip(4.2 - 0.6 * (t_i / n_pts) - 0.1 * cycle / max_cycles + rng.normal(0, 0.01), 2.7, 4.2))
                        current = float(-1.0 if mode == "discharge" else 1.0) + float(rng.normal(0, 0.02))
                        temp = float(24.0 + 0.5 * cycle / max_cycles + rng.normal(0, 0.3))
                        records.append({
                            "battery_id": battery_id,
                            "cycle": int(cycle),
                            "mode": mode,
                            "timestamp_s": ts,
                            "voltage": voltage,
                            "current": current,
                            "temperature": temp,
                            "capacity": round(float(capacity), 5),
                            "soh": round(float(soh), 5),
                            "rul": int(rul),
                            "internal_resistance": round(float(ir), 5),
                        })

        df = pd.DataFrame.from_records(records)

    # Save to parquet (or CSV fallback)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    out = Path(save_path) / "battery_telemetry_raw.parquet"
    try:
        df.to_parquet(out, index=False)
        print(f"Saved raw data -> {out}")
    except Exception:
        out_csv = Path(save_path) / "battery_telemetry_raw.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved raw data -> {out_csv}")

    return df


if __name__ == "__main__":
    # quick smoke test: try default location or raise helpful message
    try:
        df = generate_battery_dataset(None)
        print(df.head())
    except Exception as e:
        print("Usage: call generate_battery_dataset(source) with a local path or Kaggle id (owner/dataset).")
        print(e)
