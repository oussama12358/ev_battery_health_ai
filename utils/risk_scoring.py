"""
EV Battery Health AI — Risk Scoring & Anomaly Detection
========================================================
Translates raw SoH / RUL predictions into an actionable risk tier
and flags statistical anomalies in battery telemetry.

Risk framework (inspired by BMS industry standards):
  - LOW    : SoH ≥ 85%  and RUL ≥ 50 cycles
  - MEDIUM : 70% ≤ SoH < 85%  or  20 ≤ RUL < 50
  - HIGH   : SoH < 70%  or  RUL < 20
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path


# ---------------------------------------------------------------------------
# Risk Score dataclass
# ---------------------------------------------------------------------------

@dataclass
class RiskAssessment:
    soh:          float
    rul:          int
    risk_label:   str            # "LOW" | "MEDIUM" | "HIGH"
    risk_score:   float          # 0–100 continuous score
    risk_color:   str            # hex for dashboard
    recommendation: str

    def to_dict(self) -> dict:
        return self.__dict__


# ---------------------------------------------------------------------------
# Risk scoring logic
# ---------------------------------------------------------------------------

SOH_THRESHOLDS = {"low": 0.85, "medium": 0.70}
RUL_THRESHOLDS = {"low": 50,   "medium": 20}


def compute_risk_score(soh: float, rul: int) -> float:
    """
    Continuous risk score in [0, 100].
    Combines SoH degradation and RUL proximity to EOL.
    Higher score = more risk.
    """
    # SoH component: 0 at perfect health, 100 at SoH=0
    soh_risk = (1.0 - np.clip(soh, 0, 1)) * 100.0

    # RUL component: 100 at RUL=0, diminishes logarithmically
    rul_risk = 100.0 * np.exp(-rul / 30.0)

    # Weighted combination (SoH is primary signal)
    score = 0.6 * soh_risk + 0.4 * rul_risk
    return float(np.clip(score, 0.0, 100.0))


def assess_risk(soh: float, rul: int) -> RiskAssessment:
    """
    Map SoH + RUL → RiskAssessment object.
    """
    score = compute_risk_score(soh, rul)

    if soh >= SOH_THRESHOLDS["low"] and rul >= RUL_THRESHOLDS["low"]:
        label = "LOW"
        color = "#27ae60"
        rec   = "Battery is healthy. Continue normal operation."
    elif soh < SOH_THRESHOLDS["medium"] or rul < RUL_THRESHOLDS["medium"]:
        label = "HIGH"
        color = "#e74c3c"
        rec   = (
            "⚠️ Critical degradation detected. Schedule immediate replacement "
            "and avoid fast-charging cycles."
        )
    else:
        label = "MEDIUM"
        color = "#f39c12"
        rec   = (
            "Moderate degradation. Reduce peak discharge rates and monitor "
            "temperature closely."
        )

    return RiskAssessment(
        soh=round(soh, 4),
        rul=int(rul),
        risk_label=label,
        risk_score=round(score, 2),
        risk_color=color,
        recommendation=rec,
    )


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class BatteryAnomalyDetector:
    """
    Isolation Forest-based anomaly detector trained on cycle-level features.

    Detects:
      - Sudden voltage drops
      - Abnormal temperature spikes
      - Unusual current patterns
      - Rapid capacity fade acceleration
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    ANOMALY_FEATURES = [
        "dis_v_mean", "dis_v_std", "dis_v_min",
        "dis_t_mean", "dis_t_max",
        "ir_mean", "dis_charge_throughput_ah",
        "v_drop_rate", "delta_ir_mean",
    ]

    def fit(self, feat_df: pd.DataFrame) -> "BatteryAnomalyDetector":
        cols = [c for c in self.ANOMALY_FEATURES if c in feat_df.columns]
        X = feat_df[cols].fillna(0).values
        self.model.fit(X)
        self._fitted = True
        print(f"[AnomalyDetector] Fitted on {len(X):,} samples, "
              f"contamination={self.model.contamination}")
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Returns array of -1 (anomaly) or 1 (normal)."""
        assert self._fitted, "Call fit() first."
        cols = [c for c in self.ANOMALY_FEATURES if c in feat_df.columns]
        X = feat_df[cols].fillna(0).values
        return self.model.predict(X)

    def score_samples(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Anomaly scores — more negative = more anomalous."""
        assert self._fitted, "Call fit() first."
        cols = [c for c in self.ANOMALY_FEATURES if c in feat_df.columns]
        X = feat_df[cols].fillna(0).values
        return self.model.score_samples(X)

    def annotate(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly flag and score columns to feat_df copy."""
        df = feat_df.copy()
        df["anomaly_flag"]  = self.predict(feat_df)
        df["anomaly_score"] = self.score_samples(feat_df)
        df["is_anomaly"]    = df["anomaly_flag"] == -1
        return df

    def save(self, path: str = "models/saved/anomaly_detector.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[AnomalyDetector] Saved -> {path}")

    @classmethod
    def load(cls, path: str = "models/saved/anomaly_detector.pkl") -> "BatteryAnomalyDetector":
        detector = joblib.load(path)
        print(f"[AnomalyDetector] Loaded from {path}")
        return detector


# ---------------------------------------------------------------------------
# Degradation rate analyser
# ---------------------------------------------------------------------------

def compute_degradation_rate(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-battery rolling degradation rate (SoH drop per 10 cycles).
    Useful for early warning and fleet monitoring.
    """
    out = []
    for bat_id, grp in feat_df.groupby("battery_id"):
        grp = grp.sort_values("cycle").copy()
        grp["soh_10cyc_delta"] = grp["soh"].diff(10)   # negative = degradation
        grp["degradation_rate"] = -grp["soh_10cyc_delta"].fillna(0)
        grp["battery_id"] = bat_id
        out.append(grp)
    return pd.concat(out).reset_index(drop=True)


if __name__ == "__main__":
    # Quick sanity check
    examples = [
        (0.95, 120),   # healthy
        (0.78, 30),    # medium
        (0.65, 10),    # high risk
    ]
    for soh, rul in examples:
        r = assess_risk(soh, rul)
        print(f"SoH={soh:.0%}  RUL={rul}  ->  {r.risk_label} "
              f"(score={r.risk_score:.1f})  |  {r.recommendation[:60]}")
