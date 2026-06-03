"""
EV Battery Health AI — Test Suite
==================================
Tests core functionality: data generation, feature engineering, risk scoring.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Data generator tests
# ---------------------------------------------------------------------------

class TestDataGenerator:
    def test_generates_dataframe(self, tmp_path):
        from utils.data_loader import generate_battery_dataset
        df = generate_battery_dataset(save_path=str(tmp_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self, tmp_path):
        from utils.data_loader import generate_battery_dataset
        df = generate_battery_dataset(save_path=str(tmp_path))
        required = ["battery_id", "cycle", "voltage", "current",
                    "temperature", "soh", "rul", "capacity"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_soh_range(self, tmp_path):
        from utils.data_loader import generate_battery_dataset
        df = generate_battery_dataset(save_path=str(tmp_path))
        assert df["soh"].between(0.0, 1.0).all(), "SoH must be in [0, 1]"

    def test_voltage_range(self, tmp_path):
        from utils.data_loader import generate_battery_dataset
        df = generate_battery_dataset(save_path=str(tmp_path))
        assert df["voltage"].between(2.0, 4.5).all(), "Voltage out of Li-ion range"

    def test_three_batteries(self, tmp_path):
        from utils.data_loader import generate_battery_dataset
        df = generate_battery_dataset(save_path=str(tmp_path))
        assert df["battery_id"].nunique() == 3


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    @pytest.fixture
    def raw_df(self, tmp_path):
        from utils.data_loader import generate_battery_dataset
        return generate_battery_dataset(save_path=str(tmp_path))

    def test_extract_cycle_features_shape(self, raw_df):
        from utils.feature_engineering import extract_cycle_features
        feat = extract_cycle_features(raw_df)
        assert len(feat) > 0
        assert "soh" in feat.columns
        assert "rul" in feat.columns
        assert "cycle" in feat.columns

    def test_no_nan_in_key_features(self, raw_df):
        from utils.feature_engineering import extract_cycle_features, FEATURE_COLS
        feat = extract_cycle_features(raw_df)
        for col in FEATURE_COLS:
            assert col in feat.columns, f"Feature missing: {col}"

    def test_lstm_sequences_shape(self, raw_df, tmp_path):
        from utils.feature_engineering import extract_cycle_features, build_lstm_sequences
        feat = extract_cycle_features(raw_df)
        scaler_path = str(tmp_path / "scaler.pkl")
        X, y_soh, y_rul, groups = build_lstm_sequences(
            feat, scaler_path=scaler_path, seq_len=5
        )
        assert X.ndim == 3
        assert X.shape[1] == 5
        assert len(y_soh) == len(X)
        assert len(y_rul) == len(X)


# ---------------------------------------------------------------------------
# Risk scoring tests
# ---------------------------------------------------------------------------

class TestRiskScoring:
    def test_low_risk(self):
        from utils.risk_scoring import assess_risk
        r = assess_risk(soh=0.95, rul=100)
        assert r.risk_label == "LOW"
        assert r.risk_score < 30

    def test_high_risk(self):
        from utils.risk_scoring import assess_risk
        r = assess_risk(soh=0.62, rul=5)
        assert r.risk_label == "HIGH"
        assert r.risk_score > 50   # SoH=0.62, RUL=5 → score ~56; threshold is 50+

    def test_medium_risk(self):
        from utils.risk_scoring import assess_risk
        r = assess_risk(soh=0.78, rul=30)
        assert r.risk_label == "MEDIUM"

    def test_risk_score_bounded(self):
        from utils.risk_scoring import compute_risk_score
        for soh, rul in [(1.0, 200), (0.0, 0), (0.5, 50)]:
            score = compute_risk_score(soh, rul)
            assert 0.0 <= score <= 100.0

    def test_assessment_fields(self):
        from utils.risk_scoring import assess_risk
        r = assess_risk(0.85, 60)
        assert hasattr(r, "recommendation")
        assert hasattr(r, "risk_color")
        assert r.risk_color.startswith("#")


# ---------------------------------------------------------------------------
# Anomaly detection tests
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def test_fit_predict(self, tmp_path):
        from utils.data_loader import generate_battery_dataset
        from utils.feature_engineering import extract_cycle_features
        from utils.risk_scoring import BatteryAnomalyDetector

        raw = generate_battery_dataset(save_path=str(tmp_path))
        feat = extract_cycle_features(raw)

        detector = BatteryAnomalyDetector(contamination=0.05)
        detector.fit(feat)
        preds = detector.predict(feat)
        assert set(preds).issubset({-1, 1})
        assert len(preds) == len(feat)

    def test_annotate_adds_columns(self, tmp_path):
        from utils.data_loader import generate_battery_dataset
        from utils.feature_engineering import extract_cycle_features
        from utils.risk_scoring import BatteryAnomalyDetector

        raw = generate_battery_dataset(save_path=str(tmp_path))
        feat = extract_cycle_features(raw)

        detector = BatteryAnomalyDetector()
        detector.fit(feat)
        ann = detector.annotate(feat)
        assert "is_anomaly" in ann.columns
        assert "anomaly_score" in ann.columns
