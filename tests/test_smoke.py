"""
Smoke tests — verify pipeline không crash với 30 ngày data.
Không test model accuracy, chỉ test data flow và shapes.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def features_df():
    """Load cached features — yêu cầu features.parquet tồn tại."""
    path = ROOT / "data" / "features" / "features.parquet"
    assert path.exists(), f"features.parquet không tồn tại tại {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def trades_df():
    path = ROOT / "backtest" / "trades.csv"
    assert path.exists(), f"trades.csv không tồn tại tại {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def signals_df():
    """Load signal CSV gần nhất."""
    sig_dir = ROOT / "signals"
    files = sorted(sig_dir.glob("*.csv"))
    assert files, "Không tìm thấy file signal CSV nào trong signals/"
    return pd.read_csv(files[-1])


# ── Feature tests ─────────────────────────────────────────────────────────────

class TestFeatures:

    def test_features_file_exists(self):
        assert (ROOT / "data" / "features" / "features.parquet").exists()

    def test_features_has_30_tickers(self, features_df):
        assert features_df["ticker"].nunique() == 30

    def test_features_date_range(self, features_df):
        start = pd.to_datetime(features_df.index.min())
        assert start.year >= 2020, f"Dữ liệu bắt đầu quá muộn: {start}"

    def test_no_future_leakage_in_target(self, features_df):
        """future_return (forward-looking) không được nằm trong FEATURE_COLS.
        Note: ret_5d là momentum lịch sử (backward), KHÔNG phải leakage.
        """
        from src.features import FEATURE_COLS
        assert "future_return" not in FEATURE_COLS, "future_return (forward-looking) nằm trong FEATURE_COLS — LEAKAGE!"
        assert "target" not in FEATURE_COLS, "target nằm trong FEATURE_COLS — LEAKAGE!"

    def test_feature_cols_count(self, features_df):
        from src.features import FEATURE_COLS
        missing = [c for c in FEATURE_COLS if c not in features_df.columns]
        assert not missing, f"Features thiếu trong DataFrame: {missing}"

    def test_no_all_nan_feature(self, features_df):
        from src.features import FEATURE_COLS
        for col in FEATURE_COLS:
            if col in features_df.columns:
                pct_nan = features_df[col].isna().mean()
                assert pct_nan < 0.5, f"Feature '{col}' có {pct_nan:.1%} NaN — quá cao"

    def test_close_price_scale(self, features_df):
        """Giá close trong features phải là đơn vị 1000 VND (10–1000 range cho VN30)."""
        close_mean = features_df["close"].mean()
        assert 10 < close_mean < 2000, (
            f"Close price có vẻ sai đơn vị: mean={close_mean:.1f}. "
            "vnstock trả về nghìn VND (10-1000 range)."
        )


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModel:

    def test_model_files_exist(self):
        assert (ROOT / "models" / "xgb_model.json").exists()
        assert (ROOT / "models" / "lgb_model.txt").exists()
        assert (ROOT / "models" / "label_classes.npy").exists()

    def _load_with_wrapper(self):
        """Load models với LGB wrapper giống như signal_generator.py."""
        import xgboost as xgb
        import lightgbm as lgb
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        from src.model import ensemble_predict

        xgb_m = xgb.XGBClassifier()
        xgb_m.load_model(str(ROOT / "models" / "xgb_model.json"))
        lgb_booster = lgb.Booster(model_file=str(ROOT / "models" / "lgb_model.txt"))
        classes = np.load(ROOT / "models" / "label_classes.npy")
        le = LabelEncoder()
        le.classes_ = classes

        class _LGBWrapper:
            def predict_proba(self, X):
                return lgb_booster.predict(X)

        return xgb_m, _LGBWrapper(), le

    def test_ensemble_predict_shape(self, features_df):
        """ensemble_predict trả về đúng số samples."""
        from src.features import FEATURE_COLS
        from src.model import ensemble_predict

        xgb_m, lgb_m, le = self._load_with_wrapper()
        sample = features_df[FEATURE_COLS].dropna().head(50)
        if sample.empty:
            pytest.skip("Không đủ sample không-NaN")

        preds, proba = ensemble_predict(xgb_m, lgb_m, le, sample.values)
        assert len(preds) == len(sample)
        assert proba.shape == (len(sample), len(le.classes_))

    def test_ensemble_predict_valid_labels(self, features_df):
        """Predictions chỉ gồm nhãn hợp lệ."""
        from src.features import FEATURE_COLS
        from src.model import ensemble_predict

        xgb_m, lgb_m, le = self._load_with_wrapper()
        sample = features_df[FEATURE_COLS].dropna().head(50)
        if sample.empty:
            pytest.skip("Không đủ sample")

        preds, _ = ensemble_predict(xgb_m, lgb_m, le, sample.values)
        valid = set(le.classes_)
        assert all(p in valid for p in preds), f"Prediction có nhãn lạ: {set(preds) - valid}"

    def test_confidence_range(self, features_df):
        """Confidence phải nằm trong [0, 1]."""
        from src.features import FEATURE_COLS
        from src.model import ensemble_predict

        xgb_m, lgb_m, le = self._load_with_wrapper()
        sample = features_df[FEATURE_COLS].dropna().head(100)
        if sample.empty:
            pytest.skip("Không đủ sample")

        _, proba = ensemble_predict(xgb_m, lgb_m, le, sample.values)
        conf = proba.max(axis=1)
        assert conf.min() >= 0.0
        assert conf.max() <= 1.0


# ── Signals tests ─────────────────────────────────────────────────────────────

class TestSignals:

    def test_signals_has_required_cols(self, signals_df):
        required = ["ticker", "date", "signal", "confidence", "close"]
        missing = [c for c in required if c not in signals_df.columns]
        assert not missing, f"Signals CSV thiếu cột: {missing}"

    def test_signals_has_30_rows(self, signals_df):
        assert len(signals_df) == 30, f"Signals không đủ 30 mã: {len(signals_df)}"

    def test_signals_only_buy_hold(self, signals_df):
        invalid = signals_df[~signals_df["signal"].isin(["BUY", "HOLD"])]
        assert invalid.empty, f"Có signal không hợp lệ: {invalid['signal'].unique()}"

    def test_signals_confidence_range(self, signals_df):
        assert signals_df["confidence"].between(0, 1).all()

    def test_signals_price_correct_scale(self, signals_df):
        """Giá trong signals CSV đã nhân 1000 → range 10,000–2,000,000."""
        close_mean = signals_df["close"].mean()
        assert 10_000 < close_mean < 2_000_000, (
            f"Giá close trong signals CSV có vẻ sai: mean={close_mean:,.0f}. "
            "Phải là giá VND thực (nhân 1000 so với vnstock)."
        )


# ── Backtest tests ────────────────────────────────────────────────────────────

class TestBacktest:

    def test_trades_file_exists(self):
        assert (ROOT / "backtest" / "trades.csv").exists()

    def test_trades_required_cols(self, trades_df):
        required = ["date", "ticker", "direction", "entry", "exit", "confidence", "net_return", "pnl"]
        missing = [c for c in required if c not in trades_df.columns]
        assert not missing, f"Trades CSV thiếu cột: {missing}"

    def test_no_zero_entry_price(self, trades_df):
        zeros = (trades_df["entry"] == 0).sum()
        assert zeros == 0, f"{zeros} lệnh có entry price = 0"

    def test_net_return_reasonable(self, trades_df):
        """Net return không được có outlier cực đoan (>100% hay < -100% trong 5 ngày)."""
        extreme = trades_df[trades_df["net_return"].abs() > 50]
        assert extreme.empty, f"{len(extreme)} lệnh có |net_return| > 50% — kiểm tra lại"

    def test_pnl_consistent_with_return(self, trades_df):
        """pnl ≈ 10M × net_return / 100 (trong phạm vi làm tròn)."""
        expected_pnl = 10_000_000 * trades_df["net_return"] / 100
        diff = (trades_df["pnl"] - expected_pnl).abs()
        assert (diff < 1000).all(), "pnl không nhất quán với net_return"


# ── Tracker tests ─────────────────────────────────────────────────────────────

class TestTracker:

    def test_predictions_file_exists(self):
        assert (ROOT / "data" / "tracker" / "predictions.parquet").exists()

    def test_latest_report_exists(self):
        assert (ROOT / "data" / "tracker" / "latest_report.json").exists()

    def test_tracker_score_range(self):
        import json
        report = json.loads((ROOT / "data" / "tracker" / "latest_report.json").read_text())
        # Score nằm trong prediction_score.score
        pred_score = report.get("prediction_score", {})
        score = pred_score.get("score")
        assert score is not None, f"Report không có 'prediction_score.score'. Keys: {list(report.keys())}"
        assert 0 <= score <= 100, f"Score ngoài range [0,100]: {score}"
