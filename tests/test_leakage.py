"""
[P1.1] test_leakage.py — Permutation test để detect data leakage.

Methodology:
- Lấy 1 walk-forward split (split đầu tiên)
- Shuffle y_train ngẫu nhiên → nếu model vẫn predict tốt trên y_test thực
  thì có leakage (X_train đã chứa info về y)
- Pass criteria: AUC trung bình ∈ [0.45, 0.58] (không khác random nhiều)
- Fail criteria: AUC trung bình > 0.60 → leakage nghiêm trọng

Dùng XGB với n_estimators=50 để chạy nhanh (không cần full model).
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

ROOT = Path(__file__).parent.parent
N_PERMUTATIONS = 10
FAST_N_EST = 50  # Giảm estimators để test nhanh


def load_first_split():
    """Load features và trả về (X_train, y_train, X_test, y_test) của split đầu tiên."""
    from src.features import FEATURE_COLS
    from src.macro import MACRO_FEATURE_COLS
    from src.regime import REGIME_FEATURE_COLS
    from src.model import load_features, walk_forward_splits

    df = load_features()

    # Fill new P3 features with 0 if features.parquet is pre-retrain
    new_cols = MACRO_FEATURE_COLS + REGIME_FEATURE_COLS
    for col in new_cols:
        if col not in df.columns:
            df[col] = 0.0

    splits = walk_forward_splits(df)
    assert splits, "Không có split nào — kiểm tra data"

    train_mask, test_mask = splits[0]
    train = df[train_mask].dropna(subset=FEATURE_COLS + ["target"])
    test = df[test_mask].dropna(subset=FEATURE_COLS + ["target"])

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["target"].values

    return X_train, y_train, X_test, y_test


def train_fast_xgb(X_train, y_enc):
    model = xgb.XGBClassifier(
        n_estimators=FAST_N_EST,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_enc)
    return model


def compute_macro_auc(model, le, X_test, y_test):
    """Tính macro-averaged OvR AUC."""
    proba = model.predict_proba(X_test)
    y_bin = np.zeros((len(y_test), len(le.classes_)))
    for i, cls in enumerate(le.classes_):
        y_bin[:, i] = (y_test == cls).astype(int)
    try:
        return roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    except ValueError:
        return 0.5  # nếu chỉ có 1 class trong test


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def split_data():
    return load_first_split()


class TestPermutationLeakage:

    def test_real_model_above_random(self, split_data):
        """Model thực phải tốt hơn random (AUC > 0.5) — sanity check."""
        X_train, y_train, X_test, y_test = split_data
        le = LabelEncoder()
        y_enc = le.fit_transform(y_train)

        model = train_fast_xgb(X_train, y_enc)
        real_auc = compute_macro_auc(model, le, X_test, y_test)

        print(f"\nReal model AUC: {real_auc:.4f}")
        assert real_auc > 0.50, (
            f"Model thực AUC={real_auc:.3f} ≤ 0.50 — model không học được gì. "
            "Kiểm tra data pipeline."
        )

    def test_permutation_auc_near_random(self, split_data):
        """Khi shuffle y_train, model không nên predict tốt hơn random nhiều.

        Pass: AUC trung bình < 0.58
        Fail: AUC trung bình > 0.60 → có leakage
        """
        X_train, y_train, X_test, y_test = split_data
        le = LabelEncoder()
        y_enc_real = le.fit_transform(y_train)

        perm_aucs = []
        rng = np.random.default_rng(seed=0)

        for i in range(N_PERMUTATIONS):
            # Shuffle nhãn
            y_shuffled = rng.permutation(y_enc_real)
            model = train_fast_xgb(X_train, y_shuffled)
            auc = compute_macro_auc(model, le, X_test, y_test)
            perm_aucs.append(auc)

        mean_auc = np.mean(perm_aucs)
        std_auc = np.std(perm_aucs)
        print(f"\nPermutation AUC: mean={mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Individual: {[f'{a:.3f}' for a in perm_aucs]}")

        assert mean_auc < 0.60, (
            f"LEAKAGE DETECTED: AUC trung bình với shuffled labels = {mean_auc:.3f} > 0.60. "
            "Model học được signal từ X ngay cả khi y bị shuffle."
        )

    def test_permutation_auc_vs_real_gap(self, split_data):
        """Model thực phải tốt hơn permutation đáng kể.

        Nếu gap quá nhỏ (<0.02) → model gần như không học được.
        """
        X_train, y_train, X_test, y_test = split_data
        le = LabelEncoder()
        y_enc = le.fit_transform(y_train)

        # Real model
        real_model = train_fast_xgb(X_train, y_enc)
        real_auc = compute_macro_auc(real_model, le, X_test, y_test)

        # Permutation (1 lần, nhanh)
        rng = np.random.default_rng(seed=99)
        y_shuf = rng.permutation(y_enc)
        perm_model = train_fast_xgb(X_train, y_shuf)
        perm_auc = compute_macro_auc(perm_model, le, X_test, y_test)

        gap = real_auc - perm_auc
        print(f"\nReal AUC={real_auc:.4f}, Perm AUC={perm_auc:.4f}, Gap={gap:.4f}")

        assert gap > 0.01, (
            f"Gap quá nhỏ ({gap:.3f}): model thực không khác permutation nhiều. "
            "Có thể features không có predictive power, hoặc leakage khiến cả 2 đều cao."
        )


class TestFeatureLeakage:
    """Kiểm tra các feature cụ thể có khả năng bị leakage."""

    def test_future_return_not_in_features(self):
        from src.features import FEATURE_COLS
        assert "future_return" not in FEATURE_COLS, "future_return (forward) trong FEATURE_COLS!"
        assert "target" not in FEATURE_COLS, "target trong FEATURE_COLS!"

    def test_ret5d_is_backward_looking(self):
        """ret_5d phải là lợi nhuận LÙI 5 ngày, không phải TIẾN."""
        import pandas as pd
        from src.features import add_indicators

        # Tạo mock data — cần ≥50 ngày để MACD(26) và BB(20) tính được
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        df = pd.DataFrame({
            "open": 100.0, "high": 105.0, "low": 95.0,
            "close": [100 + i * 0.5 for i in range(60)],
            "volume": 1_000_000,
        }, index=dates)

        result = add_indicators(df)
        # ret_5d tại ngày t = (close_t / close_{t-5}) - 1 — không phụ thuộc tương lai
        # Kiểm tra tại ngày thứ 40: ret_5d = (close[40] / close[35]) - 1
        t = result.index[40]
        expected = (result.loc[t, "close"] / result["close"].iloc[35]) - 1
        actual = result.loc[t, "ret_5d"]
        assert abs(actual - expected) < 1e-6, (
            f"ret_5d không đúng: expected={expected:.4f}, actual={actual:.4f}. "
            "Có thể đang dùng forward shift."
        )

    def test_no_scaler_fit_on_full_data(self):
        """Không có StandardScaler hay MinMaxScaler fit trên full dataset.
        Tree models (XGB/LGB) không cần scaling → không có leakage qua scaler.
        """
        import inspect
        import src.model as model_module
        import src.features as features_module

        src_code = inspect.getsource(model_module) + inspect.getsource(features_module)
        # Nếu có StandardScaler, phải đảm bảo chỉ fit trên train
        if "StandardScaler" in src_code or "MinMaxScaler" in src_code:
            # Nếu có scaler, phải fit trên train_mask, không phải full data
            assert "fit_transform(X_train" in src_code or "fit(X_train" in src_code, (
                "Phát hiện Scaler nhưng không thấy fit trên X_train — có thể leakage!"
            )
        # Không có scaler → pass (tree models không cần)
