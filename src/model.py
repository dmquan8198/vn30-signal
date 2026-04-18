"""
model.py — Train XGBoost + LightGBM ensemble với walk-forward validation
Ensemble: average probability từ 2 models → prediction cuối
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from src.features import FEATURE_COLS, FEATURES_DIR

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = 3
TEST_MONTHS = 6


def load_features() -> pd.DataFrame:
    path = FEATURES_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError("features.parquet not found — run src/features.py first")
    return pd.read_parquet(path).sort_index()


def walk_forward_splits(df: pd.DataFrame):
    dates = df.index.unique().sort_values()
    start = dates[0]
    end = dates[-1]
    split_start = start + pd.DateOffset(years=TRAIN_YEARS)
    splits = []
    current = split_start
    while current + pd.DateOffset(months=TEST_MONTHS) <= end:
        train_end = current
        test_end = current + pd.DateOffset(months=TEST_MONTHS)
        train_mask = (df.index >= start) & (df.index < train_end)
        test_mask = (df.index >= train_end) & (df.index < test_end)
        if train_mask.sum() > 100 and test_mask.sum() > 20:
            splits.append((train_mask, test_mask))
        current += pd.DateOffset(months=TEST_MONTHS)
    return splits


def train_xgb(X_train, y_enc) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.05,
        reg_lambda=1.0,
        min_child_weight=3,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_enc)
    return model


def train_lgb(X_train, y_enc) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.04,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.05,
        reg_lambda=1.0,
        min_child_samples=15,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_enc)
    return model


def train_rf(X_train, y_enc) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_enc)
    return model


def train_ensemble(X_train, y_train):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    xgb_model = train_xgb(X_train, y_enc)
    lgb_model = train_lgb(X_train, y_enc)
    rf_model = train_rf(X_train, y_enc)
    return xgb_model, lgb_model, rf_model, le


def ensemble_predict(xgb_model, lgb_model, le, X_test, rf_model=None):
    """
    3-model consensus ensemble (XGB + LGB + RF):
    - Nếu majority (≥2/3) đồng ý → dùng prediction đó, confidence = avg proba
    - Nếu tất cả không đồng ý → HOLD
    RF là optional để duy trì backward compat với saved models.
    """
    proba_xgb = xgb_model.predict_proba(X_test)
    proba_lgb = lgb_model.predict_proba(X_test)

    pred_enc_xgb = proba_xgb.argmax(axis=1)
    pred_enc_lgb = proba_lgb.argmax(axis=1)

    hold_enc = le.transform([0])[0]

    if rf_model is not None:
        proba_rf = rf_model.predict_proba(X_test)
        pred_enc_rf = proba_rf.argmax(axis=1)
        proba_avg = (proba_xgb + proba_lgb + proba_rf) / 3

        # Majority vote: at least 2 of 3 must agree
        consensus = np.full(len(pred_enc_xgb), hold_enc, dtype=int)
        for i in range(len(pred_enc_xgb)):
            votes = [pred_enc_xgb[i], pred_enc_lgb[i], pred_enc_rf[i]]
            # Find majority
            from collections import Counter
            count = Counter(votes)
            top_label, top_count = count.most_common(1)[0]
            if top_count >= 2:
                consensus[i] = top_label
    else:
        # Fallback: original 2-model consensus
        proba_avg = (proba_xgb + proba_lgb) / 2
        consensus = np.where(pred_enc_xgb == pred_enc_lgb, pred_enc_xgb, hold_enc)

    preds = le.inverse_transform(consensus)
    return preds, proba_avg


def evaluate(preds, proba_avg, le, y_test):
    report = classification_report(
        y_test, preds,
        labels=[-1, 0, 1],
        target_names=["Sell", "Hold", "Buy"],
        output_dict=True,
        zero_division=0,
    )
    return report


def run_walk_forward():
    print("Loading features...")
    df = load_features()
    splits = walk_forward_splits(df)
    print(f"Walk-forward splits: {len(splits)}\n")

    all_reports_xgb, all_reports_lgb, all_reports_ens = [], [], []
    all_preds = []

    for i, (train_mask, test_mask) in enumerate(splits):
        train = df[train_mask]
        test = df[test_mask]

        X_train = train[FEATURE_COLS].values
        y_train = train["target"].values
        X_test = test[FEATURE_COLS].values
        y_test = test["target"].values

        xgb_m, lgb_m, rf_m, le = train_ensemble(X_train, y_train)

        # Individual model predictions
        y_enc = le.transform(y_test)
        preds_xgb = le.inverse_transform(xgb_m.predict(X_test))
        preds_lgb = le.inverse_transform(lgb_m.predict(X_test))

        # Ensemble (3-model)
        preds_ens, proba_avg = ensemble_predict(xgb_m, lgb_m, le, X_test, rf_model=rf_m)

        r_xgb = evaluate(preds_xgb, None, le, y_test)
        r_lgb = evaluate(preds_lgb, None, le, y_test)
        r_ens = evaluate(preds_ens, proba_avg, le, y_test)

        period = f"{test.index.min().date()} → {test.index.max().date()}"
        print(
            f"Split {i+1:02d} | {period} | "
            f"XGB Buy:{r_xgb['Buy']['precision']:.2f} "
            f"LGB Buy:{r_lgb['Buy']['precision']:.2f} "
            f"ENS Buy:{r_ens['Buy']['precision']:.2f} | "
            f"XGB Sell:{r_xgb['Sell']['precision']:.2f} "
            f"LGB Sell:{r_lgb['Sell']['precision']:.2f} "
            f"ENS Sell:{r_ens['Sell']['precision']:.2f}"
        )

        all_reports_xgb.append(r_xgb)
        all_reports_lgb.append(r_lgb)
        all_reports_ens.append(r_ens)

        pred_df = test[["ticker", "future_return"]].copy()
        pred_df["prediction"] = preds_ens
        pred_df["split"] = i
        all_preds.append(pred_df)

    # Final model: train trên toàn bộ data trừ 6 tháng cuối
    cutoff = df.index.max() - pd.DateOffset(months=TEST_MONTHS)
    final_train = df[df.index < cutoff]
    X_final = final_train[FEATURE_COLS].values
    y_final = final_train["target"].values

    import pickle
    final_xgb, final_lgb, final_rf, final_le = train_ensemble(X_final, y_final)
    final_xgb.save_model(MODEL_DIR / "xgb_model.json")
    final_lgb.booster_.save_model(str(MODEL_DIR / "lgb_model.txt"))
    with open(MODEL_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump(final_rf, f)
    np.save(MODEL_DIR / "label_classes.npy", final_le.classes_)

    print(f"\n✅ Models saved → {MODEL_DIR}/")

    # Summary
    def avg_prec(reports, label):
        return np.mean([r.get(label, {}).get("precision", 0) for r in reports])

    print("\n=== Walk-Forward Summary ===")
    print(f"{'Model':<12} {'Buy Prec':>10} {'Sell Prec':>10}")
    print("-" * 34)
    for name, reports in [("XGBoost", all_reports_xgb), ("LightGBM", all_reports_lgb), ("Ensemble", all_reports_ens)]:
        print(f"{name:<12} {avg_prec(reports, 'Buy'):>10.3f} {avg_prec(reports, 'Sell'):>10.3f}")

    all_preds_df = pd.concat(all_preds)
    all_preds_df.to_parquet(MODEL_DIR / "oof_predictions.parquet")
    print(f"\nOOF predictions saved → {MODEL_DIR}/oof_predictions.parquet")

    return final_xgb, final_lgb, final_le


if __name__ == "__main__":
    run_walk_forward()
