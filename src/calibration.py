"""
[P2.5] calibration.py — Isotonic regression để calibrate confidence scores.

Vấn đề: XGBoost/LightGBM confidence (predict_proba) thường không well-calibrated.
  - Model nói 70% confidence, thực tế win rate chỉ 55%
  - Isotonic regression maps raw probabilities → calibrated probabilities

Pipeline:
  1. Fit isotonic regressor trên OOF predictions (model.py lưu oof_predictions.parquet)
  2. Save calibrator
  3. Apply calibration khi generate live signals

Usage:
  from src.calibration import load_calibrator, calibrate_confidence
  cal = load_calibrator()
  calibrated = calibrate_confidence(cal, raw_confidence, signal_class)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATOR_PATH = MODEL_DIR / "calibrator.pkl"


def fit_calibrator(oof_path: Path = None) -> IsotonicRegression:
    """
    Fit isotonic regression calibrator trên OOF (out-of-fold) predictions.
    OOF predictions có cột: prediction, confidence, future_return (target thực)
    """
    if oof_path is None:
        oof_path = MODEL_DIR / "oof_predictions.parquet"

    if not oof_path.exists():
        print(f"  ⚠️  OOF predictions not found at {oof_path}")
        print("  Run src/model.py (walk-forward) to generate OOF predictions first.")
        return None

    oof = pd.read_parquet(oof_path)

    # Need: prediction label + confidence score + actual outcome
    if "prediction" not in oof.columns:
        print("  ⚠️  'prediction' column not found in OOF file")
        return None

    # Target: did BUY signal result in positive return?
    if "future_return" in oof.columns:
        oof["actual_positive"] = (oof["future_return"] > 0).astype(int)
    elif "target" in oof.columns:
        oof["actual_positive"] = (oof["target"] == 1).astype(int)
    else:
        print("  ⚠️  No 'future_return' or 'target' column in OOF file")
        return None

    # Calibrate on BUY predictions only
    buy_mask = oof["prediction"] == 1  # 1 = BUY in encoded labels
    if buy_mask.sum() < 30:
        buy_mask = oof["prediction"].astype(str).str.upper() == "BUY"

    buy_oof = oof[buy_mask].copy()
    if len(buy_oof) < 20:
        print(f"  ⚠️  Too few BUY predictions ({len(buy_oof)}) for calibration")
        return None

    # If confidence column not in OOF, we can't calibrate
    if "confidence" not in buy_oof.columns:
        print("  ⚠️  No 'confidence' column in OOF predictions — re-run backtest with confidence scores")
        return None

    X = buy_oof["confidence"].values
    y = buy_oof["actual_positive"].values

    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(X, y)

    # Evaluate calibration
    n_bins = 5
    try:
        prob_true, prob_pred = calibration_curve(y, X, n_bins=n_bins)
        ece_before = float(np.mean(np.abs(prob_true - prob_pred)))

        y_cal = cal.predict(X)
        prob_true_cal, prob_pred_cal = calibration_curve(y, y_cal, n_bins=n_bins)
        ece_after = float(np.mean(np.abs(prob_true_cal - prob_pred_cal)))
    except Exception:
        ece_before = ece_after = None

    print(f"  Calibration fitted on {len(buy_oof)} BUY OOF predictions")
    if ece_before is not None:
        print(f"  ECE before: {ece_before:.4f}  →  after: {ece_after:.4f}")

    # Save calibrator
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump(cal, f)
    print(f"  Calibrator saved → {CALIBRATOR_PATH}")

    return cal


def load_calibrator() -> IsotonicRegression | None:
    """Load saved calibrator, or return None if not available."""
    if not CALIBRATOR_PATH.exists():
        return None
    with open(CALIBRATOR_PATH, "rb") as f:
        return pickle.load(f)


def calibrate_confidence(
    calibrator: IsotonicRegression | None,
    raw_confidence: float | np.ndarray,
) -> float | np.ndarray:
    """
    Apply calibration to raw confidence scores.
    If no calibrator available, return raw score unchanged.
    """
    if calibrator is None:
        return raw_confidence

    scalar = isinstance(raw_confidence, (int, float))
    arr = np.atleast_1d(np.array(raw_confidence, dtype=float))
    calibrated = calibrator.predict(arr)

    return float(calibrated[0]) if scalar else calibrated


def calibration_report(calibrator: IsotonicRegression, oof_path: Path = None) -> dict:
    """Generate calibration report comparing raw vs calibrated probabilities."""
    if oof_path is None:
        oof_path = MODEL_DIR / "oof_predictions.parquet"
    if not oof_path.exists():
        return {}

    oof = pd.read_parquet(oof_path)
    if "confidence" not in oof.columns:
        return {}

    buy_mask = (oof["prediction"] == 1) | (oof["prediction"].astype(str).str.upper() == "BUY")
    buy_oof = oof[buy_mask].copy()

    if len(buy_oof) < 10:
        return {}

    target_col = "future_return" if "future_return" in buy_oof.columns else "target"
    if target_col == "future_return":
        actual = (buy_oof["future_return"] > 0).astype(int)
    else:
        actual = (buy_oof["target"] == 1).astype(int)

    raw = buy_oof["confidence"].values
    calibrated = calibrate_confidence(calibrator, raw)

    # Bin analysis
    bins = np.arange(0.5, 1.0, 0.1)
    rows = []
    for lo in bins:
        hi = lo + 0.1
        mask = (raw >= lo) & (raw < hi)
        if mask.sum() < 5:
            continue
        rows.append({
            "confidence_bin": f"{lo:.0%}–{hi:.0%}",
            "n": int(mask.sum()),
            "actual_win_rate": round(float(actual[mask].mean()), 3),
            "raw_confidence": round(float(raw[mask].mean()), 3),
            "calibrated_confidence": round(float(calibrated[mask].mean()), 3),
        })

    return {
        "n_buy_predictions": len(buy_oof),
        "bins": rows,
    }


if __name__ == "__main__":
    print("=== Confidence Calibration ===\n")
    cal = fit_calibrator()
    if cal is not None:
        report = calibration_report(cal)
        if report and report.get("bins"):
            print(f"\n  {'Bin':<12} {'N':>5} {'ActualWR':>9} {'RawConf':>9} {'CalConf':>9}")
            print(f"  {'─'*50}")
            for row in report["bins"]:
                print(
                    f"  {row['confidence_bin']:<12} {row['n']:>5} "
                    f"{row['actual_win_rate']:>8.1%} {row['raw_confidence']:>8.1%} "
                    f"{row['calibrated_confidence']:>8.1%}"
                )
