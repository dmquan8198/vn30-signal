"""
[P2.2] drift.py — Feature drift detection using Population Stability Index (PSI).

PSI measures distributional shift between training data and recent live data:
  PSI < 0.1  → No significant change (stable)
  PSI 0.1-0.2 → Moderate change (monitor)
  PSI > 0.2  → Significant shift → retrain recommended

Output: reports/drift_YYYYMMDD.json
Alert: logged to reports/drift_alerts.csv
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ALERT_LOG = REPORTS_DIR / "drift_alerts.csv"
FEATURES_DIR = ROOT / "data" / "features"

# PSI thresholds
PSI_WARN = 0.10
PSI_ALERT = 0.20

# Features to monitor (most important for model)
MONITOR_FEATURES = [
    "rsi_14", "macd", "macd_signal",
    "ret_5d", "ret_20d",
    "vol_ratio", "atr_pct",
    "market_ret_5d", "market_vol_20d",
    "rel_strength_20d",
    "sector_rel_strength",
]

# Training reference window vs live window
TRAIN_CUTOFF_MONTHS = 6    # use last 6 months of training data as reference
LIVE_WINDOW_DAYS = 30      # compare against last 30 days of live data


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between reference and current distributions.
    Uses quantile-based binning on reference distribution.
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 10 or len(current) < 10:
        return 0.0

    # Bin boundaries from reference quantiles
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # Avoid division by zero or log(0)
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def load_feature_data() -> pd.DataFrame:
    path = FEATURES_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError("features.parquet not found — run src/features.py first")
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def run_drift_check() -> dict:
    """
    Compare feature distributions:
      Reference: training data from 6-18 months ago
      Current: last 30 days of data
    """
    try:
        df = load_feature_data()
    except FileNotFoundError as e:
        return {"error": str(e), "alerts": []}

    end_date = df.index.max()
    live_start = end_date - pd.DateOffset(days=LIVE_WINDOW_DAYS)
    ref_end = end_date - pd.DateOffset(months=TRAIN_CUTOFF_MONTHS)
    ref_start = ref_end - pd.DateOffset(months=12)

    reference = df[df.index < ref_end]
    if ref_start is not None:
        reference = reference[reference.index >= ref_start]
    current = df[df.index >= live_start]

    if reference.empty or current.empty:
        return {"error": "Insufficient data for drift check", "alerts": []}

    results = {}
    alerts = []

    for feat in MONITOR_FEATURES:
        if feat not in df.columns:
            continue
        ref_vals = reference[feat].dropna().values
        cur_vals = current[feat].dropna().values
        psi = compute_psi(ref_vals, cur_vals)

        status = "stable"
        if psi > PSI_ALERT:
            status = "alert"
            alerts.append(f"{feat} PSI={psi:.3f}")
        elif psi > PSI_WARN:
            status = "warn"

        results[feat] = {
            "psi": round(psi, 4),
            "status": status,
            "ref_mean": round(float(np.nanmean(ref_vals)), 4) if len(ref_vals) else None,
            "cur_mean": round(float(np.nanmean(cur_vals)), 4) if len(cur_vals) else None,
            "ref_n": len(ref_vals),
            "cur_n": len(cur_vals),
        }

    overall_status = "alert" if alerts else ("warn" if any(v["status"] == "warn" for v in results.values()) else "stable")

    report = {
        "generated_at": datetime.now().isoformat(),
        "reference_period": f"{reference.index.min().date()} → {reference.index.max().date()}",
        "current_period": f"{current.index.min().date()} → {current.index.max().date()}",
        "overall_status": overall_status,
        "psi_thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT},
        "features": results,
        "alerts": alerts,
        "recommendation": (
            "⚠️ Retrain recommended — significant feature drift detected" if overall_status == "alert"
            else "Monitor — moderate drift, consider retraining" if overall_status == "warn"
            else "✅ No significant drift"
        ),
    }

    # Save report
    date_str = datetime.now().strftime("%Y%m%d")
    out = REPORTS_DIR / f"drift_{date_str}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Append to alert log
    log_row = pd.DataFrame([{
        "date": datetime.now().isoformat(),
        "status": overall_status,
        "n_alerts": len(alerts),
        "alert_features": "; ".join(alerts) if alerts else "",
    }])
    if ALERT_LOG.exists():
        existing = pd.read_csv(ALERT_LOG)
        log_row = pd.concat([existing, log_row], ignore_index=True)
    log_row.to_csv(ALERT_LOG, index=False)

    return report


def print_drift_report(report: dict):
    print(f"\n{'='*55}")
    print(f"  DRIFT DETECTION REPORT (PSI)")
    print(f"  Reference: {report.get('reference_period', 'N/A')}")
    print(f"  Current:   {report.get('current_period', 'N/A')}")
    print(f"{'='*55}")
    print(f"\n  Overall: {report.get('overall_status', '').upper()}")
    print(f"  {report.get('recommendation', '')}")
    print(f"\n  {'Feature':<25} {'PSI':>6}  {'Status':<8}  {'Ref Mean':>9}  {'Cur Mean':>9}")
    print(f"  {'-'*65}")
    for feat, info in sorted(report.get("features", {}).items(), key=lambda x: -x[1]["psi"]):
        icon = "🔴" if info["status"] == "alert" else "🟡" if info["status"] == "warn" else "🟢"
        print(
            f"  {feat:<25} {info['psi']:>6.3f}  {icon} {info['status']:<6}  "
            f"{info.get('ref_mean') or 0:>9.4f}  {info.get('cur_mean') or 0:>9.4f}"
        )

    if report.get("alerts"):
        print(f"\n  🚨 Alerts: {', '.join(report['alerts'])}")


if __name__ == "__main__":
    print("Running drift check...")
    report = run_drift_check()
    print_drift_report(report)
    print(f"\n✅ Saved → reports/drift_{datetime.now().strftime('%Y%m%d')}.json")
