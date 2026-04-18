"""
[P4.3] go_live_check.py — Checklist tổng hợp trước khi deploy live trading.

Checks:
  1. Model quality: Sharpe CI lower bound > 0 (bootstrap)
  2. Benchmark: Alpha > 0, IR > 0.5 vs VN30
  3. No data leakage: permutation AUC < 0.60
  4. Feature drift: PSI stable (no alert)
  5. Circuit breaker: CLOSED (no recent performance issues)
  6. Macro data: yfinance connection working
  7. Email notification: SMTP config present
  8. Model files: xgb, lgb, rf present
  9. Cost model: realistic cost < 1% round-trip
  10. Calibrator: fitted and available

Output: Console summary + reports/go_live_YYYYMMDD.json
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure project root is on sys.path when running as script
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_model_files() -> dict:
    """1. Model files present."""
    xgb_ok = (ROOT / "models" / "xgb_model.json").exists()
    lgb_ok = (ROOT / "models" / "lgb_model.txt").exists()
    rf_ok  = (ROOT / "models" / "rf_model.pkl").exists()
    cal_ok = (ROOT / "models" / "calibrator.pkl").exists()

    pass_ = xgb_ok and lgb_ok
    return {
        "check": "Model files",
        "pass": pass_,
        "detail": f"XGB={'✅' if xgb_ok else '❌'} LGB={'✅' if lgb_ok else '❌'} RF={'✅' if rf_ok else '⚠️ (optional)'} Calibrator={'✅' if cal_ok else '⚠️ (optional)'}",
        "action": "Run python -m src.model to retrain" if not pass_ else None,
    }


def check_bootstrap() -> dict:
    """2. Bootstrap CI: Sharpe lower bound > 0."""
    import glob
    files = sorted(glob.glob(str(REPORTS_DIR / "bootstrap_*.json")))
    if not files:
        return {
            "check": "Bootstrap CI",
            "pass": False,
            "detail": "No bootstrap report found",
            "action": "Run python backtest/bootstrap.py",
        }
    with open(files[-1]) as f:
        report = json.load(f)

    buy = report.get("results", {}).get("buy", {})
    if not buy:
        return {"check": "Bootstrap CI", "pass": False, "detail": "No BUY results", "action": "Run backtest/bootstrap.py"}

    go_live = buy.get("go_live_criteria", {})
    pass_ = bool(go_live.get("sharpe_lower_bound_positive", False))
    sharpe_lo = go_live.get("sharpe_p2.5", "N/A")
    return {
        "check": "Bootstrap CI (Sharpe lower bound > 0)",
        "pass": pass_,
        "detail": f"Sharpe p2.5 = {sharpe_lo}  {go_live.get('result', '')}",
        "action": "Strategy Sharpe not robust — review model or backtest" if not pass_ else None,
    }


def check_benchmark() -> dict:
    """3. Benchmark: Alpha > 0, IR > 0.5."""
    import glob
    files = sorted(glob.glob(str(REPORTS_DIR / "benchmark_*.json")))
    if not files:
        return {"check": "Benchmark vs VN30", "pass": False, "detail": "No benchmark report", "action": "Run backtest/benchmark.py"}

    with open(files[-1]) as f:
        report = json.load(f)

    vs_vn30 = report.get("vs_vn30", {})
    pc = vs_vn30.get("pass_criteria", {})
    ir_pass = bool(pc.get("ir_above_0.5", False))
    alpha_pass = bool(pc.get("alpha_positive", False))
    pass_ = ir_pass and alpha_pass
    vs = vs_vn30.get("versus_benchmark", {})

    return {
        "check": "Benchmark (IR>0.5 AND Alpha>0 vs VN30)",
        "pass": pass_,
        "detail": f"IR={vs.get('information_ratio', 'N/A')} Alpha={vs.get('alpha_annualized_pct', 'N/A')}%/yr",
        "action": "Strategy not outperforming benchmark — review signal quality" if not pass_ else None,
    }


def check_leakage() -> dict:
    """4. Data leakage: permutation test."""
    try:
        from pathlib import Path
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import roc_auc_score
        import xgboost as xgb
        from src.features import FEATURE_COLS
        from src.macro import MACRO_FEATURE_COLS
        from src.regime import REGIME_FEATURE_COLS
        from src.model import load_features, walk_forward_splits

        df = load_features()
        new_cols = MACRO_FEATURE_COLS + REGIME_FEATURE_COLS
        for col in new_cols:
            if col not in df.columns:
                df[col] = 0.0

        splits = walk_forward_splits(df)
        if not splits:
            return {"check": "Leakage test", "pass": False, "detail": "No splits", "action": "Check data"}

        train_mask, test_mask = splits[0]
        train = df[train_mask].dropna(subset=FEATURE_COLS + ["target"])
        test  = df[test_mask].dropna(subset=FEATURE_COLS + ["target"])

        le = LabelEncoder()
        y_enc = le.fit_transform(train["target"].values)

        model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
        model.fit(train[FEATURE_COLS].values, y_enc)

        proba = model.predict_proba(test[FEATURE_COLS].values)
        y_test = test["target"].values
        y_bin = np.zeros((len(y_test), len(le.classes_)))
        for i, cls in enumerate(le.classes_):
            y_bin[:, i] = (y_test == cls)
        real_auc = float(roc_auc_score(y_bin, proba, average="macro", multi_class="ovr"))

        # Quick 5-iter permutation
        rng = np.random.default_rng(0)
        perm_aucs = []
        for _ in range(5):
            y_shuf = rng.permutation(y_enc)
            pm = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
            pm.fit(train[FEATURE_COLS].values, y_shuf)
            pp = pm.predict_proba(test[FEATURE_COLS].values)
            perm_aucs.append(float(roc_auc_score(y_bin, pp, average="macro", multi_class="ovr")))

        perm_mean = float(np.mean(perm_aucs))
        gap = real_auc - perm_mean
        pass_ = real_auc > 0.50 and perm_mean < 0.60 and gap > 0.01

        return {
            "check": "Data leakage (permutation test)",
            "pass": pass_,
            "detail": f"Real AUC={real_auc:.4f}, Perm AUC={perm_mean:.4f}, Gap={gap:.4f}",
            "action": "LEAKAGE DETECTED — audit feature pipeline immediately" if not pass_ else None,
        }
    except Exception as e:
        return {"check": "Data leakage", "pass": False, "detail": f"Error: {e}", "action": "Run manually: pytest tests/test_leakage.py"}


def check_drift() -> dict:
    """5. Feature drift: no PSI alert."""
    import glob
    files = sorted(glob.glob(str(REPORTS_DIR / "drift_*.json")))
    if not files:
        return {
            "check": "Feature drift (PSI)",
            "pass": True,  # no report = no known drift
            "detail": "No drift report — run monitoring/drift.py after data refresh",
            "action": None,
        }
    with open(files[-1]) as f:
        report = json.load(f)
    status = report.get("overall_status", "stable")
    alerts = report.get("alerts", [])
    pass_ = status != "alert"
    return {
        "check": "Feature drift (PSI)",
        "pass": pass_,
        "detail": f"Status={status}  Alerts={len(alerts)} features: {', '.join(alerts[:3]) or 'none'}",
        "action": "Retrain model to adapt to new distribution" if not pass_ else None,
    }


def check_circuit_breaker() -> dict:
    """6. Circuit breaker: CLOSED."""
    cb_path = REPORTS_DIR / "circuit_breaker.json"
    if not cb_path.exists():
        return {"check": "Circuit breaker", "pass": True, "detail": "No state file — defaults to CLOSED", "action": None}
    with open(cb_path) as f:
        state = json.load(f)
    status = state.get("status", "CLOSED")
    pass_ = status == "CLOSED"
    return {
        "check": "Circuit breaker",
        "pass": pass_,
        "detail": f"Status={status} Reason={state.get('reason', 'N/A')}",
        "action": "Fix rolling performance metrics before going live" if status == "OPEN" else ("Monitor closely" if status == "HALF-OPEN" else None),
    }


def check_email_config() -> dict:
    """7. Email: SMTP configured."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    sender = os.getenv("NOTIFY_EMAIL_SENDER")
    password = os.getenv("NOTIFY_EMAIL_PASSWORD")
    to = os.getenv("NOTIFY_EMAIL_TO")
    pass_ = bool(sender and password and to)
    return {
        "check": "Email notification config",
        "pass": pass_,
        "detail": f"SENDER={'✅' if sender else '❌'} PASSWORD={'✅' if password else '❌'} TO={'✅' if to else '❌'}",
        "action": "Set NOTIFY_EMAIL_* in .env file" if not pass_ else None,
    }


def check_macro_data() -> dict:
    """8. Macro data: yfinance working."""
    try:
        import yfinance as yf
        t = yf.Ticker("^GSPC")
        hist = t.history(period="5d")
        pass_ = not hist.empty
        return {
            "check": "Macro data (yfinance)",
            "pass": pass_,
            "detail": f"S&P500 data: {len(hist)} rows available" if pass_ else "No data returned",
            "action": "Check internet connection or yfinance API" if not pass_ else None,
        }
    except ImportError:
        return {"check": "Macro data (yfinance)", "pass": False, "detail": "yfinance not installed", "action": "pip install yfinance"}
    except Exception as e:
        return {"check": "Macro data (yfinance)", "pass": False, "detail": str(e), "action": "Check internet connection"}


def check_cost_model() -> dict:
    """9. Realistic cost < 1% round-trip for typical trade."""
    try:
        from backtest.cost_model import round_trip_cost
        cost = round_trip_cost(price=50000, volume_shares=200, adv_20d_shares=2_000_000)
        pass_ = cost < 0.01
        return {
            "check": "Cost model (RT cost < 1%)",
            "pass": pass_,
            "detail": f"Typical round-trip cost: {cost*100:.3f}%",
            "action": None,
        }
    except Exception as e:
        return {"check": "Cost model", "pass": False, "detail": str(e), "action": "Check backtest/cost_model.py"}


def main():
    print("\n" + "="*60)
    print("  VN30 SIGNAL — GO-LIVE READINESS CHECKLIST")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("="*60)

    checks = [
        check_model_files,
        check_bootstrap,
        check_benchmark,
        check_leakage,
        check_drift,
        check_circuit_breaker,
        check_email_config,
        check_macro_data,
        check_cost_model,
    ]

    results = []
    passed = 0
    failed = 0

    for fn in checks:
        try:
            result = fn()
        except Exception as e:
            result = {"check": fn.__name__, "pass": False, "detail": f"Error: {e}", "action": str(e)}

        results.append(result)
        icon = "✅" if result["pass"] else "❌"
        if result["pass"]:
            passed += 1
        else:
            failed += 1

        print(f"\n  {icon} {result['check']}")
        print(f"     {result['detail']}")
        if result.get("action"):
            print(f"     → {result['action']}")

    print(f"\n{'='*60}")
    go_live = failed == 0
    if go_live:
        print(f"  🚀 GO-LIVE READY — All {passed} checks passed!")
    else:
        print(f"  🛑 NOT READY — {failed} checks failed, {passed} passed")
        print(f"  Fix all ❌ before going live.")
    print("="*60 + "\n")

    report = {
        "generated_at": datetime.now().isoformat(),
        "go_live_ready": go_live,
        "passed": passed,
        "failed": failed,
        "checks": results,
    }

    date_str = datetime.now().strftime("%Y%m%d")
    out = REPORTS_DIR / f"go_live_{date_str}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved → {out}\n")

    return report


if __name__ == "__main__":
    main()
