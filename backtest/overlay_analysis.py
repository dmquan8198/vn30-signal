"""
[P2.4] overlay_analysis.py — Ablation study: đo impact của từng overlay rule.

Phương pháp: so sánh performance khi BẬT vs TẮT từng rule:
  - Baseline: chỉ dùng ML signal (không overlay)
  - +Foreign flow: BẬT foreign flow filter
  - +Insider: BẬT insider trading filter
  - +Floor streak: BẬT floor/ceiling streak filter
  - +News sentiment: BẬT news sentiment overlay
  - Full: tất cả overlays (production)

Metrics: Sharpe, Win rate, Avg return, N_trades giữ lại

Output: reports/overlay_analysis_YYYYMMDD.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL = 10_000_000
PERIODS_PER_YEAR = 252 / 5
CONFIDENCE_THRESHOLD = 0.60


def load_trades() -> pd.DataFrame:
    path = ROOT / "backtest" / "trades.csv"
    if not path.exists():
        raise FileNotFoundError("backtest/trades.csv not found — run src/backtest.py first")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["direction"] == "BUY"].copy()


def load_signals_with_overlays() -> pd.DataFrame:
    """Load signal files that contain overlay columns."""
    signals_dir = ROOT / "signals"
    if not signals_dir.exists():
        return pd.DataFrame()

    dfs = []
    for f in sorted(signals_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f)
            df["signal_date"] = f.stem  # YYYY-MM-DD
            dfs.append(df)
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def compute_metrics(returns: np.ndarray, n_total: int) -> dict:
    if len(returns) == 0:
        return {"sharpe": None, "win_rate": None, "avg_return": None, "n_trades": 0, "filter_rate": 1.0}

    std = returns.std()
    sharpe = (returns.mean() / std) * np.sqrt(PERIODS_PER_YEAR) if std > 0 else 0.0
    pnl_cum = (CAPITAL * returns / 100).cumsum()
    running_max = np.maximum.accumulate(pnl_cum)
    max_dd = float((pnl_cum - running_max).min())

    return {
        "n_trades": len(returns),
        "filter_rate": round(1 - len(returns) / n_total, 3) if n_total > 0 else 0,
        "sharpe": round(sharpe, 3),
        "win_rate": round(float((returns > 0).mean()), 3),
        "avg_return_pct": round(float(returns.mean()), 4),
        "total_pnl_vnd": round(float((CAPITAL * returns / 100).sum()), 0),
        "max_dd_vnd": round(max_dd, 0),
    }


def simulate_overlay_rules(signals_df: pd.DataFrame) -> dict:
    """
    Simulate different overlay rule combinations on signal data.
    Returns metrics for each scenario.
    """
    results = {}
    n_total = len(signals_df)

    # ── Baseline: ML signal only, no overlay
    baseline = signals_df[signals_df["ml_signal"] == "BUY"]["net_return"].values
    results["baseline_ml_only"] = compute_metrics(baseline, n_total)
    results["baseline_ml_only"]["description"] = "ML signal only (no overlays)"

    # ── +Foreign flow
    if "foreign_signal" in signals_df.columns:
        mask = (
            (signals_df["ml_signal"] == "BUY") &
            (~signals_df["foreign_signal"].isin(["strong_sell"]))
        )
        arr = signals_df[mask]["net_return"].values
        results["plus_foreign_flow"] = compute_metrics(arr, n_total)
        results["plus_foreign_flow"]["description"] = "ML + foreign flow filter (remove strong_sell)"
    else:
        results["plus_foreign_flow"] = {"description": "Foreign flow data not available in signals"}

    # ── +Insider filter
    if "insider_sell_flag" in signals_df.columns:
        mask = (
            (signals_df["ml_signal"] == "BUY") &
            (signals_df["insider_sell_flag"] == 0)
        )
        arr = signals_df[mask]["net_return"].values
        results["plus_insider"] = compute_metrics(arr, n_total)
        results["plus_insider"]["description"] = "ML + insider filter (remove insider sell)"
    else:
        results["plus_insider"] = {"description": "Insider data not available in signals"}

    # ── +Floor streak (2+ consecutive)
    if "floor_streak" in signals_df.columns:
        mask = (
            (signals_df["ml_signal"] == "BUY") &
            (signals_df["floor_streak"] < 2)
        )
        arr = signals_df[mask]["net_return"].values
        results["plus_floor_streak"] = compute_metrics(arr, n_total)
        results["plus_floor_streak"]["description"] = "ML + floor streak filter (floor_streak < 2)"
    else:
        results["plus_floor_streak"] = {"description": "Floor streak data not available"}

    # ── +News sentiment
    if "news_sentiment" in signals_df.columns:
        mask = (
            (signals_df["ml_signal"] == "BUY") &
            (signals_df["news_sentiment"] >= 0)
        )
        arr = signals_df[mask]["net_return"].values
        results["plus_news_sentiment"] = compute_metrics(arr, n_total)
        results["plus_news_sentiment"]["description"] = "ML + news sentiment (non-negative only)"
    else:
        results["plus_news_sentiment"] = {"description": "News sentiment not available in signals"}

    # ── Full production (final signal == BUY after all overlays)
    if "signal" in signals_df.columns:
        prod = signals_df[signals_df["signal"] == "BUY"]["net_return"].values
        results["full_production"] = compute_metrics(prod, n_total)
        results["full_production"]["description"] = "Full production (all overlays applied)"
    else:
        results["full_production"] = {"description": "Production signal column not available"}

    return results


def ablation_from_trades(trades: pd.DataFrame) -> dict:
    """
    Fallback: ablation using only trades.csv data.
    Simulates what happens if we remove high-confidence filter.
    """
    n_total = len(trades)
    results = {}

    # All BUY trades
    all_ret = trades["net_return"].values
    results["all_buy_trades"] = compute_metrics(all_ret, n_total)
    results["all_buy_trades"]["description"] = "All BUY trades (any confidence)"

    # Confidence >= 0.60 (standard)
    if "confidence" in trades.columns:
        conf60 = trades[trades["confidence"] >= 0.60]["net_return"].values
        results["conf_60plus"] = compute_metrics(conf60, n_total)
        results["conf_60plus"]["description"] = "Confidence >= 60% filter"

        conf70 = trades[trades["confidence"] >= 0.70]["net_return"].values
        results["conf_70plus"] = compute_metrics(conf70, n_total)
        results["conf_70plus"]["description"] = "Confidence >= 70% filter (stricter)"

        conf80 = trades[trades["confidence"] >= 0.80]["net_return"].values
        results["conf_80plus"] = compute_metrics(conf80, n_total)
        results["conf_80plus"]["description"] = "Confidence >= 80% filter (aggressive)"

    return results


def print_ablation_table(results: dict, title: str):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")
    print(f"  {'Scenario':<28} {'N':>5} {'Filt%':>6} {'Sharpe':>7} {'WinR':>6} {'AvgRet':>8} {'PnL(M)':>8}")
    print(f"  {'─'*62}")
    for key, m in results.items():
        if "sharpe" not in m or m["sharpe"] is None:
            print(f"  {key:<28}  (insufficient data)")
            continue
        pnl_m = (m.get("total_pnl_vnd") or 0) / 1_000_000
        filt_pct = m.get("filter_rate", 0) * 100
        print(
            f"  {key:<28} {m['n_trades']:>5} {filt_pct:>5.1f}% "
            f"{m['sharpe']:>7.3f} {m['win_rate']:>5.1%} "
            f"{m['avg_return_pct']:>7.3f}% {pnl_m:>7.2f}M"
        )


def main():
    print("Loading trades...")
    trades = load_trades()
    print(f"  {len(trades)} BUY trades loaded")

    print("\nLoading signal files...")
    signals_df = load_signals_with_overlays()

    report = {
        "generated_at": datetime.now().isoformat(),
        "n_trades_total": len(trades),
    }

    # Confidence-level ablation (always available from trades.csv)
    print("\nRunning confidence-level ablation...")
    conf_results = ablation_from_trades(trades)
    print_ablation_table(conf_results, "Confidence Threshold Ablation")
    report["confidence_ablation"] = conf_results

    # Overlay-level ablation (needs signal files with overlay columns)
    if not signals_df.empty and "ml_signal" in signals_df.columns:
        print("\nRunning overlay-rule ablation...")
        overlay_results = simulate_overlay_rules(signals_df)
        print_ablation_table(overlay_results, "Overlay Rule Ablation")
        report["overlay_ablation"] = overlay_results
    else:
        note = "Signal files lack overlay columns — overlay ablation skipped"
        print(f"\n  ⚠️  {note}")
        report["overlay_ablation"] = {"note": note}

    # Save report
    date_str = datetime.now().strftime("%Y%m%d")
    out = REPORTS_DIR / f"overlay_analysis_{date_str}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved → {out}")

    return report


if __name__ == "__main__":
    main()
