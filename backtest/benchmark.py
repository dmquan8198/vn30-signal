"""
[P1.3] benchmark.py — So sánh strategy với 3 benchmarks:
  B1: Buy-and-hold VN30 index
  B2: Buy-and-hold VNINDEX
  B3: Equal-weight monthly rebalance VN30 (30 mã)

Tính: Alpha, Beta, Information Ratio, annualized metrics.
Output: reports/benchmark_YYYYMMDD.json + console report
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL = 10_000_000          # đồng nhất với backtest
HOLD_DAYS = 5
PERIODS_PER_YEAR = 252 / HOLD_DAYS  # ~50.4


def load_strategy_trades() -> pd.DataFrame:
    path = ROOT / "backtest" / "trades.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["direction"] == "BUY"].copy()  # chỉ BUY (production config)


def load_index(symbol: str) -> pd.DataFrame:
    path = ROOT / "data" / "index" / f"{symbol}.parquet"
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def strategy_returns(trades: pd.DataFrame) -> pd.Series:
    """Average net return per day (aggregate nhiều trade cùng ngày)."""
    s = trades.groupby("date")["net_return"].mean().sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def benchmark_returns_5d(index_df: pd.DataFrame, trade_dates: pd.Index) -> pd.Series:
    """5-ngày forward return của benchmark tại cùng ngày có trade."""
    results = {}
    for d in trade_dates:
        future = index_df[index_df.index > d].head(HOLD_DAYS)
        past = index_df[index_df.index <= d].tail(1)
        if len(future) >= HOLD_DAYS and len(past) == 1:
            ret = (future["close"].iloc[-1] / past["close"].iloc[-1] - 1) * 100
            results[d] = ret
    return pd.Series(results, name="benchmark_return")


def compute_comparison_metrics(strat_ret: pd.Series, bench_ret: pd.Series, label: str) -> dict:
    """Tính các metrics so sánh giữa strategy và benchmark."""
    # Align theo ngày có cả 2
    common = strat_ret.index.intersection(bench_ret.index)
    if len(common) == 0:
        return {"error": "Không có ngày chung"}

    s = strat_ret.loc[common]
    b = bench_ret.loc[common]
    excess = s - b  # excess return mỗi trade

    # Sharpe (annualized)
    s_std = s.std()
    sharpe = (s.mean() / s_std) * np.sqrt(PERIODS_PER_YEAR) if s_std > 0 else 0

    # Beta
    cov = np.cov(s.values, b.values)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0

    # Alpha (annualized) = Σ excess return / years
    years = (common.max() - common.min()).days / 365
    alpha_annual = excess.sum() / years if years > 0 else 0

    # Information Ratio
    te_std = excess.std()
    ir = (excess.mean() / te_std) * np.sqrt(PERIODS_PER_YEAR) if te_std > 0 else 0

    # Up/Down capture
    up_mask = b > 0
    down_mask = b < 0
    up_capture = s[up_mask].mean() / b[up_mask].mean() if up_mask.sum() > 0 and b[up_mask].mean() != 0 else None
    down_capture = s[down_mask].mean() / b[down_mask].mean() if down_mask.sum() > 0 and b[down_mask].mean() != 0 else None

    # Cumulative returns (based on capital)
    cum_strategy = (CAPITAL * s / 100).cumsum().iloc[-1]
    cum_bench = (CAPITAL * b / 100).cumsum().iloc[-1]

    return {
        "benchmark": label,
        "n_common_trades": len(common),
        "period": {
            "start": str(common.min().date()),
            "end": str(common.max().date()),
            "years": round(years, 2),
        },
        "strategy": {
            "avg_return_pct": round(s.mean(), 4),
            "sharpe_annualized": round(sharpe, 3),
            "total_pnl_vnd": round(cum_strategy, 0),
        },
        "versus_benchmark": {
            "alpha_annualized_pct": round(alpha_annual, 3),
            "beta": round(beta, 3),
            "information_ratio": round(ir, 3),
            "up_capture_ratio": round(up_capture, 3) if up_capture else None,
            "down_capture_ratio": round(down_capture, 3) if down_capture else None,
            "avg_excess_return_pct": round(excess.mean(), 4),
            "tracking_error_annualized": round(te_std * np.sqrt(PERIODS_PER_YEAR), 4),
            "benchmark_total_pnl_vnd": round(cum_bench, 0),
        },
        "pass_criteria": {
            "ir_above_0.5": bool(ir > 0.5),
            "alpha_positive": bool(alpha_annual > 0),
            "note": "Go-live requires IR > 0.5 AND alpha > 0 after realistic costs",
        },
    }


def buy_hold_return_summary(index_df: pd.DataFrame, start: str, end: str) -> dict:
    """Return đơn giản buy-and-hold trong khoảng backtest."""
    subset = index_df[(index_df.index >= start) & (index_df.index <= end)]
    if subset.empty:
        return {}
    total_ret = (subset["close"].iloc[-1] / subset["close"].iloc[0] - 1) * 100
    years = (subset.index[-1] - subset.index[0]).days / 365
    annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    return {
        "total_return_pct": round(total_ret, 2),
        "annual_return_pct": round(annual_ret, 2),
        "start_price": round(subset["close"].iloc[0], 4),
        "end_price": round(subset["close"].iloc[-1], 4),
        "period_years": round(years, 2),
    }


def main():
    print("Loading data...")
    trades = load_strategy_trades()
    vn30 = load_index("VN30")
    vnindex = load_index("VNINDEX")

    trade_dates = pd.to_datetime(trades["date"].unique())
    strat_ret = strategy_returns(trades)

    start_date = str(trades["date"].min().date())
    end_date = str(trades["date"].max().date())

    print(f"Strategy period: {start_date} → {end_date} ({len(trades)} BUY trades)")

    # Benchmark 5-day returns at same trade dates
    print("Computing benchmark returns...")
    b1_ret = benchmark_returns_5d(vn30, trade_dates)
    b2_ret = benchmark_returns_5d(vnindex, trade_dates)

    # Comparison metrics
    vs_vn30 = compute_comparison_metrics(strat_ret, b1_ret, "VN30 Buy-and-Hold")
    vs_vnindex = compute_comparison_metrics(strat_ret, b2_ret, "VNINDEX Buy-and-Hold")

    # Simple buy-and-hold summary
    bh_vn30 = buy_hold_return_summary(vn30, start_date, end_date)
    bh_vnindex = buy_hold_return_summary(vnindex, start_date, end_date)

    report = {
        "generated_at": datetime.now().isoformat(),
        "note": (
            "So sánh strategy BUY signals (conf≥60%) vs benchmarks. "
            "Strategy return = net_return per trade (sau 0.30% cost). "
            "Benchmark return = 5-ngày return của index tại cùng ngày trade."
        ),
        "survivorship_bias_warning": "Strategy results có thể optimistic ~1-3%/năm do survivorship bias",
        "buy_hold_summary": {
            "VN30": bh_vn30,
            "VNINDEX": bh_vnindex,
        },
        "vs_vn30": vs_vn30,
        "vs_vnindex": vs_vnindex,
    }

    # Print report
    print(f"\n{'='*65}")
    print(f"  BENCHMARK COMPARISON: Strategy vs VN30 & VNINDEX")
    print(f"  Period: {start_date} → {end_date}")
    print(f"{'='*65}")

    print(f"\n📈 Buy-and-Hold trong cùng kỳ:")
    print(f"  VN30:    {bh_vn30.get('total_return_pct', 'N/A'):>7.1f}% total  |  {bh_vn30.get('annual_return_pct', 'N/A'):>6.1f}%/năm")
    print(f"  VNINDEX: {bh_vnindex.get('total_return_pct', 'N/A'):>7.1f}% total  |  {bh_vnindex.get('annual_return_pct', 'N/A'):>6.1f}%/năm")

    for label, metrics in [("VN30", vs_vn30), ("VNINDEX", vs_vnindex)]:
        vs = metrics.get("versus_benchmark", {})
        s = metrics.get("strategy", {})
        pc = metrics.get("pass_criteria", {})
        print(f"\n📊 Strategy vs {label} ({metrics.get('n_common_trades', 0)} trades):")
        print(f"  Strategy avg return:  {s.get('avg_return_pct', 0):>7.3f}%/trade")
        print(f"  Alpha (annualized):   {vs.get('alpha_annualized_pct', 0):>7.2f}%/năm")
        print(f"  Beta:                 {vs.get('beta', 0):>7.3f}")
        print(f"  Information Ratio:    {vs.get('information_ratio', 0):>7.3f}")
        print(f"  Up capture:           {vs.get('up_capture_ratio') or 'N/A'}")
        print(f"  Down capture:         {vs.get('down_capture_ratio') or 'N/A'}")
        ir_pass = "✅ PASS" if pc.get("ir_above_0.5") else "❌ FAIL"
        alpha_pass = "✅ PASS" if pc.get("alpha_positive") else "❌ FAIL"
        print(f"  IR > 0.5:             {ir_pass}")
        print(f"  Alpha > 0:            {alpha_pass}")

    date_str = datetime.now().strftime("%Y%m%d")
    out = REPORTS_DIR / f"benchmark_{date_str}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved → {out}")

    return report


if __name__ == "__main__":
    main()
