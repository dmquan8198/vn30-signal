"""
compute_baseline.py — Tính baseline metrics từ backtest hiện tại.
Lưu vào reports/baseline_YYYYMMDD.json để so sánh sau mỗi phase.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(trades: pd.DataFrame, label: str = "all") -> dict:
    if trades.empty:
        return {}

    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"])
    trades = trades.sort_values("date")

    total = len(trades)
    wins = (trades["net_return"] > 0).sum()
    win_rate = wins / total

    avg_return = trades["net_return"].mean()
    median_return = trades["net_return"].median()
    avg_win = trades[trades["net_return"] > 0]["net_return"].mean() if wins > 0 else 0
    avg_loss = trades[trades["net_return"] <= 0]["net_return"].mean() if (total - wins) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Sharpe annualized (252 trading days / 5 hold days = 50.4 periods/year)
    periods_per_year = 252 / 5
    std = trades["net_return"].std()
    sharpe = (avg_return / std) * np.sqrt(periods_per_year) if std > 0 else 0

    # Sortino (downside std only)
    downside = trades[trades["net_return"] < 0]["net_return"]
    sortino_std = downside.std() if len(downside) > 1 else std
    sortino = (avg_return / sortino_std) * np.sqrt(periods_per_year) if sortino_std > 0 else 0

    # Cumulative P&L and drawdown
    cum_pnl = trades["pnl"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd_vnd = drawdown.min()
    total_pnl = cum_pnl.iloc[-1]

    # Annual return estimate (based on capital: assume 10M * avg positions)
    # Use total pnl / years
    years = (trades["date"].max() - trades["date"].min()).days / 365
    annual_pnl = total_pnl / years if years > 0 else total_pnl

    # Exposure: % of trading days with open position
    date_range = pd.date_range(trades["date"].min(), trades["date"].max(), freq="B")
    open_days = len(trades) * 5  # each trade holds 5 days
    exposure_pct = min(open_days / len(date_range), 1.0) if len(date_range) > 0 else 0

    # Trades per year
    trades_per_year = total / years if years > 0 else total

    # Hit rate for BUY signals with confidence >= 60%
    buy_high_conf = trades[
        (trades["direction"] == "BUY") & (trades["confidence"] >= 0.60)
    ]
    hit_rate_buy = (buy_high_conf["net_return"] > 0).mean() if not buy_high_conf.empty else None

    # By direction
    by_dir = {}
    for d in ["BUY", "SELL"]:
        sub = trades[trades["direction"] == d]
        if not sub.empty:
            by_dir[d] = {
                "n": len(sub),
                "win_rate": round((sub["net_return"] > 0).mean(), 4),
                "avg_return_pct": round(sub["net_return"].mean(), 4),
                "total_pnl": round(sub["pnl"].sum(), 0),
            }

    return {
        "label": label,
        "period": {
            "start": str(trades["date"].min().date()),
            "end": str(trades["date"].max().date()),
            "years": round(years, 2),
        },
        "trades": {
            "total": total,
            "per_year": round(trades_per_year, 1),
            "by_direction": by_dir,
        },
        "returns": {
            "win_rate": round(win_rate, 4),
            "avg_net_return_pct": round(avg_return, 4),
            "median_net_return_pct": round(median_return, 4),
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 3),
            "best_trade_pct": round(trades["net_return"].max(), 4),
            "worst_trade_pct": round(trades["net_return"].min(), 4),
        },
        "risk": {
            "sharpe_annualized": round(sharpe, 3),
            "sortino_annualized": round(sortino, 3),
            "max_drawdown_vnd": round(max_dd_vnd, 0),
            "max_drawdown_pct_of_trade_capital": round(max_dd_vnd / 10_000_000, 4),
        },
        "pnl": {
            "total_vnd": round(total_pnl, 0),
            "annual_vnd": round(annual_pnl, 0),
        },
        "efficiency": {
            "exposure_pct": round(exposure_pct, 4),
            "hit_rate_buy_conf60": round(hit_rate_buy, 4) if hit_rate_buy is not None else None,
        },
        "cost_model": {
            "transaction_cost_per_side": 0.0015,
            "round_trip_cost": 0.003,
            "note": "Thiếu sell tax (0.1%) và slippage — realistic cost ~0.5-0.8%",
        },
    }


def main():
    trades_path = ROOT / "backtest" / "trades.csv"
    if not trades_path.exists():
        print("ERROR: backtest/trades.csv không tồn tại. Chạy `python -m src.backtest` trước.")
        return

    trades = pd.read_csv(trades_path)
    print(f"Loaded {len(trades)} trades from {trades_path}")

    # Overall
    overall = compute_metrics(trades, label="all_directions")

    # BUY only (production config — SELL bị mute ở live)
    buy_only = compute_metrics(trades[trades["direction"] == "BUY"], label="buy_only")

    # High confidence BUY only
    buy_hc = compute_metrics(
        trades[(trades["direction"] == "BUY") & (trades["confidence"] >= 0.60)],
        label="buy_conf60plus"
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "source_file": str(trades_path),
        "config": {
            "transaction_cost": 0.0015,
            "hold_days": 5,
            "confidence_threshold": 0.60,
            "capital_per_trade": 10_000_000,
            "train_years": 3,
            "test_months": 6,
        },
        "note": "BASELINE — trước bất kỳ cải tiến nào. Dùng để so sánh sau Phase 1, 2, 3.",
        "all_directions": overall,
        "buy_only": buy_only,
        "buy_conf60plus": buy_hc,
    }

    date_str = datetime.now().strftime("%Y%m%d")
    out_path = REPORTS_DIR / f"baseline_{date_str}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Baseline saved → {out_path}")
    print("\n=== BASELINE METRICS SUMMARY ===")

    for section_key, section in [("all_directions", overall), ("buy_only", buy_only), ("buy_conf60plus", buy_hc)]:
        if not section:
            continue
        r = section
        print(f"\n[{section_key}] n={r['trades']['total']} trades | {r['period']['start']} → {r['period']['end']}")
        print(f"  Win rate:      {r['returns']['win_rate']:.1%}")
        print(f"  Avg return:    {r['returns']['avg_net_return_pct']:.2f}%")
        print(f"  Sharpe:        {r['risk']['sharpe_annualized']:.2f}")
        print(f"  Sortino:       {r['risk']['sortino_annualized']:.2f}")
        print(f"  Max DD:        {r['risk']['max_drawdown_vnd']:,.0f} VND")
        print(f"  Total P&L:     {r['pnl']['total_vnd']:,.0f} VND")
        print(f"  Annual P&L:    {r['pnl']['annual_vnd']:,.0f} VND")
        print(f"  Trades/year:   {r['trades']['per_year']:.0f}")
        if r['efficiency']['hit_rate_buy_conf60'] is not None:
            print(f"  Hit rate≥60%:  {r['efficiency']['hit_rate_buy_conf60']:.1%}")


if __name__ == "__main__":
    main()
