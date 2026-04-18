"""
[P1.5] cost_model.py — Realistic transaction cost model cho VN30.

Current (underestimate): 0.15% mỗi chiều = 0.30% round-trip
Realistic: broker fee + sell tax + slippage = 0.50-0.80% round-trip

Breakdown:
- Broker fee: 0.15%/chiều (standard VN retail)
- Sell tax: 0.10% chỉ chiều bán (thuế TNCN từ bán CK)
- Slippage: 0.05-0.30% tùy thanh khoản
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent


def realistic_cost(
    side: str,
    price: float,
    volume_shares: int,
    adv_20d_shares: float,
) -> float:
    """
    Tính chi phí giao dịch thực tế (tỷ lệ trên giá trị lệnh).

    Args:
        side: 'buy' hoặc 'sell'
        price: Giá cổ phiếu (VND)
        volume_shares: Số lượng cổ phần trong lệnh
        adv_20d_shares: Average daily volume 20 ngày (số cổ phần)

    Returns:
        Tỷ lệ chi phí (0.0 → 1.0), ví dụ 0.006 = 0.6%
    """
    broker_fee = 0.0015  # 0.15%/chiều — standard Vietnam retail

    # Thuế bán (0.1% giá trị bán — áp dụng khi bán)
    sell_tax = 0.001 if side == "sell" else 0.0

    # Slippage dựa trên market impact (% of ADV)
    if adv_20d_shares > 0:
        order_pct_adv = volume_shares / adv_20d_shares
    else:
        order_pct_adv = 0.01  # default giả sử 1% ADV

    if order_pct_adv < 0.001:       # < 0.1% ADV — nhỏ, không impact
        slippage = 0.0005            # 0.05%
    elif order_pct_adv < 0.005:     # 0.1-0.5% ADV
        slippage = 0.0010            # 0.10%
    elif order_pct_adv < 0.01:      # 0.5-1% ADV
        slippage = 0.0015            # 0.15%
    elif order_pct_adv < 0.05:      # 1-5% ADV — sizeable
        slippage = 0.0025            # 0.25%
    else:                            # > 5% ADV — large, high impact
        slippage = 0.0050            # 0.50%

    return broker_fee + sell_tax + slippage


def round_trip_cost(
    price: float,
    volume_shares: int,
    adv_20d_shares: float,
) -> float:
    """Tổng chi phí khứ hồi (buy + sell)."""
    buy_cost = realistic_cost("buy", price, volume_shares, adv_20d_shares)
    sell_cost = realistic_cost("sell", price, volume_shares, adv_20d_shares)
    return buy_cost + sell_cost


def estimate_vn30_adv(ticker: str) -> float:
    """Ước tính ADV 20 ngày từ cached data (số cổ phần)."""
    path = ROOT / "data" / "raw" / f"{ticker}.parquet"
    if not path.exists():
        return 1_000_000  # fallback: 1M cổ phần/ngày
    df = pd.read_parquet(path).tail(20)
    if "volume" not in df.columns or df.empty:
        return 1_000_000
    return float(df["volume"].mean())


def estimate_order_size(capital: float, price_vnd: float) -> int:
    """Số cổ phần tương ứng với capital cho trước."""
    if price_vnd <= 0:
        return 0
    # VN: lô cơ sở 100 cổ phần, làm tròn xuống
    raw = int(capital / price_vnd)
    return (raw // 100) * 100


def get_realistic_cost_for_trade(
    ticker: str,
    price_vnd: float,
    capital: float = 10_000_000,
) -> dict:
    """
    Tính realistic cost cho 1 lệnh với capital cho trước.
    price_vnd: giá thực (đã nhân 1000 so với vnstock unit)
    """
    adv = estimate_vn30_adv(ticker)
    shares = estimate_order_size(capital, price_vnd)

    buy_c = realistic_cost("buy", price_vnd, shares, adv)
    sell_c = realistic_cost("sell", price_vnd, shares, adv)
    total = buy_c + sell_c

    return {
        "ticker": ticker,
        "price_vnd": price_vnd,
        "shares": shares,
        "adv_20d_shares": round(adv),
        "order_pct_adv": round(shares / adv * 100, 3) if adv > 0 else None,
        "buy_cost_pct": round(buy_c * 100, 4),
        "sell_cost_pct": round(sell_c * 100, 4),
        "round_trip_cost_pct": round(total * 100, 4),
        "cost_vs_current_model_pct": round((total - 0.003) * 100, 4),  # delta so với 0.30%
    }


def rerun_backtest_with_realistic_cost(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Tính lại P&L với realistic cost thay vì 0.30% hardcode.
    Trả về DataFrame với thêm cột: realistic_net_return, realistic_pnl
    """
    CAPITAL = 10_000_000

    results = []
    for _, row in trades.iterrows():
        ticker = row["ticker"]
        # Entry price trong trades.csv là đơn vị 1000 VND (từ vnstock)
        price_vnd = row["entry"] * 1000  # convert về VND thực

        cost_info = get_realistic_cost_for_trade(ticker, price_vnd, CAPITAL)
        rt_cost = cost_info["round_trip_cost_pct"] / 100  # convert về ratio

        raw_ret = row["raw_return"] / 100  # convert về ratio
        realistic_net = raw_ret - rt_cost
        realistic_pnl = CAPITAL * realistic_net

        results.append({
            **row.to_dict(),
            "realistic_rt_cost_pct": cost_info["round_trip_cost_pct"],
            "realistic_net_return": round(realistic_net * 100, 4),
            "realistic_pnl": round(realistic_pnl, 0),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import json

    print("=== Realistic Cost Model Demo ===\n")

    # Demo cho từng mã VN30
    demo_tickers = ["VCB", "FPT", "HPG", "VIC", "ACB"]
    demo_capital = 10_000_000

    print(f"{'Ticker':<6} {'Giá(VND)':>10} {'Shares':>8} {'%ADV':>8} {'RT Cost':>9} {'vs 0.30%':>10}")
    print("-" * 60)

    for ticker in demo_tickers:
        path = ROOT / "data" / "raw" / f"{ticker}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        price_raw = float(df["close"].iloc[-1])  # đơn vị 1000 VND
        price_vnd = price_raw * 1000

        info = get_realistic_cost_for_trade(ticker, price_vnd, demo_capital)
        print(
            f"{ticker:<6} {price_vnd:>10,.0f} {info['shares']:>8,} "
            f"{info['order_pct_adv'] or 0:>7.3f}% "
            f"{info['round_trip_cost_pct']:>8.3f}% "
            f"{info['cost_vs_current_model_pct']:>+9.3f}%"
        )

    print("\n--- Backtest với Realistic Cost ---")
    trades_path = ROOT / "backtest" / "trades.csv"
    trades = pd.read_csv(trades_path)
    buy_trades = trades[trades["direction"] == "BUY"].copy()

    realistic = rerun_backtest_with_realistic_cost(buy_trades)

    old_sharpe = (buy_trades["net_return"].mean() / buy_trades["net_return"].std()) * np.sqrt(252 / 5)
    new_sharpe = (realistic["realistic_net_return"].mean() / realistic["realistic_net_return"].std()) * np.sqrt(252 / 5)

    print(f"\nCurrent model (0.30% RT):")
    print(f"  Avg net return: {buy_trades['net_return'].mean():.3f}%")
    print(f"  Sharpe:         {old_sharpe:.3f}")
    print(f"  Total P&L:      {buy_trades['pnl'].sum():,.0f} VND")

    print(f"\nRealistic cost (~0.50-0.80% RT):")
    print(f"  Avg net return: {realistic['realistic_net_return'].mean():.3f}%")
    print(f"  Sharpe:         {new_sharpe:.3f}")
    print(f"  Total P&L:      {realistic['realistic_pnl'].sum():,.0f} VND")
    print(f"\n  Sharpe delta:   {new_sharpe - old_sharpe:+.3f}")
    print(f"  P&L delta:      {realistic['realistic_pnl'].sum() - buy_trades['pnl'].sum():,.0f} VND")
