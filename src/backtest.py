"""
backtest.py — Mô phỏng trading theo OOF signals
Giả định: mua/bán theo giá đóng cửa ngày T+1 sau signal
Tính: P&L, win rate, Sharpe, drawdown
"""

import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from src.features import FEATURE_COLS, FEATURES_DIR
from src.model import ensemble_predict, walk_forward_splits, load_features, train_ensemble

MODEL_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "backtest"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === Trading params ===
TRANSACTION_COST = 0.0015   # 0.15% mỗi chiều (mua hoặc bán)
HOLD_DAYS = 5               # giữ 5 ngày sau khi vào lệnh
CONFIDENCE_THRESHOLD = 0.60 # chỉ trade khi confidence ≥ 60%
CAPITAL_PER_TRADE = 10_000_000  # 10 triệu VND mỗi lệnh


def run_backtest():
    print("Loading features & rebuilding OOF predictions with confidence...")
    df = load_features()
    splits = walk_forward_splits(df)

    all_trades = []

    for i, (train_mask, test_mask) in enumerate(splits):
        train = df[train_mask]
        test = df[test_mask]

        X_train = train[FEATURE_COLS].values
        y_train = train["target"].values
        X_test = test[FEATURE_COLS].values

        xgb_m, lgb_m, le = train_ensemble(X_train, y_train)

        # Wrap lgb
        class LGBWrapper:
            def __init__(self, booster):
                self.booster = booster
            def predict_proba(self, X):
                return self.booster.predict_proba(X)

        preds, proba_avg = ensemble_predict(xgb_m, lgb_m, le, X_test)
        confidence = proba_avg.max(axis=1)

        test = test.copy()
        test["prediction"] = preds
        test["confidence"] = confidence

        # Chỉ lấy các signal BUY/SELL với confidence đủ cao
        signals = test[
            (test["prediction"] != 0) &
            (test["confidence"] >= CONFIDENCE_THRESHOLD)
        ].copy()

        for idx, row in signals.iterrows():
            direction = row["prediction"]   # 1=Buy, -1=Sell
            entry_price = row["close"]

            # Tìm giá exit sau HOLD_DAYS ngày (trong cùng ticker)
            ticker_df = df[df["ticker"] == row["ticker"]].sort_index()
            future = ticker_df[ticker_df.index > idx].head(HOLD_DAYS)

            if len(future) < HOLD_DAYS:
                continue  # Không đủ ngày để exit

            exit_price = future["close"].iloc[-1]

            # P&L tính theo direction
            raw_return = (exit_price - entry_price) / entry_price
            if direction == -1:
                raw_return = -raw_return  # Sell: lãi khi giảm

            # Trừ phí cả 2 chiều
            net_return = raw_return - 2 * TRANSACTION_COST
            pnl = CAPITAL_PER_TRADE * net_return

            all_trades.append({
                "date": idx,
                "ticker": row["ticker"],
                "direction": "BUY" if direction == 1 else "SELL",
                "entry": round(entry_price, 2),
                "exit": round(exit_price, 2),
                "confidence": round(row["confidence"], 3),
                "raw_return": round(raw_return * 100, 2),
                "net_return": round(net_return * 100, 2),
                "pnl": round(pnl, 0),
                "split": i,
            })

        period = f"{test.index.min().date()} → {test.index.max().date()}"
        n_sig = len(signals)
        print(f"Split {i+1:02d} | {period} | Signals: {n_sig}")

    trades = pd.DataFrame(all_trades).sort_values("date")
    return trades


def analyze(trades: pd.DataFrame):
    if trades.empty:
        print("No trades.")
        return

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS  ({trades['date'].min().date()} → {trades['date'].max().date()})")
    print(f"{'='*60}")

    total = len(trades)
    wins = (trades["net_return"] > 0).sum()
    losses = (trades["net_return"] < 0).sum()

    print(f"\n📊 Overview")
    print(f"  Tổng lệnh:        {total}")
    print(f"  Thắng:            {wins} ({wins/total:.1%})")
    print(f"  Thua:             {losses} ({losses/total:.1%})")
    print(f"  Tổng P&L:         {trades['pnl'].sum():,.0f} VND")
    print(f"  P&L trung bình:   {trades['pnl'].mean():,.0f} VND / lệnh")

    print(f"\n📈 Returns")
    print(f"  Return avg:       {trades['net_return'].mean():.2f}%")
    print(f"  Return median:    {trades['net_return'].median():.2f}%")
    print(f"  Best trade:       {trades['net_return'].max():.2f}%")
    print(f"  Worst trade:      {trades['net_return'].min():.2f}%")

    # Sharpe (annualized, assume 252 trading days/year, 5-day holding)
    trades_per_year = 252 / 5
    ret_std = trades["net_return"].std()
    sharpe = (trades["net_return"].mean() / ret_std) * np.sqrt(trades_per_year) if ret_std > 0 else 0
    print(f"  Sharpe ratio:     {sharpe:.2f}")

    # Cumulative P&L + Max Drawdown
    trades["cumulative_pnl"] = trades["pnl"].cumsum()
    running_max = trades["cumulative_pnl"].cummax()
    drawdown = trades["cumulative_pnl"] - running_max
    max_dd = drawdown.min()
    print(f"  Max drawdown:     {max_dd:,.0f} VND  ({max_dd / CAPITAL_PER_TRADE:.1%} of 1 trade capital)")

    # By direction
    print(f"\n🔍 Phân tích theo hướng")
    for direction in ["BUY", "SELL"]:
        sub = trades[trades["direction"] == direction]
        if sub.empty:
            continue
        w = (sub["net_return"] > 0).sum()
        print(f"  {direction}: {len(sub)} lệnh | Win {w/len(sub):.1%} | Return avg {sub['net_return'].mean():.2f}% | P&L {sub['pnl'].sum():,.0f}")

    # By ticker
    print(f"\n🏆 Top 5 mã sinh lời nhất")
    by_ticker = trades.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
    for ticker, pnl in by_ticker.head(5).items():
        n = len(trades[trades["ticker"] == ticker])
        print(f"  {ticker:<6} {pnl:>12,.0f} VND ({n} lệnh)")

    print(f"\n❌ Top 5 mã thua lỗ nhất")
    for ticker, pnl in by_ticker.tail(5).items():
        n = len(trades[trades["ticker"] == ticker])
        print(f"  {ticker:<6} {pnl:>12,.0f} VND ({n} lệnh)")

    # Monthly P&L
    print(f"\n📅 P&L theo tháng")
    trades["month"] = pd.to_datetime(trades["date"]).dt.to_period("M")
    monthly = trades.groupby("month")["pnl"].sum()
    for month, pnl in monthly.items():
        bar_len = int(abs(pnl) / 2_000_000)
        bar = ("█" * min(bar_len, 20))
        sign = "+" if pnl >= 0 else "-"
        print(f"  {month}  {sign}{abs(pnl):>10,.0f}  {bar}")

    return trades


def save_trades(trades: pd.DataFrame):
    path = RESULTS_DIR / "trades.csv"
    trades.to_csv(path, index=False)
    print(f"\n✅ Trades saved → {path}")
    return path


if __name__ == "__main__":
    trades = run_backtest()
    trades = analyze(trades)
    save_trades(trades)
