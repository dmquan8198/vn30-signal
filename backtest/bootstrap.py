"""
[P1.7] bootstrap.py — Block bootstrap cho backtest metrics.
Tính 95% confidence interval cho Sharpe, Annual return, Max DD, Win rate.

Methodology: Block bootstrap (block_size=20 ngày) với 1000 iterations.
Go-live criteria: Sharpe CI lower bound > 0.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
BLOCK_SIZE = 20      # 20 ngày = ~1 tháng giao dịch
PERIODS_PER_YEAR = 252 / 5
CAPITAL = 10_000_000


def block_bootstrap_sample(data: np.ndarray, block_size: int, rng) -> np.ndarray:
    """Lấy mẫu block bootstrap giữ nguyên autocorrelation trong mỗi block."""
    n = len(data)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, max(1, n - block_size), size=n_blocks)
    blocks = [data[s:s + block_size] for s in starts]
    sample = np.concatenate(blocks)[:n]
    return sample


def compute_sharpe(returns: np.ndarray) -> float:
    std = returns.std()
    if std == 0:
        return 0.0
    return (returns.mean() / std) * np.sqrt(PERIODS_PER_YEAR)


def compute_max_dd(returns: np.ndarray, capital: float = CAPITAL) -> float:
    pnl = (capital * returns / 100).cumsum()
    running_max = np.maximum.accumulate(pnl)
    dd = pnl - running_max
    return float(dd.min())


def compute_annual_return(returns: np.ndarray, n_years: float) -> float:
    total = returns.sum()  # sum of percent returns per trade
    return total / n_years if n_years > 0 else total


def run_bootstrap(trades: pd.DataFrame, direction: str = "BUY") -> dict:
    sub = trades[trades["direction"] == direction].copy()
    if sub.empty:
        return {}

    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date")
    returns = sub["net_return"].values

    n = len(returns)
    years = (sub["date"].max() - sub["date"].min()).days / 365

    rng = np.random.default_rng(seed=42)
    sharpes, max_dds, win_rates, annual_rets = [], [], [], []

    for _ in range(N_BOOTSTRAP):
        sample = block_bootstrap_sample(returns, BLOCK_SIZE, rng)
        sharpes.append(compute_sharpe(sample))
        max_dds.append(compute_max_dd(sample))
        win_rates.append((sample > 0).mean())
        annual_rets.append(compute_annual_return(sample, years))

    def ci95(arr):
        lo = np.percentile(arr, 2.5)
        hi = np.percentile(arr, 97.5)
        med = np.median(arr)
        return {"p2.5": round(lo, 4), "median": round(med, 4), "p97.5": round(hi, 4)}

    real_sharpe = compute_sharpe(returns)
    real_max_dd = compute_max_dd(returns)
    real_win_rate = (returns > 0).mean()
    real_annual_ret = compute_annual_return(returns, years)

    sharpe_lo = np.percentile(sharpes, 2.5)
    go_live_pass = bool(sharpe_lo > 0)

    return {
        "direction": direction,
        "n_trades": n,
        "n_bootstrap": N_BOOTSTRAP,
        "block_size": BLOCK_SIZE,
        "observed": {
            "sharpe": round(real_sharpe, 4),
            "max_dd_vnd": round(real_max_dd, 0),
            "win_rate": round(real_win_rate, 4),
            "annual_return_pct_sum": round(real_annual_ret, 4),
        },
        "bootstrap_95ci": {
            "sharpe": ci95(sharpes),
            "max_dd_vnd": ci95(max_dds),
            "win_rate": ci95(win_rates),
            "annual_return_pct_sum": ci95(annual_rets),
        },
        "go_live_criteria": {
            "sharpe_lower_bound_positive": go_live_pass,
            "sharpe_p2.5": round(sharpe_lo, 4),
            "result": "✅ PASS — Sharpe lower bound > 0" if go_live_pass else "❌ FAIL — Sharpe lower bound ≤ 0, strategy NOT reliable",
        },
    }


def main():
    trades_path = ROOT / "backtest" / "trades.csv"
    trades = pd.read_csv(trades_path)

    print(f"Running block bootstrap (n={N_BOOTSTRAP}, block={BLOCK_SIZE})...")
    print("This may take ~30 seconds...\n")

    results = {}
    for direction in ["BUY", "SELL"]:
        sub = trades[trades["direction"] == direction]
        if sub.empty:
            continue
        print(f"  Bootstrapping {direction} signals ({len(sub)} trades)...")
        results[direction.lower()] = run_bootstrap(trades, direction)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BOOTSTRAP RESULTS (95% CI, Block={BLOCK_SIZE})")
    print(f"{'='*60}")

    for key, r in results.items():
        if not r:
            continue
        ci = r["bootstrap_95ci"]
        obs = r["observed"]
        gc = r["go_live_criteria"]

        print(f"\n[{key.upper()} signals — {r['n_trades']} trades]")
        print(f"  Sharpe:       {obs['sharpe']:.3f}  CI: [{ci['sharpe']['p2.5']:.3f}, {ci['sharpe']['p97.5']:.3f}]")
        print(f"  Win rate:     {obs['win_rate']:.1%}  CI: [{ci['win_rate']['p2.5']:.1%}, {ci['win_rate']['p97.5']:.1%}]")
        print(f"  Max DD:       {obs['max_dd_vnd']:,.0f}  CI: [{ci['max_dd_vnd']['p2.5']:,.0f}, {ci['max_dd_vnd']['p97.5']:,.0f}] VND")
        print(f"\n  Go-live: {gc['result']}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "methodology": f"Block bootstrap, n={N_BOOTSTRAP}, block_size={BLOCK_SIZE} trades",
        "results": results,
    }

    date_str = datetime.now().strftime("%Y%m%d")
    out = REPORTS_DIR / f"bootstrap_{date_str}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved → {out}")

    return report


if __name__ == "__main__":
    main()
