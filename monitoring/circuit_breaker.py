"""
[P2.3] circuit_breaker.py — Tự động tắt signal khi hiệu suất suy giảm.

Logic:
  - CLOSED (normal): signals run normally
  - OPEN (halted): tắt signal hoàn toàn, chỉ phát cảnh báo
  - HALF-OPEN: chạy signal nhưng giảm confidence threshold lên 0.70

Trigger conditions (bất kỳ 1 trong 3):
  1. Rolling 20-trade Sharpe < 0 (hiệu suất âm)
  2. Rolling 20-trade win rate < 40% (dưới mức flip coin)
  3. Max drawdown vượt 15% capital trong 30 ngày

State file: reports/circuit_breaker.json
Alert log:  reports/cb_alerts.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = REPORTS_DIR / "circuit_breaker.json"
ALERT_LOG = REPORTS_DIR / "cb_alerts.csv"

# Thresholds
ROLLING_WINDOW = 20          # trades
MIN_SHARPE = 0.0             # rolling Sharpe must stay above this
MIN_WIN_RATE = 0.40          # 40%
MAX_DD_PCT = 0.15            # 15% of capital
CAPITAL = 10_000_000
MAX_DD_VND = CAPITAL * MAX_DD_PCT

# Half-open confidence threshold (stricter than normal 0.60)
HALF_OPEN_THRESHOLD = 0.70


class CircuitBreaker:
    def __init__(self):
        self.state_file = STATE_FILE
        self._state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            "status": "CLOSED",
            "opened_at": None,
            "reason": None,
            "consecutive_bad_days": 0,
            "last_checked": None,
        }

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def _load_recent_trades(self, days: int = 30) -> pd.DataFrame:
        trades_path = ROOT / "backtest" / "trades.csv"
        if not trades_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(trades_path)
        df["date"] = pd.to_datetime(df["date"])
        cutoff = pd.Timestamp.now() - pd.DateOffset(days=days)
        return df[df["date"] >= cutoff].copy()

    def _compute_rolling_metrics(self, trades: pd.DataFrame) -> dict:
        if trades.empty or len(trades) < 5:
            return {"sharpe": None, "win_rate": None, "max_dd_vnd": None, "n_trades": 0}

        buy_trades = trades[trades["direction"] == "BUY"].copy()
        if buy_trades.empty:
            return {"sharpe": None, "win_rate": None, "max_dd_vnd": None, "n_trades": 0}

        buy_trades = buy_trades.sort_values("date")
        recent = buy_trades.tail(ROLLING_WINDOW)
        returns = recent["net_return"].values

        std = returns.std()
        sharpe = (returns.mean() / std) * np.sqrt(252 / 5) if std > 0 else 0.0
        win_rate = float((returns > 0).mean())

        pnl_cum = (CAPITAL * returns / 100).cumsum()
        running_max = np.maximum.accumulate(pnl_cum)
        max_dd = float((pnl_cum - running_max).min())

        return {
            "sharpe": round(sharpe, 3),
            "win_rate": round(win_rate, 3),
            "max_dd_vnd": round(max_dd, 0),
            "n_trades": len(recent),
        }

    def check(self) -> dict:
        """
        Run circuit breaker check. Returns status dict with:
          status: CLOSED | OPEN | HALF-OPEN
          reason: description
          metrics: rolling metrics
          confidence_threshold: current threshold to use for signal generation
        """
        trades = self._load_recent_trades(days=60)
        metrics = self._compute_rolling_metrics(trades)

        # Determine triggers
        triggers = []
        if metrics["sharpe"] is not None:
            if metrics["sharpe"] < MIN_SHARPE:
                triggers.append(f"Rolling Sharpe={metrics['sharpe']:.3f} < {MIN_SHARPE}")
            if metrics["win_rate"] < MIN_WIN_RATE:
                triggers.append(f"Win rate={metrics['win_rate']:.1%} < {MIN_WIN_RATE:.0%}")
            if metrics["max_dd_vnd"] < -MAX_DD_VND:
                triggers.append(f"Max DD={metrics['max_dd_vnd']:,.0f} VND > {MAX_DD_PCT:.0%} capital")

        now = datetime.now().isoformat()

        if triggers:
            prev_status = self._state.get("status", "CLOSED")
            if prev_status == "CLOSED":
                new_status = "OPEN"
            elif prev_status == "HALF-OPEN":
                new_status = "OPEN"
            else:
                new_status = "OPEN"

            self._state.update({
                "status": new_status,
                "opened_at": self._state.get("opened_at") or now,
                "reason": " | ".join(triggers),
                "last_checked": now,
                "consecutive_bad_days": self._state.get("consecutive_bad_days", 0) + 1,
            })
            self._log_alert(new_status, triggers, metrics)

        else:
            # No triggers — consider recovery
            prev_status = self._state.get("status", "CLOSED")
            if prev_status == "OPEN":
                # Move to HALF-OPEN for one period before fully closing
                new_status = "HALF-OPEN"
            else:
                new_status = "CLOSED"

            self._state.update({
                "status": new_status,
                "opened_at": self._state.get("opened_at") if new_status != "CLOSED" else None,
                "reason": "Recovery — metrics within bounds" if prev_status != "CLOSED" else None,
                "last_checked": now,
                "consecutive_bad_days": 0,
            })

        self._save_state()

        threshold = HALF_OPEN_THRESHOLD if self._state["status"] == "HALF-OPEN" else 0.60

        return {
            "status": self._state["status"],
            "reason": self._state.get("reason") or "All metrics within bounds",
            "metrics": metrics,
            "confidence_threshold": threshold,
            "checked_at": now,
        }

    def _log_alert(self, status: str, triggers: list, metrics: dict):
        row = pd.DataFrame([{
            "date": datetime.now().isoformat(),
            "status": status,
            "triggers": "; ".join(triggers),
            "sharpe": metrics.get("sharpe"),
            "win_rate": metrics.get("win_rate"),
            "max_dd_vnd": metrics.get("max_dd_vnd"),
            "n_trades": metrics.get("n_trades"),
        }])
        if ALERT_LOG.exists():
            existing = pd.read_csv(ALERT_LOG)
            row = pd.concat([existing, row], ignore_index=True)
        row.to_csv(ALERT_LOG, index=False)

    @property
    def status(self) -> str:
        return self._state.get("status", "CLOSED")

    @property
    def confidence_threshold(self) -> float:
        return HALF_OPEN_THRESHOLD if self.status == "HALF-OPEN" else 0.60

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN"


def main():
    cb = CircuitBreaker()
    result = cb.check()

    icons = {"CLOSED": "✅", "HALF-OPEN": "⚠️", "OPEN": "🚨"}
    icon = icons.get(result["status"], "?")

    print(f"\n{'='*50}")
    print(f"  CIRCUIT BREAKER — {result['status']}")
    print(f"{'='*50}")
    print(f"\n  {icon} Status: {result['status']}")
    print(f"  Reason:  {result['reason']}")

    m = result["metrics"]
    if m["n_trades"] > 0:
        print(f"\n  Rolling metrics (last {ROLLING_WINDOW} trades):")
        print(f"    Sharpe:    {m['sharpe']}")
        print(f"    Win rate:  {m['win_rate']:.1%}" if m['win_rate'] is not None else "    Win rate:  N/A")
        print(f"    Max DD:    {m['max_dd_vnd']:,.0f} VND" if m['max_dd_vnd'] is not None else "    Max DD:    N/A")
        print(f"    N trades:  {m['n_trades']}")
    else:
        print("\n  No recent trades found.")

    print(f"\n  Confidence threshold: {result['confidence_threshold']:.0%}")
    print(f"\n✅ State saved → {STATE_FILE}")


if __name__ == "__main__":
    main()
