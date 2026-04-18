"""
[P2.6] sizing.py — Position sizing và portfolio constraints cho VN30 signal.

Rules:
  - MAX_POSITIONS: tối đa 5 vị thế đồng thời (diversification)
  - BASE_CAPITAL: 10 triệu VND mỗi lệnh (equal weight baseline)
  - Confidence-scaled: capital tỷ lệ với confidence (Kelly-inspired, capped)
  - Sector concentration limit: tối đa 2 cổ phiếu cùng sector
  - No double-buy: không mua lại nếu đã có vị thế mở

Usage:
  from portfolio.sizing import PortfolioSizer
  sizer = PortfolioSizer(total_capital=50_000_000)
  selected = sizer.select_trades(df_signals)
  print(selected[["ticker", "signal", "confidence", "position_size_vnd"]])
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

ROOT = Path(__file__).parent.parent

# Portfolio constraints
MAX_POSITIONS = 5
BASE_CAPITAL = 10_000_000      # VND per position (equal weight)
MIN_CONFIDENCE = 0.60          # must have at least 60% confidence
MAX_CONFIDENCE_SCALE = 1.5     # max multiplier for high-confidence bets
MAX_SAME_SECTOR = 2            # max stocks from same sector

# VN30 sector mapping (simplified)
SECTOR_MAP = {
    "VCB": "bank", "BID": "bank", "CTG": "bank", "MBB": "bank",
    "TCB": "bank", "VPB": "bank", "HDB": "bank", "STB": "bank",
    "ACB": "bank", "LPB": "bank",
    "VIC": "realestate", "VHM": "realestate", "NVL": "realestate",
    "PDR": "realestate",
    "HPG": "steel", "HSG": "steel",
    "FPT": "tech",
    "VNM": "consumer", "MSN": "consumer", "MWG": "consumer",
    "SAB": "consumer",
    "GAS": "energy", "PLX": "energy", "POW": "energy",
    "VJC": "aviation", "HVN": "aviation",
    "SSI": "securities", "VND": "securities", "HCM": "securities",
    "BCM": "industrial", "REE": "industrial",
}


class PortfolioSizer:
    def __init__(
        self,
        total_capital: float = BASE_CAPITAL * MAX_POSITIONS,
        max_positions: int = MAX_POSITIONS,
        base_capital: float = BASE_CAPITAL,
        existing_positions: list[str] | None = None,
    ):
        self.total_capital = total_capital
        self.max_positions = max_positions
        self.base_capital = base_capital
        self.existing_positions = set(existing_positions or [])

    def _get_sector(self, ticker: str) -> str:
        return SECTOR_MAP.get(ticker, "other")

    def _confidence_scale(self, confidence: float) -> float:
        """Scale capital by confidence: 60% conf → 1.0x, 80%+ → 1.5x."""
        if confidence < MIN_CONFIDENCE:
            return 0.0
        # Linear scale from 1.0 at 60% to MAX_CONFIDENCE_SCALE at 90%+
        scale = 1.0 + (confidence - MIN_CONFIDENCE) / (0.90 - MIN_CONFIDENCE) * (MAX_CONFIDENCE_SCALE - 1.0)
        return min(scale, MAX_CONFIDENCE_SCALE)

    def select_trades(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Select and size positions from signal DataFrame.

        Returns DataFrame with added columns:
          - position_size_vnd: capital to deploy (0 if filtered out)
          - position_reason: why selected or filtered
          - sector: sector classification
        """
        df = signals.copy()

        # Only consider BUY signals above threshold
        buy_mask = (df["signal"] == "BUY") & (df["confidence"] >= MIN_CONFIDENCE)
        df["sector"] = df["ticker"].map(lambda t: self._get_sector(t))
        df["position_size_vnd"] = 0.0
        df["position_reason"] = "HOLD — no signal"

        candidates = df[buy_mask].sort_values("confidence", ascending=False).copy()

        selected_tickers = []
        sector_counts: dict[str, int] = {}
        remaining_capital = self.total_capital

        for idx, row in candidates.iterrows():
            ticker = row["ticker"]
            sector = row["sector"]
            conf = row["confidence"]

            # Filter: already have position
            if ticker in self.existing_positions:
                df.at[idx, "position_reason"] = "Filtered — existing position"
                continue

            # Filter: max positions reached
            if len(selected_tickers) >= self.max_positions:
                df.at[idx, "position_reason"] = f"Filtered — max {self.max_positions} positions reached"
                continue

            # Filter: sector concentration
            if sector_counts.get(sector, 0) >= MAX_SAME_SECTOR:
                df.at[idx, "position_reason"] = f"Filtered — sector concentration ({sector} limit {MAX_SAME_SECTOR})"
                continue

            # Calculate position size
            scale = self._confidence_scale(conf)
            position_size = min(self.base_capital * scale, remaining_capital)
            if position_size <= 0:
                df.at[idx, "position_reason"] = "Filtered — insufficient capital"
                continue

            df.at[idx, "position_size_vnd"] = round(position_size, 0)
            df.at[idx, "position_reason"] = (
                f"Selected — {conf:.0%} conf, {scale:.2f}x size, sector={sector}"
            )
            selected_tickers.append(ticker)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            remaining_capital -= position_size

        # Mark HOLD rows
        hold_mask = df["signal"] == "HOLD"
        df.loc[hold_mask & (df["position_reason"] == "HOLD — no signal"), "position_reason"] = "HOLD — signal"

        return df

    def summary(self, sized_df: pd.DataFrame) -> dict:
        selected = sized_df[sized_df["position_size_vnd"] > 0]
        total_deployed = selected["position_size_vnd"].sum()
        return {
            "date": str(date.today()),
            "n_positions": len(selected),
            "max_positions": self.max_positions,
            "total_capital": self.total_capital,
            "deployed_vnd": round(total_deployed, 0),
            "cash_remaining_vnd": round(self.total_capital - total_deployed, 0),
            "utilization_pct": round(total_deployed / self.total_capital * 100, 1) if self.total_capital > 0 else 0,
            "positions": selected[["ticker", "confidence", "position_size_vnd", "sector", "position_reason"]].to_dict("records"),
        }


def apply_position_sizing(signals: pd.DataFrame, total_capital: float = BASE_CAPITAL * MAX_POSITIONS) -> pd.DataFrame:
    """Convenience wrapper: apply portfolio sizer to signal DataFrame."""
    sizer = PortfolioSizer(total_capital=total_capital)
    return sizer.select_trades(signals)


if __name__ == "__main__":
    # Demo: load latest signals and apply sizing
    signals_dir = ROOT / "signals"
    files = sorted(signals_dir.glob("*.csv"))
    if not files:
        print("No signal files found. Run src/signal_generator.py first.")
    else:
        latest = pd.read_csv(files[-1])
        print(f"Loading signals from {files[-1].name}")
        print(f"Total signals: {len(latest)}, BUY: {(latest['signal']=='BUY').sum()}")

        sizer = PortfolioSizer(total_capital=50_000_000)
        sized = sizer.select_trades(latest)
        summary = sizer.summary(sized)

        print(f"\n{'='*55}")
        print(f"  PORTFOLIO ALLOCATION — {summary['date']}")
        print(f"{'='*55}")
        print(f"  Capital:    {summary['total_capital']:>15,.0f} VND")
        print(f"  Deployed:   {summary['deployed_vnd']:>15,.0f} VND ({summary['utilization_pct']}%)")
        print(f"  Cash:       {summary['cash_remaining_vnd']:>15,.0f} VND")
        print(f"  Positions:  {summary['n_positions']}/{summary['max_positions']}")

        if summary["positions"]:
            print(f"\n  {'Ticker':<8} {'Conf':>6} {'Size (VND)':>15} {'Sector':<14}")
            print(f"  {'─'*50}")
            for p in summary["positions"]:
                print(f"  {p['ticker']:<8} {p['confidence']:>6.1%} {p['position_size_vnd']:>15,.0f}  {p['sector']}")
        else:
            print("\n  No positions selected today.")

        # Show filtered out
        filtered = sized[(sized["signal"] == "BUY") & (sized["position_size_vnd"] == 0)]
        if not filtered.empty:
            print(f"\n  Filtered out ({len(filtered)} BUY signals):")
            for _, r in filtered.iterrows():
                print(f"    {r['ticker']} — {r['position_reason']}")
