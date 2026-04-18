"""
sector.py — Sector rotation features từ VN30 OHLCV data
Không cần API bên ngoài — tính từ data đã fetch
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Phân nhóm VN30 theo sector
SECTORS = {
    "banking":     ["ACB", "BID", "CTG", "HDB", "MBB", "SHB", "SSB", "STB", "TCB", "TPB", "VCB", "VIB", "VPB"],
    "realestate":  ["BCM", "VHM", "VIC", "VRE"],
    "energy":      ["GAS", "PLX", "POW"],
    "material":    ["GVR", "HPG"],
    "consumer":    ["MWG", "MSN", "SAB", "VNM"],
    "financial":   ["SSI"],
    "aviation":    ["VJC"],
    "tech":        ["FPT"],
}

TICKER_TO_SECTOR = {t: s for s, tickers in SECTORS.items() for t in tickers}


def build_sector_returns(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Tính sector average return mỗi ngày.
    Returns: DataFrame indexed by date, columns = sector names
    """
    sector_closes = {}
    for sector, tickers in SECTORS.items():
        available = [t for t in tickers if t in all_data]
        if not available:
            continue
        closes = pd.concat(
            [all_data[t]["close"].rename(t) for t in available], axis=1
        )
        # Normalize to base 100 at first date
        normed = closes / closes.iloc[0] * 100
        sector_closes[sector] = normed.mean(axis=1)

    df = pd.DataFrame(sector_closes)
    return df


def add_sector_features(stock_df: pd.DataFrame, ticker: str,
                        sector_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm sector features vào stock dataframe:
    - sector_ret_5d: return 5 ngày của sector chứa mã này
    - sector_ret_10d: return 10 ngày của sector
    - stock_vs_sector_5d: outperformance của stock so với sector
    - sector_rank_5d: xếp hạng return của stock trong sector (0=worst, 1=best)
    """
    df = stock_df.copy()
    sector = TICKER_TO_SECTOR.get(ticker)

    if sector is None or sector not in sector_returns.columns:
        for col in SECTOR_FEATURE_COLS:
            df[col] = 0.0
        return df

    sec = sector_returns[sector]

    for n in [5, 10]:
        sec_ret = sec.pct_change(n)
        df[f"sector_ret_{n}d"] = sec_ret.reindex(df.index)

    # Stock outperformance vs sector
    stock_ret_5d = df["close"].pct_change(5)
    df["stock_vs_sector_5d"] = stock_ret_5d - df["sector_ret_5d"]

    # Sector momentum: is sector trending up or down?
    sec_ma10 = sec.rolling(10).mean()
    sec_above_ma = (sec > sec_ma10).astype(int)
    df["sector_bull"] = sec_above_ma.reindex(df.index).fillna(0).astype(int)

    return df


SECTOR_FEATURE_COLS = [
    "sector_ret_5d",
    "sector_ret_10d",
    "stock_vs_sector_5d",
    "sector_bull",
]
