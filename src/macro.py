"""
[P3.1] macro.py — Macro features: USD/VND, Brent oil, VIX-proxy.

Nguồn: Yahoo Finance (yfinance) — data miễn phí, daily OHLCV.
  - USDVND=X: Tỷ giá USD/VND (proxy cho áp lực ngoại hối)
  - BZ=F: Brent crude oil (ảnh hưởng GAS, PLX, HPG)
  - ^VIX: CBOE VIX (global risk appetite proxy)
  - ^GSPC: S&P 500 (global sentiment cho dòng vốn ngoại)

Cache: data/macro/{symbol}.parquet
Refresh: tự động nếu cache > 1 ngày cũ

Features thêm vào model:
  - usdvnd_ret5d: VND mất giá → capital outflow từ TTCK VN
  - oil_ret5d: giá dầu tác động chi phí ngành sản xuất/năng lượng
  - vix_level: risk-off khi VIX cao → bán tháo emerging markets
  - vix_ret5d: spike VIX → tháo chạy khỏi rủi ro
  - sp500_ret5d: S&P 500 xu hướng dẫn dắt dòng tiền ngoại vào VN
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).parent.parent
MACRO_DIR = ROOT / "data" / "macro"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_HOURS = 20  # refresh nếu cache > 20h cũ (1 phiên giao dịch)

SYMBOLS = {
    "USDVND": "USDVND=X",
    "OIL": "BZ=F",
    "VIX": "^VIX",
    "SP500": "^GSPC",
}

MACRO_FEATURE_COLS = [
    "usdvnd_ret5d",   # VND depreciation
    "oil_ret5d",      # oil price momentum
    "vix_level",      # absolute fear index
    "vix_ret5d",      # VIX change (spike = risk-off)
    "sp500_ret5d",    # S&P 500 momentum → foreign capital flow proxy
]


def _cache_path(key: str) -> Path:
    return MACRO_DIR / f"{key}.parquet"


def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(hours=CACHE_TTL_HOURS)


def fetch_macro(start: str = "2020-01-01", force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Fetch all macro symbols from Yahoo Finance, cache locally.
    Returns dict of {key: DataFrame with date index and 'close' column}.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  ⚠️  yfinance not installed — macro features unavailable")
        print("       Run: pip install yfinance")
        return {}

    results = {}
    for key, symbol in SYMBOLS.items():
        cache = _cache_path(key)
        if not force_refresh and not _is_stale(cache):
            try:
                results[key] = pd.read_parquet(cache)
                continue
            except Exception:
                pass

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, auto_adjust=True)
            if df.empty:
                print(f"  ⚠️  No data for {symbol} ({key})")
                continue
            df.index = pd.to_datetime(df.index.tz_localize(None) if df.index.tz else df.index)
            df = df[["Close"]].rename(columns={"Close": "close"})
            df.to_parquet(cache)
            results[key] = df
        except Exception as e:
            print(f"  ⚠️  Macro fetch error {key} ({symbol}): {e}")
            if cache.exists():
                results[key] = pd.read_parquet(cache)

    return results


def build_macro_features(macro_data: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """
    Build macro feature DataFrame indexed by date.
    Returns DataFrame with MACRO_FEATURE_COLS as columns.
    """
    if macro_data is None:
        macro_data = fetch_macro()

    if not macro_data:
        return pd.DataFrame(columns=MACRO_FEATURE_COLS)

    frames = {}
    for key, df in macro_data.items():
        if df.empty:
            continue
        frames[key] = df["close"]

    if not frames:
        return pd.DataFrame(columns=MACRO_FEATURE_COLS)

    combined = pd.DataFrame(frames)
    combined = combined.ffill()  # forward-fill weekends/holidays

    result = pd.DataFrame(index=combined.index)

    if "USDVND" in combined.columns:
        result["usdvnd_ret5d"] = combined["USDVND"].pct_change(5)

    if "OIL" in combined.columns:
        result["oil_ret5d"] = combined["OIL"].pct_change(5)

    if "VIX" in combined.columns:
        result["vix_level"] = combined["VIX"] / 40.0        # normalize: VIX 40 = 1.0
        result["vix_ret5d"] = combined["VIX"].pct_change(5)

    if "SP500" in combined.columns:
        result["sp500_ret5d"] = combined["SP500"].pct_change(5)

    # Fill missing cols with 0 (neutral)
    for col in MACRO_FEATURE_COLS:
        if col not in result.columns:
            result[col] = 0.0

    return result[MACRO_FEATURE_COLS]


def add_macro_features(df: pd.DataFrame, macro_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Merge macro features vào stock DataFrame theo ngày.
    Nếu không có macro data → điền 0 (neutral, không làm hỏng model).
    """
    df = df.copy()

    if macro_features is None or macro_features.empty:
        for col in MACRO_FEATURE_COLS:
            df[col] = 0.0
        return df

    # Align: macro data là daily, cần match với stock dates
    macro_aligned = macro_features.reindex(df.index, method="ffill")
    for col in MACRO_FEATURE_COLS:
        if col in macro_aligned.columns:
            df[col] = macro_aligned[col].values
        else:
            df[col] = 0.0

    return df


if __name__ == "__main__":
    print("=== Macro Feature Demo ===\n")
    print("Fetching macro data from Yahoo Finance...")
    data = fetch_macro()

    if not data:
        print("No macro data available.")
    else:
        for key, df in data.items():
            print(f"  {key:>8}: {len(df)} rows  {df.index.min().date()} → {df.index.max().date()}")

        print("\nBuilding macro features...")
        features = build_macro_features(data)
        print(features.tail(5).to_string())
