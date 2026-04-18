"""
fetch.py — Crawl lịch sử giá OHLCV cho 30 mã VN30
Lưu vào data/raw/<TICKER>.parquet
"""

import time
import pandas as pd
from pathlib import Path
from vnstock import Vnstock

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG",
    "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW",
    "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB",
    "VIC", "VJC", "VNM", "VPB", "VRE",
]

START_DATE = "2020-01-01"


def fetch_ticker(ticker: str, start: str = START_DATE) -> pd.DataFrame | None:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        stock = Vnstock().stock(symbol=ticker, source="KBS")
        df = stock.quote.history(start=start, end=today, interval="1D")
        if df is None or df.empty:
            print(f"  ⚠️  {ticker}: no data")
            return None
        df = df.rename(columns=str.lower)
        df["ticker"] = ticker
        # 'time' column → DatetimeIndex
        if "time" in df.columns:
            df = df.set_index("time")
        df.index = pd.to_datetime(df.index).normalize()  # strip time component
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return None


def fetch_all(tickers: list[str] = VN30, delay: float = 0.5):
    results = {}
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:02d}/{len(tickers)}] Fetching {ticker}...", end=" ")
        df = fetch_ticker(ticker)
        if df is not None:
            path = RAW_DIR / f"{ticker}.parquet"
            df.to_parquet(path)
            print(f"✅ {len(df)} rows → {path.name}")
            results[ticker] = df
        time.sleep(delay)
    print(f"\n✅ Done: {len(results)}/{len(tickers)} tickers saved to {RAW_DIR}")
    return results


def load_ticker(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{ticker}.parquet not found — run fetch_all() first")
    return pd.read_parquet(path)


def load_all() -> dict[str, pd.DataFrame]:
    files = sorted(RAW_DIR.glob("*.parquet"))
    return {f.stem: pd.read_parquet(f) for f in files}


INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDICES = ["VNINDEX", "VN30"]


def fetch_index(symbol: str, start: str = START_DATE) -> pd.DataFrame | None:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        stock = Vnstock().stock(symbol=symbol, source="KBS")
        df = stock.quote.history(start=start, end=today, interval="1D")
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)
        if "time" in df.columns:
            df = df.set_index("time")
        df.index = pd.to_datetime(df.index).normalize()
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"  ❌ {symbol}: {e}")
        return None


def fetch_indices(delay: float = 1.0):
    for symbol in INDICES:
        print(f"Fetching {symbol}...", end=" ")
        df = fetch_index(symbol)
        if df is not None:
            path = INDEX_DIR / f"{symbol}.parquet"
            df.to_parquet(path)
            print(f"✅ {len(df)} rows")
        time.sleep(delay)


def load_index(symbol: str) -> pd.DataFrame:
    path = INDEX_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{symbol}.parquet not found — run fetch_indices() first")
    return pd.read_parquet(path)


if __name__ == "__main__":
    fetch_all()
    fetch_indices()
