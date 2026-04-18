"""
features.py — Tính technical indicators + market context + target label
Target: return 5 ngày tới > threshold → Buy (1), < -threshold → Sell (-1), else Hold (0)
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path
from src.fetch import load_all, load_index
from src.sector import build_sector_returns, add_sector_features, SECTOR_FEATURE_COLS
from src.earnings import add_earnings_features, EARNINGS_FEATURE_COLS
# CEILING_FLOOR_COLS defined below in this file

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

FORWARD_DAYS = 5       # swing: dự đoán return sau 5 ngày
BUY_THRESHOLD = 0.03   # +3% → Buy
SELL_THRESHOLD = -0.02  # -2% → Sell


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Returns ---
    for n in [1, 3, 5, 10, 20]:
        df[f"ret_{n}d"] = df["close"].pct_change(n)

    # --- Moving Averages ---
    for n in [5, 10, 20, 50]:
        df[f"ma{n}"] = df["close"].rolling(n).mean()
        df[f"close_vs_ma{n}"] = df["close"] / df[f"ma{n}"] - 1

    # --- Trend strength ---
    df["ma20_slope"] = df["ma20"].pct_change(5)

    # --- Volatility ---
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr14_pct"] = df["atr14"] / df["close"]
    df["bb_width"] = ta.bbands(df["close"], length=20)["BBB_20_2.0_2.0"]

    # --- Momentum ---
    df["rsi14"] = ta.rsi(df["close"], length=14)
    df["rsi7"] = ta.rsi(df["close"], length=7)
    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]

    # --- Volume ---
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    df["obv"] = ta.obv(df["close"], df["volume"])
    df["obv_ma10"] = df["obv"].rolling(10).mean()
    df["obv_vs_ma"] = df["obv"] / df["obv_ma10"] - 1

    # --- Price position ---
    rolling_high = df["high"].rolling(20).max()
    rolling_low = df["low"].rolling(20).min()
    df["price_pct_range20"] = (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)

    # --- Candle body ---
    df["body_size"] = (df["close"] - df["open"]) / df["open"]
    df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["open"]
    df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["open"]

    return df


def build_market_context() -> pd.DataFrame:
    """
    Tính features từ VNINDEX và VN30 index.
    Returns DataFrame indexed by date với các market-wide features.
    """
    vni = load_index("VNINDEX")
    vn30 = load_index("VN30")

    ctx = pd.DataFrame(index=vni.index)

    # VNINDEX returns
    for n in [1, 3, 5, 10, 20]:
        ctx[f"vni_ret_{n}d"] = vni["close"].pct_change(n)

    # VNINDEX MAs và position
    for n in [10, 20, 50]:
        ma = vni["close"].rolling(n).mean()
        ctx[f"vni_vs_ma{n}"] = vni["close"] / ma - 1

    # VNINDEX trend direction (MA20 slope)
    ctx["vni_ma20_slope"] = vni["close"].rolling(20).mean().pct_change(5)

    # VNINDEX RSI
    ctx["vni_rsi14"] = ta.rsi(vni["close"], length=14)

    # VNINDEX volatility (rolling std of returns)
    ctx["vni_vol10"] = vni["close"].pct_change().rolling(10).std()
    ctx["vni_vol20"] = vni["close"].pct_change().rolling(20).std()

    # VNINDEX regime: bull/bear/neutral (based on MA cross)
    ma20 = vni["close"].rolling(20).mean()
    ma50 = vni["close"].rolling(50).mean()
    ctx["vni_bull_regime"] = (ma20 > ma50).astype(int)

    # VN30 index returns (thường lead VNINDEX)
    for n in [1, 3, 5]:
        ctx[f"vn30_ret_{n}d"] = vn30["close"].pct_change(n)

    # VN30 vs VNINDEX divergence (sector rotation signal)
    ctx["vn30_vs_vni"] = vn30["close"].pct_change(5) - vni["close"].pct_change(5)

    # Market breadth proxy: VNINDEX volume trend
    ctx["vni_vol_ratio"] = vni["volume"] / vni["volume"].rolling(20).mean()

    return ctx


def add_market_context(df: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Merge market context vào stock dataframe theo ngày."""
    ctx_cols = [c for c in ctx.columns]
    merged = df.join(ctx[ctx_cols], how="left")
    return merged


def add_relative_strength(df: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """
    Stock returns so với VNINDEX — đo lường strength của cổ phiếu vs thị trường.
    """
    df = df.copy()
    for n in [5, 10, 20]:
        stock_ret = df["close"].pct_change(n)
        vni_ret = ctx["vni_ret_5d"] if n == 5 else ctx[f"vni_ret_{n}d"]
        df[f"rs_vs_vni_{n}d"] = stock_ret - vni_ret.reindex(df.index)
    return df


def add_target(df: pd.DataFrame, forward: int = FORWARD_DAYS) -> pd.DataFrame:
    df = df.copy()
    future_return = df["close"].shift(-forward) / df["close"] - 1
    df["target"] = 0  # Hold
    df.loc[future_return >= BUY_THRESHOLD, "target"] = 1   # Buy
    df.loc[future_return <= SELL_THRESHOLD, "target"] = -1  # Sell
    df["future_return"] = future_return
    return df


MARKET_CONTEXT_COLS = [
    # Regime & trend — low noise, high signal
    "vni_bull_regime",     # MA20 > MA50: bull/bear phase
    "vni_vs_ma50",         # distance từ long-term MA → overbought/oversold thị trường
    "vni_ma20_slope",      # hướng trend ngắn hạn của thị trường
    # Volatility regime — risk-on / risk-off
    "vni_vol20",           # market volatility: cao → uncertain, thấp → trending
    # Momentum
    "vni_rsi14",           # market momentum
    "vni_ret_5d",          # market return ngắn hạn (không cần 1d/3d/10d/20d — correlated)
    # Rotation & relative
    "vn30_vs_vni",         # large-cap vs broad market divergence
    "vni_vol_ratio",       # volume surge/dry-up
]

RELATIVE_STRENGTH_COLS = [
    "rs_vs_vni_5d", "rs_vs_vni_10d", "rs_vs_vni_20d",
]

# ±7% band trên HSX (tất cả VN30)
CEILING_THRESH = 0.068
FLOOR_THRESH = -0.068


def _streak(arr: np.ndarray) -> np.ndarray:
    """Đếm số ngày liên tiếp True kết thúc tại mỗi vị trí."""
    out = np.zeros(len(arr), dtype=float)
    for i in range(len(arr)):
        if arr[i]:
            out[i] = out[i - 1] + 1 if i > 0 else 1
        else:
            out[i] = 0
    return out


def add_ceiling_floor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm features về chuỗi trần/sàn liên tiếp — đặc thù TTCK Việt Nam.
    Khi giá chạm trần/sàn liên tục → tín hiệu breakout hoặc bán tháo.
    """
    df = df.copy()
    daily_ret = df["close"].pct_change(1)

    is_ceil = (daily_ret >= CEILING_THRESH).astype(int)
    is_floor = (daily_ret <= FLOOR_THRESH).astype(int)

    # Số ngày trần/sàn trong cửa sổ ngắn
    df["ceil_days_5d"] = is_ceil.rolling(5).sum()
    df["floor_days_5d"] = is_floor.rolling(5).sum()
    df["ceil_days_10d"] = is_ceil.rolling(10).sum()
    df["floor_days_10d"] = is_floor.rolling(10).sum()

    # Chuỗi liên tiếp kết thúc hôm nay
    df["ceil_streak"] = _streak(is_ceil.values)
    df["floor_streak"] = _streak(is_floor.values)

    return df


CEILING_FLOOR_COLS = [
    "ceil_days_5d", "floor_days_5d",
    "ceil_days_10d", "floor_days_10d",
    "ceil_streak", "floor_streak",
]


STOCK_FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "close_vs_ma5", "close_vs_ma10", "close_vs_ma20", "close_vs_ma50",
    "ma20_slope",
    "atr14_pct", "bb_width",
    "rsi14", "rsi7",
    "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d",
    "vol_ratio", "obv_vs_ma",
    "price_pct_range20",
    "body_size", "upper_shadow", "lower_shadow",
]

FEATURE_COLS = STOCK_FEATURE_COLS + MARKET_CONTEXT_COLS + RELATIVE_STRENGTH_COLS + SECTOR_FEATURE_COLS + EARNINGS_FEATURE_COLS + CEILING_FLOOR_COLS


def build_dataset(tickers: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    if tickers is None:
        tickers = load_all()

    print("Building market context (VNINDEX + VN30)...")
    ctx = build_market_context()

    print("Building sector returns...")
    sector_returns = build_sector_returns(tickers)

    frames = []
    for ticker, df in tickers.items():
        df = add_indicators(df)
        df = add_market_context(df, ctx)
        df = add_relative_strength(df, ctx)
        df = add_sector_features(df, ticker, sector_returns)
        df = add_earnings_features(df, ticker)
        df = add_ceiling_floor_features(df)
        df = add_target(df)
        df["ticker"] = ticker
        frames.append(df)

    combined = pd.concat(frames)
    combined = combined.dropna(subset=FEATURE_COLS + ["target"])
    combined = combined.sort_index()
    return combined


def save_features(df: pd.DataFrame, path: Path | None = None) -> Path:
    if path is None:
        path = FEATURES_DIR / "features.parquet"
    df.to_parquet(path)
    print(f"✅ Features saved → {path}  ({len(df):,} rows, {df['ticker'].nunique()} tickers)")
    return path


if __name__ == "__main__":
    print("Building features for all VN30...")
    df = build_dataset()
    save_features(df)
    print(f"\nTotal features: {len(FEATURE_COLS)}")
    print(f"  Stock features:   {len(STOCK_FEATURE_COLS)}")
    print(f"  Market context:   {len(MARKET_CONTEXT_COLS)}")
    print(f"  Relative strength:{len(RELATIVE_STRENGTH_COLS)}")
    print("\nTarget distribution:")
    print(df["target"].value_counts().rename({-1: "Sell", 0: "Hold", 1: "Buy"}))
