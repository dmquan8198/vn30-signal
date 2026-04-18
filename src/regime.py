"""
[P3.2] regime.py — Market regime detection cho VN30 system.

4-state regime detector dựa trên VNINDEX:
  0: BEAR      — MA20 < MA50, vol cao, xu hướng giảm
  1: SIDEWAYS  — MA gần nhau, vol thấp, không rõ hướng
  2: BULL      — MA20 > MA50, vol thường, xu hướng tăng
  3: BREAKOUT  — BULL + volume surge (vol_ratio > 1.5x)

Thay thế/bổ sung cho simple binary vni_bull_regime.
Dùng để điều chỉnh confidence threshold và position sizing.

Features bổ sung:
  - regime_state: 0-3 (ordinal)
  - regime_bull: 1 nếu BULL hoặc BREAKOUT
  - regime_volatility: rolling 20d vol của VNINDEX (z-score)
  - trend_strength: distance của MA20 so với MA50 (normalized)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

ROOT = Path(__file__).parent.parent

REGIME_FEATURE_COLS = [
    "regime_state",
    "regime_bull",
    "regime_volatility_z",
    "trend_strength",
]


def detect_regime(vnindex: pd.DataFrame) -> pd.DataFrame:
    """
    Tính 4-state market regime từ VNINDEX data.

    Input: DataFrame với 'close' và 'volume' columns, indexed by date.
    Output: DataFrame indexed by date với REGIME_FEATURE_COLS.
    """
    df = vnindex.copy()

    # Moving averages
    ma20 = df["close"].rolling(20).mean()
    ma50 = df["close"].rolling(50).mean()

    # Trend direction: MA20 vs MA50
    ma_cross_bull = (ma20 > ma50).astype(int)

    # Trend strength: normalized distance between MAs
    trend_strength = (ma20 - ma50) / ma50

    # Volatility: rolling 20d std of returns, z-scored over 252d window
    daily_ret = df["close"].pct_change()
    vol20 = daily_ret.rolling(20).std()
    vol_mean = vol20.rolling(252).mean()
    vol_std = vol20.rolling(252).std().replace(0, 1e-6)
    vol_z = (vol20 - vol_mean) / vol_std

    # Volume surge
    if "volume" in df.columns:
        vol_ratio = df["volume"] / df["volume"].rolling(20).mean()
    else:
        vol_ratio = pd.Series(1.0, index=df.index)

    # 4-state classification
    def classify_regime(row):
        bull = row["ma_cross"]
        vol_surge = row["vol_ratio"] > 1.5
        vol_z_val = row["vol_z"]

        if not bull:
            return 0  # BEAR
        elif bull and vol_surge:
            return 3  # BREAKOUT
        elif bull:
            return 2  # BULL
        else:
            return 1  # SIDEWAYS

    regime_df = pd.DataFrame({
        "ma_cross": ma_cross_bull,
        "vol_ratio": vol_ratio,
        "vol_z": vol_z,
    }, index=df.index)

    regime_df["regime_state"] = regime_df.apply(classify_regime, axis=1).astype(float)

    # Simplify sideways classification post-hoc
    # If MA cross is ambiguous (|trend_strength| < 0.005) → SIDEWAYS
    sideways_mask = trend_strength.abs() < 0.005
    regime_df.loc[sideways_mask, "regime_state"] = 1.0

    result = pd.DataFrame(index=df.index)
    result["regime_state"] = regime_df["regime_state"]
    result["regime_bull"] = (result["regime_state"] >= 2).astype(float)
    result["regime_volatility_z"] = vol_z
    result["trend_strength"] = trend_strength

    return result


def add_regime_features(df: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    """Merge regime features vào stock DataFrame theo ngày."""
    df = df.copy()
    regime_aligned = regime.reindex(df.index, method="ffill")
    for col in REGIME_FEATURE_COLS:
        if col in regime_aligned.columns:
            df[col] = regime_aligned[col].values
        else:
            df[col] = 0.0
    return df


def get_current_regime(regime: pd.DataFrame) -> dict:
    """Trả về regime hiện tại (dòng cuối)."""
    if regime.empty:
        return {"regime_state": 2, "regime_name": "BULL", "regime_bull": 1}
    last = regime.iloc[-1]
    names = {0: "BEAR", 1: "SIDEWAYS", 2: "BULL", 3: "BREAKOUT"}
    state = int(last.get("regime_state", 2))
    return {
        "regime_state": state,
        "regime_name": names.get(state, "UNKNOWN"),
        "regime_bull": int(last.get("regime_bull", 1)),
        "regime_volatility_z": round(float(last.get("regime_volatility_z", 0)), 3),
        "trend_strength": round(float(last.get("trend_strength", 0)), 4),
    }


if __name__ == "__main__":
    from src.fetch import load_index
    print("=== Regime Detection Demo ===\n")
    vni = load_index("VNINDEX")
    regime = detect_regime(vni)
    current = get_current_regime(regime)

    state_names = {0: "BEAR 🔴", 1: "SIDEWAYS ⚪", 2: "BULL 🟢", 3: "BREAKOUT 🚀"}
    print(f"  Current regime: {state_names[current['regime_state']]}")
    print(f"  Trend strength: {current['trend_strength']:+.4f}")
    print(f"  Vol z-score:    {current['regime_volatility_z']:+.3f}")

    print(f"\n  Last 30 days:")
    tail = regime.tail(30)
    counts = tail["regime_state"].value_counts().sort_index()
    for state, cnt in counts.items():
        print(f"    {state_names[int(state)]}: {int(cnt)} days")
