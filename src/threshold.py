"""
[P3.5] threshold.py — Dynamic confidence threshold dựa trên điều kiện thị trường.

Logic:
  Base threshold: 0.60

  Điều chỉnh theo regime:
    BREAKOUT (3): -0.03 → 0.57  (momentum mạnh, mở cửa thêm)
    BULL (2):     ±0.00 → 0.60  (bình thường)
    SIDEWAYS (1): +0.05 → 0.65  (không rõ hướng, thận trọng hơn)
    BEAR (0):     +0.10 → 0.70  (thị trường giảm, chỉ vào khi rất chắc)

  Điều chỉnh theo VIX:
    VIX > 30 (fear zone): +0.05 (risk-off globally)
    VIX < 15 (complacent): -0.02 (low vol, momentum works better)

  Điều chỉnh theo earnings season:
    Trong mùa KQKD: +0.03 (tăng uncertainty)

  Clamp: [0.50, 0.80]
"""

import pandas as pd
from src.regime import get_current_regime
from src.earnings import in_earnings_window

BASE_THRESHOLD = 0.60
MIN_THRESHOLD = 0.50
MAX_THRESHOLD = 0.80

REGIME_ADJUSTMENTS = {
    0: +0.10,   # BEAR
    1: +0.05,   # SIDEWAYS
    2: +0.00,   # BULL
    3: -0.03,   # BREAKOUT
}


def compute_dynamic_threshold(
    regime: dict | None = None,
    vix_level: float | None = None,
    in_earnings: bool = False,
    circuit_breaker_status: str = "CLOSED",
) -> dict:
    """
    Compute dynamic confidence threshold.

    Args:
        regime: dict from get_current_regime() with 'regime_state'
        vix_level: current VIX value (absolute, e.g. 18.5)
        in_earnings: whether today is in earnings season
        circuit_breaker_status: CLOSED / HALF-OPEN / OPEN

    Returns:
        dict with threshold, components, and reasoning
    """
    threshold = BASE_THRESHOLD
    components = {"base": BASE_THRESHOLD}

    # Circuit breaker override
    if circuit_breaker_status == "OPEN":
        return {
            "threshold": MAX_THRESHOLD,
            "reason": "Circuit breaker OPEN — max threshold applied",
            "components": {"base": BASE_THRESHOLD, "circuit_breaker": MAX_THRESHOLD - BASE_THRESHOLD},
        }
    elif circuit_breaker_status == "HALF-OPEN":
        cb_adj = +0.10
        threshold += cb_adj
        components["circuit_breaker_half_open"] = cb_adj

    # Regime adjustment
    regime_state = 2  # default BULL
    if regime:
        regime_state = regime.get("regime_state", 2)
    regime_adj = REGIME_ADJUSTMENTS.get(regime_state, 0)
    threshold += regime_adj
    if regime_adj != 0:
        components[f"regime_{regime_state}"] = regime_adj

    # VIX adjustment
    if vix_level is not None:
        if vix_level > 30:
            vix_adj = +0.05
            threshold += vix_adj
            components["vix_high"] = vix_adj
        elif vix_level < 15:
            vix_adj = -0.02
            threshold += vix_adj
            components["vix_low"] = vix_adj

    # Earnings season
    if in_earnings:
        earn_adj = +0.03
        threshold += earn_adj
        components["earnings_season"] = earn_adj

    threshold = round(max(MIN_THRESHOLD, min(MAX_THRESHOLD, threshold)), 3)

    regime_names = {0: "BEAR", 1: "SIDEWAYS", 2: "BULL", 3: "BREAKOUT"}
    reason_parts = [f"regime={regime_names.get(regime_state, '?')}"]
    if vix_level:
        reason_parts.append(f"VIX={vix_level:.1f}")
    if in_earnings:
        reason_parts.append("earnings_season")

    return {
        "threshold": threshold,
        "reason": f"Dynamic threshold: {', '.join(reason_parts)}",
        "components": components,
    }


def get_today_threshold(
    regime: dict | None = None,
    macro_features: "pd.DataFrame | None" = None,
    circuit_breaker_status: str = "CLOSED",
) -> float:
    """
    Convenience function: compute today's threshold from available data.
    Returns just the threshold float.
    """
    today = pd.Timestamp.today()
    in_earnings = bool(in_earnings_window(today))

    vix_level = None
    if macro_features is not None and not macro_features.empty:
        # vix_level was stored normalized (vix/40), convert back
        vix_col = "vix_level"
        if vix_col in macro_features.columns:
            last_vix_norm = macro_features[vix_col].iloc[-1]
            vix_level = float(last_vix_norm) * 40  # denormalize

    result = compute_dynamic_threshold(
        regime=regime,
        vix_level=vix_level,
        in_earnings=in_earnings,
        circuit_breaker_status=circuit_breaker_status,
    )
    return result["threshold"]


if __name__ == "__main__":
    print("=== Dynamic Threshold Demo ===\n")
    from src.fetch import load_index
    from src.regime import detect_regime

    vni = load_index("VNINDEX")
    regime_df = detect_regime(vni)
    regime = get_current_regime(regime_df)

    today = pd.Timestamp.today()
    in_earnings = bool(in_earnings_window(today))

    scenarios = [
        {"regime": {"regime_state": 0}, "vix_level": 32, "in_earnings": False, "label": "BEAR + VIX spike"},
        {"regime": {"regime_state": 1}, "vix_level": 20, "in_earnings": True,  "label": "SIDEWAYS + earnings"},
        {"regime": {"regime_state": 2}, "vix_level": 18, "in_earnings": False, "label": "BULL normal"},
        {"regime": {"regime_state": 3}, "vix_level": 14, "in_earnings": False, "label": "BREAKOUT low vol"},
    ]

    print(f"  {'Scenario':<30} {'Threshold':>10}")
    print(f"  {'─'*42}")
    for s in scenarios:
        r = compute_dynamic_threshold(s["regime"], s["vix_level"], s["in_earnings"])
        print(f"  {s['label']:<30} {r['threshold']:>9.3f}  ({r['reason']})")

    print(f"\n  Today's regime: {regime['regime_name']}")
    print(f"  Earnings season: {in_earnings}")
    today_thresh = get_today_threshold(regime=regime, circuit_breaker_status="CLOSED")
    print(f"  → Threshold for today: {today_thresh:.3f}")
