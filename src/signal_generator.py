"""
signal_generator.py — Tạo tín hiệu BUY/HOLD cho ngày hôm nay
Chỉ dùng BUY signal (backtest xác nhận SELL không đáng tin cậy)
Chạy sau 15h mỗi ngày. Output: signals/YYYY-MM-DD.csv
"""

import time
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.fetch import fetch_ticker, fetch_indices, load_all, VN30
from src.features import (
    add_indicators, add_market_context, add_relative_strength,
    build_market_context, add_ceiling_floor_features, FEATURE_COLS
)
from src.sector import build_sector_returns, add_sector_features
from src.earnings import add_earnings_features
from src.macro import build_macro_features, add_macro_features
from src.regime import detect_regime, add_regime_features, get_current_regime
from src.model import ensemble_predict
from src.news import get_today_sentiment, apply_news_overlay, save_sentiment, save_articles
from src.live_overlay import apply_live_overlay
from src import tracker
from src.calibration import load_calibrator, calibrate_confidence

MODEL_DIR = Path(__file__).parent.parent / "models"
SIGNALS_DIR = Path(__file__).parent.parent / "signals"
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

HIGH_CONFIDENCE_THRESHOLD = 0.60


def load_models():
    import pickle
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_DIR / "xgb_model.json")
    lgb_booster = lgb.Booster(model_file=str(MODEL_DIR / "lgb_model.txt"))
    rf_path = MODEL_DIR / "rf_model.pkl"
    rf_model = None
    if rf_path.exists():
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
    classes = np.load(MODEL_DIR / "label_classes.npy")
    le = LabelEncoder()
    le.classes_ = classes
    return xgb_model, lgb_booster, le, rf_model


class LGBWrapper:
    """Wrap lgb.Booster thành sklearn-like interface."""
    def predict_proba(self, X):
        return lgb_booster_ref.predict(X)


def get_signal_today(refresh_data: bool = True, confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    xgb_model, lgb_booster, le, rf_model = load_models()
    calibrator = load_calibrator()  # None if not fitted yet

    class _LGBWrapper:
        def predict_proba(self, X):
            return lgb_booster.predict(X)

    lgb_model = _LGBWrapper()
    today = pd.Timestamp.today().normalize()

    if refresh_data:
        print("  Refreshing index data...")
        fetch_indices(delay=1.0)

    ctx = build_market_context()

    print("  Fetching macro features (USD/VND, oil, VIX)...")
    macro_features = build_macro_features()

    print("  Detecting market regime...")
    from src.fetch import load_index
    vni = load_index("VNINDEX")
    regime = detect_regime(vni)
    current_regime = get_current_regime(regime)
    regime_names = {0: "BEAR 🔴", 1: "SIDEWAYS ⚪", 2: "BULL 🟢", 3: "BREAKOUT 🚀"}
    print(f"    Regime: {regime_names[current_regime['regime_state']]}")

    # Build sector returns from cached data (no extra API calls)
    print("  Building sector returns...")
    all_data = load_all()
    sector_returns = build_sector_returns(all_data)

    results = []

    # 20 req/min limit → 3s delay an toàn; index đã dùng 2 req nên còn 18
    FETCH_DELAY = 3.5  # giây

    for i, ticker in enumerate(VN30):
        try:
            if refresh_data:
                if i > 0:
                    time.sleep(FETCH_DELAY)
                start = (today - pd.DateOffset(days=150)).strftime("%Y-%m-%d")
                df = fetch_ticker(ticker, start=start)
            else:
                from src.fetch import load_ticker
                df = load_ticker(ticker).tail(150)

            if df is None or len(df) < 60:
                continue

            df = add_indicators(df)
            df = add_market_context(df, ctx)
            df = add_relative_strength(df, ctx)
            df = add_sector_features(df, ticker, sector_returns)
            df = add_earnings_features(df, ticker)
            df = add_ceiling_floor_features(df)
            df = add_macro_features(df, macro_features)
            df = add_regime_features(df, regime)
            df = df.dropna(subset=FEATURE_COLS)

            if df.empty:
                continue

            latest = df.iloc[[-1]]
            latest_date = latest.index[0]

            X = latest[FEATURE_COLS].values
            preds, proba_avg = ensemble_predict(xgb_model, lgb_model, le, X, rf_model=rf_model)
            signal_val = preds[0]
            raw_confidence = float(proba_avg[0].max())
            confidence = float(calibrate_confidence(calibrator, raw_confidence))

            # P2.7: SELL as negative filter — không override SELL→HOLD.
            # SELL signal từ model = cảnh báo tránh mua.
            # Nếu model SELL + signal hiện tại BUY → downgrade BUY → HOLD với tag cảnh báo.
            # SELL thuần không phải lệnh bán (backtest xác nhận SELL không đáng tin).
            ml_sell_flag = (signal_val == -1)
            if signal_val == -1:
                signal_val = 0  # render as HOLD, but flag ml_sell

            label_map = {0: "HOLD", 1: "BUY"}
            signal = label_map[signal_val]

            results.append({
                "ticker": ticker,
                "date": latest_date.date(),
                "signal": signal,
                "ml_signal": signal,  # pre-overlay signal for ablation
                "confidence": round(confidence, 3),
                "ml_sell_flag": int(ml_sell_flag),  # 1 = model wanted to SELL
                "close": float(latest["close"].values[0]) * 1000,
                "rsi14": round(float(latest["rsi14"].values[0]), 1),
                "ret_5d": round(float(latest["ret_5d"].values[0]) * 100, 2),
                "vni_bull": int(latest["vni_bull_regime"].values[0]),
                "floor_streak": int(latest["floor_streak"].values[0]),
                "ceil_streak": int(latest["ceil_streak"].values[0]),
            })

        except Exception as e:
            print(f"  ⚠️  {ticker}: {e}")

    df_signals = pd.DataFrame(results)
    if df_signals.empty:
        return df_signals

    # BUY high-conf lên đầu, sau đó sort by confidence
    df_signals["_sort"] = df_signals.apply(
        lambda r: (0 if r["signal"] == "BUY" and r["confidence"] >= HIGH_CONFIDENCE_THRESHOLD
                   else 1 if r["signal"] == "BUY"
                   else 2),
        axis=1
    )
    df_signals = df_signals.sort_values(["_sort", "confidence"], ascending=[True, False])
    df_signals = df_signals.drop(columns=["_sort"])
    return df_signals


def save_signals(df: pd.DataFrame, date: str | None = None) -> Path:
    if date is None:
        date = pd.Timestamp.today().strftime("%Y-%m-%d")
    path = SIGNALS_DIR / f"{date}.csv"
    df.to_csv(path, index=False)
    return path


def print_signals(df: pd.DataFrame):
    if df.empty:
        print("No signals generated.")
        return

    date_str = str(df["date"].iloc[0])
    bull = df["vni_bull"].iloc[0] if "vni_bull" in df.columns else "?"
    regime = "🟢 BULL" if bull == 1 else "🔴 BEAR"

    print(f"\n{'='*62}")
    print(f"  VN30 BUY SIGNALS — {date_str}   Market: {regime}")
    print(f"{'='*62}")

    buy_signals = df[df["signal"] == "BUY"]
    high = buy_signals[buy_signals["confidence"] >= HIGH_CONFIDENCE_THRESHOLD]
    low  = buy_signals[buy_signals["confidence"] < HIGH_CONFIDENCE_THRESHOLD]

    has_news = "news_tag" in df.columns

    print(f"\n⭐ ACTIONABLE BUY (confidence ≥ {HIGH_CONFIDENCE_THRESHOLD:.0%})")
    if high.empty:
        print("   Không có tín hiệu đủ mạnh hôm nay.")
    else:
        print(f"   {'Ticker':<6} {'Close':>7} {'Conf':>6} {'RSI':>6} {'Ret5d':>7}  {'News'}")
        print(f"   {'-'*52}")
        for _, r in high.iterrows():
            tag = f"  {r['news_tag']}" if has_news and r.get("news_tag") else ""
            print(f"   {r['ticker']:<6} {r['close']:>10,g} {r['confidence']:>6.1%} {r['rsi14']:>6.1f} {r['ret_5d']:>6.1f}%{tag}")

    if not low.empty:
        print(f"\n   BUY thấp confidence ({len(low)} mã — theo dõi thêm)")
        for _, r in low.iterrows():
            tag = f"  {r['news_tag']}" if has_news and r.get("news_tag") else ""
            print(f"   {r['ticker']:<6} {r['close']:>10,g} {r['confidence']:>6.1%} RSI:{r['rsi14']:>5.1f}{tag}")

    # Downgraded BUY → HOLD vì tin xấu
    if has_news:
        downgraded = df[(df["signal"] == "HOLD") & df["news_tag"].str.contains("tin xấu|sentiment âm", na=False)]
        if not downgraded.empty:
            print(f"\n🔻 Downgraded BUY → HOLD (do news)")
            for _, r in downgraded.iterrows():
                print(f"   {r['ticker']:<6} {r['close']:>10,g}  {r['news_tag']}")

    # HOLD đáng chú ý do tin tốt
    if has_news:
        watch = df[(df["signal"] == "HOLD") & (df["news_tag"] == "👀 watch: tin tốt")]
        if not watch.empty:
            print(f"\n👀 WATCH — HOLD nhưng có tin tốt")
            for _, r in watch.iterrows():
                sent = r.get("news_sentiment_1d", 0)
                print(f"   {r['ticker']:<6} {r['close']:>7.1f}  sent={sent:+.2f}  {r['news_tag']}")

    hold = df[df["signal"] == "HOLD"]
    print(f"\n⚪ HOLD ({len(hold)} mã)\n")


if __name__ == "__main__":
    import sys
    from monitoring.circuit_breaker import CircuitBreaker

    refresh = "--refresh" in sys.argv
    mode = "live data" if refresh else "cached data"
    print(f"Generating signals ({mode})...")

    # Check circuit breaker — may raise threshold or halt
    cb = CircuitBreaker()
    cb_result = cb.check()
    if cb.is_open:
        print(f"\n🚨 CIRCUIT BREAKER OPEN: {cb_result['reason']}")
        print("   Signals halted. Fix performance issues before resuming.")
        sys.exit(1)
    elif cb_result["status"] == "HALF-OPEN":
        print(f"⚠️  Circuit breaker HALF-OPEN")

    df = get_signal_today(refresh_data=refresh)

    # Dynamic threshold display (informational — filter already applied per-ticker)
    from src.threshold import get_today_threshold
    from src.macro import build_macro_features as _bm
    from src.regime import detect_regime as _dr, get_current_regime as _gcr
    from src.fetch import load_index as _li
    _vni = _li("VNINDEX")
    _regime = _gcr(_dr(_vni))
    _macro = _bm()
    dyn_thresh = get_today_threshold(_regime, _macro, cb_result["status"])
    print(f"  Dynamic threshold: {dyn_thresh:.0%}  (regime={_regime['regime_name']})")

    # News overlay (trong nước + quốc tế + geo risk)
    print("Fetching news sentiment...")
    sentiment, articles = get_today_sentiment(verbose=True)
    save_sentiment(sentiment)
    save_articles(articles)
    df = apply_news_overlay(df, sentiment)

    # Live overlay: foreign flow + insider trading
    print("Applying live overlays (foreign flow + insider)...")
    df = apply_live_overlay(df)

    print_signals(df)
    path = save_signals(df)
    print(f"✅ Saved → {path}")

    # Tracker: ghi dự đoán + kiểm tra kết quả cũ + báo cáo
    print("\nRunning prediction tracker...")
    tracker.run(signals=df)

    # Gửi email thông báo
    print("\nSending email notification...")
    from src.notifications import send_signal_email
    send_signal_email(df)
