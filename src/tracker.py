"""
tracker.py — Prediction Accuracy Tracking System

Luồng hoạt động:
  1. Mỗi ngày sau khi tạo signal → record_predictions() lưu dự đoán
  2. Sau 5 ngày → resolve_predictions() kiểm tra kết quả thực tế
  3. analyze() → tính toán metrics, tìm pattern đúng/sai
  4. generate_report() → báo cáo + gợi ý cải thiện

Chỉ số tổng hợp: "Prediction Score" (0–100)
  - 50 = tương đương chọn ngẫu nhiên
  - 60 = tốt
  - 70+ = rất tốt
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.fetch import load_ticker, VN30
from src.sector import TICKER_TO_SECTOR

TRACKER_DIR = Path(__file__).parent.parent / "data" / "tracker"
TRACKER_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = TRACKER_DIR / "predictions.parquet"
REPORT_FILE      = TRACKER_DIR / "latest_report.json"

FORWARD_DAYS   = 5
BUY_TARGET     = 0.03   # ≥3% = "đúng" (khớp với target training)
RANDOM_BASE    = 0.24   # fallback hardcode khi không có đủ data
BASELINE_WINDOW = 60    # số phiên gần nhất để tính dynamic baseline

BASELINE_LOG = Path(__file__).parent.parent / "reports" / "baseline_drift.csv"


# ─── 0. DYNAMIC BASELINE ─────────────────────────────────────────────────────

def compute_dynamic_baseline(window: int = BASELINE_WINDOW) -> float:
    """
    Tính baseline động: % ngày trong {window} phiên gần nhất
    có 5-day forward return ≥ BUY_TARGET (3%), cross toàn bộ VN30.

    Thay thế RANDOM_BASE=0.24 hardcode bằng giá trị phản ánh regime thị trường hiện tại.
    Fallback về RANDOM_BASE nếu không đủ data.
    """
    try:
        records = []
        for ticker in VN30:
            df = load_ticker(ticker)
            if df is None or df.empty or "close" not in df.columns:
                continue
            df = df.tail(window + FORWARD_DAYS + 5).copy()
            df["fwd_ret"] = df["close"].shift(-FORWARD_DAYS) / df["close"] - 1
            df = df.dropna(subset=["fwd_ret"]).tail(window)
            records.append((df["fwd_ret"] >= BUY_TARGET).values)

        if not records:
            return RANDOM_BASE

        all_hits = np.concatenate(records)
        baseline = float(all_hits.mean())

        # Log để track drift
        try:
            BASELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
            log_row = pd.DataFrame([{
                "date": datetime.now().date().isoformat(),
                "dynamic_baseline": round(baseline, 4),
                "static_baseline": RANDOM_BASE,
                "window_sessions": window,
                "n_data_points": len(all_hits),
            }])
            if BASELINE_LOG.exists():
                existing = pd.read_csv(BASELINE_LOG)
                log_df = pd.concat([existing, log_row], ignore_index=True).drop_duplicates("date", keep="last")
            else:
                log_df = log_row
            log_df.to_csv(BASELINE_LOG, index=False)
        except Exception:
            pass  # logging failure không được crash tracker

        return max(0.05, min(0.80, baseline))  # clip về range hợp lý

    except Exception:
        return RANDOM_BASE


# ─── 1. GHI DỰ ĐOÁN ──────────────────────────────────────────────────────────

def record_predictions(signals: pd.DataFrame) -> int:
    """
    Lưu tín hiệu hôm nay vào predictions.parquet.
    Chỉ lưu những mã chưa được ghi cho ngày này.
    Trả về số dự đoán mới được ghi.
    """
    if signals.empty:
        return 0

    today_str = str(signals["date"].iloc[0])

    # Load existing
    if PREDICTIONS_FILE.exists():
        existing = pd.read_parquet(PREDICTIONS_FILE)
        already_recorded = set(
            existing[existing["signal_date"] == today_str]["ticker"].tolist()
        )
    else:
        existing = pd.DataFrame()
        already_recorded = set()

    rows = []
    for _, r in signals.iterrows():
        if r["ticker"] in already_recorded:
            continue

        # Parse tags để biết context lúc ra tín hiệu
        tag = str(r.get("news_tag", "") or "")
        has_news_confirm  = "news confirmed" in tag or "tin tốt" in tag
        has_foreign_buy   = "ngoại mua" in tag
        has_foreign_sell  = "ngoại bán" in tag or "ngoại rút" in tag
        has_insider_buy   = "nội bộ đang mua" in tag
        has_insider_sell  = "nội bộ đang bán" in tag
        has_floor_streak  = "sàn" in tag and "ngày" in tag
        has_ceil_streak   = "trần" in tag and "ngày" in tag
        room_full         = "kín room" in tag

        rows.append({
            "signal_date":       today_str,
            "ticker":            r["ticker"],
            "signal":            r["signal"],
            "confidence":        float(r["confidence"]),
            "close_at_signal":   float(r["close"]),
            "rsi14":             float(r.get("rsi14", 0)),
            "ret_5d_at_signal":  float(r.get("ret_5d", 0)),
            "market_regime":     int(r.get("vni_bull", 0)),
            "sector":            TICKER_TO_SECTOR.get(r["ticker"], "unknown"),
            "floor_streak":      int(r.get("floor_streak", 0)),
            "ceil_streak":       int(r.get("ceil_streak", 0)),
            "news_tag":          tag[:200],
            "has_news_confirm":  int(has_news_confirm),
            "has_foreign_buy":   int(has_foreign_buy),
            "has_foreign_sell":  int(has_foreign_sell),
            "has_insider_buy":   int(has_insider_buy),
            "has_insider_sell":  int(has_insider_sell),
            "has_floor_streak":  int(has_floor_streak),
            "has_ceil_streak":   int(has_ceil_streak),
            "room_full":         int(room_full),
            # resolve fields (điền sau)
            "resolved":          False,
            "resolve_date":      None,
            "close_at_resolve":  None,
            "actual_return":     None,
            "hit":               None,    # return >= BUY_TARGET cho BUY signal
            "any_gain":          None,    # return > 0 (thắng bất kỳ)
        })

    if not rows:
        return 0

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined.to_parquet(PREDICTIONS_FILE, index=False)
    return len(rows)


# ─── 2. KIỂM TRA KẾT QUẢ ─────────────────────────────────────────────────────

def resolve_predictions(use_cached: bool = True) -> int:
    """
    Với các dự đoán BUY chưa resolve và đã đủ FORWARD_DAYS ngày,
    lấy giá thực tế để đánh giá đúng/sai.
    Trả về số dự đoán vừa được resolve.
    """
    if not PREDICTIONS_FILE.exists():
        return 0

    df = pd.read_parquet(PREDICTIONS_FILE)
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=FORWARD_DAYS)

    # Chỉ resolve BUY signals chưa xử lý và đủ 5 ngày
    mask = (
        (~df["resolved"]) &
        (df["signal"] == "BUY") &
        (pd.to_datetime(df["signal_date"]) <= cutoff)
    )
    pending = df[mask]
    if pending.empty:
        return 0

    resolved_count = 0
    for idx, row in pending.iterrows():
        ticker = row["ticker"]
        signal_date = pd.Timestamp(row["signal_date"])

        try:
            if use_cached:
                from src.fetch import load_ticker
                price_df = load_ticker(ticker)
            else:
                from src.fetch import fetch_ticker
                start = (signal_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
                price_df = fetch_ticker(ticker, start=start)

            if price_df is None or price_df.empty:
                continue

            # Tìm giá đóng cửa sau FORWARD_DAYS ngày giao dịch
            future = price_df[price_df.index > signal_date]
            if len(future) < FORWARD_DAYS:
                continue  # Chưa đủ dữ liệu

            close_future = float(future.iloc[FORWARD_DAYS - 1]["close"])
            close_signal = float(row["close_at_signal"])
            actual_return = (close_future / close_signal) - 1

            df.at[idx, "resolved"]       = True
            df.at[idx, "resolve_date"]   = future.index[FORWARD_DAYS - 1].date().isoformat()
            df.at[idx, "close_at_resolve"] = close_future
            df.at[idx, "actual_return"]  = round(actual_return, 4)
            df.at[idx, "hit"]            = int(actual_return >= BUY_TARGET)
            df.at[idx, "any_gain"]       = int(actual_return > 0)
            resolved_count += 1

        except Exception as e:
            print(f"  ⚠️  Resolve {ticker}: {e}")

    if resolved_count > 0:
        df.to_parquet(PREDICTIONS_FILE, index=False)

    return resolved_count


# ─── 3. TÍNH PREDICTION SCORE ────────────────────────────────────────────────

def compute_prediction_score(resolved: pd.DataFrame) -> dict:
    """
    Prediction Score (0–100):
      Thành phần 1 (50%): Hit Rate chuẩn hóa so với baseline ngẫu nhiên
      Thành phần 2 (30%): Avg return của BUY signals so với 0
      Thành phần 3 (20%): Calibration — confidence cao có thật sự đúng nhiều hơn không

    Score 50 = ngang random, 60 = tốt, 70+ = rất tốt
    """
    if resolved.empty:
        return {"score": None, "components": {}}

    buy = resolved[resolved["signal"] == "BUY"]
    if buy.empty or buy["hit"].isna().all():
        return {"score": None, "components": {}}

    buy = buy.dropna(subset=["hit", "actual_return"])

    # Component 1: Hit Rate (chuẩn hóa so với dynamic baseline)
    hit_rate = buy["hit"].mean()
    dynamic_base = compute_dynamic_baseline()
    # Scale: dynamic_base → 0 điểm, 1.0 → 100 điểm
    c1 = max(0, min(100, (hit_rate - dynamic_base) / (1 - dynamic_base) * 100))

    # Component 2: Avg Return
    avg_ret = buy["actual_return"].mean()
    # Scale: -10% → 0, 0% → 40, +10% → 80, +20% → 100
    c2 = max(0, min(100, avg_ret * 400 + 40))

    # Component 3: Calibration (Spearman correlation confidence ~ hit)
    if len(buy) >= 10:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(buy["confidence"], buy["hit"])
        c3 = max(0, min(100, (corr + 1) / 2 * 100))
    else:
        c3 = 50  # không đủ data → neutral

    score = round(c1 * 0.5 + c2 * 0.3 + c3 * 0.2, 1)

    return {
        "score": score,
        "components": {
            "hit_rate_score":   round(c1, 1),
            "avg_return_score": round(c2, 1),
            "calibration_score": round(c3, 1),
        },
        "raw": {
            "hit_rate":  round(hit_rate, 3),
            "avg_return": round(avg_ret, 4),
            "n_resolved": len(buy),
            "dynamic_baseline": round(dynamic_base, 4),
            "static_baseline": RANDOM_BASE,
        }
    }


# ─── 4. PHÂN TÍCH PATTERN ────────────────────────────────────────────────────

def analyze(resolved: pd.DataFrame) -> dict:
    """
    Tìm pattern trong đúng/sai:
    - Theo confidence band
    - Theo market regime (BULL/BEAR)
    - Theo tag (news, foreign, insider)
    - Theo sector
    - Theo earnings season
    - Trend 30 ngày gần nhất vs tổng thể
    """
    buy = resolved[resolved["signal"] == "BUY"].dropna(subset=["hit", "actual_return"])
    if buy.empty:
        return {}

    result = {}

    # --- Confidence bands ---
    bands = [(0, 0.5, "<50%"), (0.5, 0.6, "50-60%"), (0.6, 0.7, "60-70%"), (0.7, 1.01, "≥70%")]
    conf_analysis = []
    for lo, hi, label in bands:
        sub = buy[(buy["confidence"] >= lo) & (buy["confidence"] < hi)]
        if len(sub) >= 3:
            conf_analysis.append({
                "band": label,
                "n": len(sub),
                "hit_rate": round(sub["hit"].mean(), 3),
                "avg_return": round(sub["actual_return"].mean(), 4),
            })
    result["by_confidence"] = conf_analysis

    # --- Market regime ---
    regime_analysis = []
    for regime, label in [(1, "BULL"), (0, "BEAR")]:
        sub = buy[buy["market_regime"] == regime]
        if len(sub) >= 3:
            regime_analysis.append({
                "regime": label,
                "n": len(sub),
                "hit_rate": round(sub["hit"].mean(), 3),
                "avg_return": round(sub["actual_return"].mean(), 4),
            })
    result["by_regime"] = regime_analysis

    # --- Tags ---
    tag_cols = {
        "has_news_confirm": "Có xác nhận tin tức",
        "has_foreign_buy":  "Ngoại mua ròng",
        "has_foreign_sell": "Ngoại bán ròng",
        "has_insider_buy":  "Nội bộ đang mua",
        "has_floor_streak": "Chuỗi sàn",
        "room_full":        "Kín room ngoại",
    }
    tag_analysis = []
    for col, label in tag_cols.items():
        if col not in buy.columns:
            continue
        with_tag = buy[buy[col] == 1]
        without  = buy[buy[col] == 0]
        if len(with_tag) >= 3 and len(without) >= 3:
            tag_analysis.append({
                "tag": label,
                "n_with": len(with_tag),
                "hit_with": round(with_tag["hit"].mean(), 3),
                "n_without": len(without),
                "hit_without": round(without["hit"].mean(), 3),
                "delta": round(with_tag["hit"].mean() - without["hit"].mean(), 3),
            })
    result["by_tag"] = sorted(tag_analysis, key=lambda x: abs(x["delta"]), reverse=True)

    # --- Sector ---
    sector_analysis = []
    for sector in buy["sector"].unique():
        sub = buy[buy["sector"] == sector]
        if len(sub) >= 5:
            sector_analysis.append({
                "sector": sector,
                "n": len(sub),
                "hit_rate": round(sub["hit"].mean(), 3),
                "avg_return": round(sub["actual_return"].mean(), 4),
            })
    result["by_sector"] = sorted(sector_analysis, key=lambda x: x["hit_rate"], reverse=True)

    # --- Trend: 30 ngày gần nhất vs tổng thể ---
    buy["signal_date_ts"] = pd.to_datetime(buy["signal_date"])
    cutoff_30d = buy["signal_date_ts"].max() - pd.Timedelta(days=30)
    recent = buy[buy["signal_date_ts"] >= cutoff_30d]
    overall_hit = buy["hit"].mean()
    recent_hit  = recent["hit"].mean() if len(recent) >= 3 else None
    result["trend"] = {
        "overall_hit_rate": round(overall_hit, 3),
        "recent_30d_hit_rate": round(recent_hit, 3) if recent_hit is not None else None,
        "n_overall": len(buy),
        "n_recent_30d": len(recent),
        "degrading": bool(recent_hit is not None and recent_hit < overall_hit - 0.05),
    }

    return result


# ─── 5. GỢI Ý CẢI THIỆN ─────────────────────────────────────────────────────

def generate_suggestions(score_data: dict, analysis: dict) -> list[dict]:
    """
    Dựa trên phân tích, tự động gợi ý:
    - Điều chỉnh threshold
    - Tránh điều kiện thị trường nào
    - Thông tin nào nên bổ sung thêm
    """
    suggestions = []

    if not analysis:
        return suggestions

    # --- Confidence threshold ---
    conf_bands = analysis.get("by_confidence", [])
    for band in conf_bands:
        if band["band"] == "<50%" and band["n"] >= 5 and band["hit_rate"] < 0.25:
            suggestions.append({
                "type": "threshold",
                "priority": "high",
                "title": "Nâng ngưỡng confidence lên 60%+",
                "detail": f"Tín hiệu dưới 50% confidence chỉ đúng {band['hit_rate']:.0%} — thấp hơn random. Bỏ qua nhóm này.",
                "action": "Tăng HIGH_CONFIDENCE_THRESHOLD từ 0.60 lên 0.65",
            })
        if band["band"] == "≥70%" and band["n"] >= 5 and band["hit_rate"] > 0.55:
            suggestions.append({
                "type": "signal_quality",
                "priority": "high",
                "title": "Tín hiệu ≥70% confidence rất đáng tin",
                "detail": f"Nhóm này đúng {band['hit_rate']:.0%} — cao nhất. Ưu tiên vào lệnh khi confidence ≥70%.",
                "action": "Tăng vốn/lệnh khi confidence ≥70%",
            })

    # --- Market regime ---
    regime = {r["regime"]: r for r in analysis.get("by_regime", [])}
    if "BEAR" in regime and regime["BEAR"]["hit_rate"] < 0.25 and regime["BEAR"]["n"] >= 5:
        suggestions.append({
            "type": "market_filter",
            "priority": "high",
            "title": "Hạn chế giao dịch trong thị trường BEAR",
            "detail": f"Khi VNINDEX dưới MA50, độ chính xác chỉ {regime['BEAR']['hit_rate']:.0%}. Nên giảm vốn hoặc bỏ qua tín hiệu.",
            "action": "Thêm điều kiện: chỉ hành động khi vni_bull_regime = 1",
        })
    if "BULL" in regime and "BEAR" in regime:
        diff = regime["BULL"]["hit_rate"] - regime["BEAR"]["hit_rate"]
        if diff > 0.15:
            suggestions.append({
                "type": "info_needed",
                "priority": "medium",
                "title": "Thêm chỉ số sức mạnh xu hướng thị trường",
                "detail": f"BULL vs BEAR chênh lệch {diff:.0%} độ chính xác. Cần thêm ADX (Average Directional Index) để đo độ mạnh xu hướng.",
                "action": "Bổ sung: ADX của VNINDEX vào market context features",
            })

    # --- Tags ---
    for tag_info in analysis.get("by_tag", []):
        if tag_info["tag"] == "Có xác nhận tin tức" and tag_info["delta"] > 0.08 and tag_info["n_with"] >= 3:
            suggestions.append({
                "type": "signal_quality",
                "priority": "medium",
                "title": "Ưu tiên mã có xác nhận tin tức",
                "detail": f"Tín hiệu kèm tin tốt đúng {tag_info['hit_with']:.0%} vs {tag_info['hit_without']:.0%} khi không có tin — chênh {tag_info['delta']:+.0%}.",
                "action": "Ưu tiên vào lệnh khi có badge '📰 news confirmed'",
            })
        if tag_info["tag"] == "Ngoại mua ròng" and tag_info["delta"] > 0.08 and tag_info["n_with"] >= 3:
            suggestions.append({
                "type": "signal_quality",
                "priority": "medium",
                "title": "Xác nhận của khối ngoại rất có giá trị",
                "detail": f"BUY + ngoại mua đúng {tag_info['hit_with']:.0%} vs {tag_info['hit_without']:.0%} — khối ngoại đang đồng thuận với AI.",
                "action": "Ưu tiên lệnh khi có '🌊 ngoại mua'",
            })
        if tag_info["tag"] == "Chuỗi sàn" and tag_info["delta"] < -0.10 and tag_info["n_with"] >= 3:
            suggestions.append({
                "type": "risk_filter",
                "priority": "high",
                "title": "Tránh mua khi có chuỗi sàn",
                "detail": f"BUY kèm chuỗi sàn chỉ đúng {tag_info['hit_with']:.0%} — thấp hơn bình thường {abs(tag_info['delta']):.0%}. Đây là bán tháo, không phải cơ hội.",
                "action": "Đã có filter tự động, cân nhắc mở rộng: sàn 1 ngày cũng nên tránh",
            })

    # --- Sector ---
    bad_sectors = [s for s in analysis.get("by_sector", []) if s["hit_rate"] < 0.20 and s["n"] >= 5]
    good_sectors = [s for s in analysis.get("by_sector", []) if s["hit_rate"] > 0.45 and s["n"] >= 5]
    if bad_sectors:
        names = ", ".join(s["sector"] for s in bad_sectors)
        suggestions.append({
            "type": "sector_filter",
            "priority": "medium",
            "title": f"Hạn chế nhóm ngành: {names}",
            "detail": f"Dự đoán các ngành này đang kém — hit rate dưới 20%. Có thể cần thêm đặc thù ngành.",
            "action": f"Thêm info: với ngành {names}, bổ sung chỉ số đặc thù (vd: NIM cho ngân hàng, tồn kho cho consumer)",
        })
    if good_sectors:
        names = ", ".join(s["sector"] for s in good_sectors)
        suggestions.append({
            "type": "signal_quality",
            "priority": "low",
            "title": f"Mô hình dự đoán tốt nhất: {names}",
            "detail": f"Hit rate {good_sectors[0]['hit_rate']:.0%}+ — đây là nhóm đáng tin cậy nhất.",
            "action": "Có thể tăng vốn/lệnh với các mã trong nhóm này",
        })

    # --- Trend degrading ---
    trend = analysis.get("trend", {})
    if trend.get("degrading"):
        suggestions.append({
            "type": "model_health",
            "priority": "high",
            "title": "Độ chính xác đang giảm trong 30 ngày gần đây",
            "detail": f"Tổng thể {trend['overall_hit_rate']:.0%} nhưng 30 ngày gần nhất chỉ {trend['recent_30d_hit_rate']:.0%}. Thị trường có thể đang thay đổi regime.",
            "action": "Retrain model với data mới nhất, hoặc tạm thời nâng threshold lên 65%",
        })

    # --- Luôn gợi ý thêm data mới ---
    suggestions.append({
        "type": "info_needed",
        "priority": "low",
        "title": "Dữ liệu có thể bổ sung để tăng độ chính xác",
        "detail": (
            "1. Tỷ giá USD/VND hàng ngày — ảnh hưởng dòng vốn ngoại\n"
            "2. Lãi suất liên ngân hàng (VNIBOR) — tín hiệu thanh khoản\n"
            "3. Giá dầu Brent — ảnh hưởng GAS, PLX, POW\n"
            "4. Ngày GDKHQ cổ tức — giá thường giảm đúng ngày này\n"
            "5. Phát hành thêm / mua lại cổ phiếu — pha loãng hoặc hỗ trợ giá"
        ),
        "action": "Thêm dần theo thứ tự ưu tiên, bắt đầu từ tỷ giá USD/VND",
    })

    return suggestions


# ─── 6. BÁO CÁO TỔNG HỢP ─────────────────────────────────────────────────────

def generate_report(verbose: bool = True) -> dict:
    """
    Tạo báo cáo đầy đủ: score + phân tích + gợi ý.
    Lưu vào data/tracker/latest_report.json
    """
    if not PREDICTIONS_FILE.exists():
        print("  Chưa có dữ liệu dự đoán. Chạy signal_generator trước.")
        return {}

    df = pd.read_parquet(PREDICTIONS_FILE)
    resolved = df[df["resolved"] == True].copy()

    score_data  = compute_prediction_score(resolved)
    analysis    = analyze(resolved)
    suggestions = generate_suggestions(score_data, analysis)

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_predictions": len(df[df["signal"] == "BUY"]),
        "total_resolved": len(resolved[resolved["signal"] == "BUY"]) if not resolved.empty else 0,
        "prediction_score": score_data,
        "analysis": analysis,
        "suggestions": suggestions,
    }

    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        _print_report(report)

    return report


def _print_report(report: dict):
    score = report["prediction_score"]
    analysis = report["analysis"]
    suggestions = report["suggestions"]

    n_total    = report["total_predictions"]
    n_resolved = report["total_resolved"]

    print(f"\n{'='*60}")
    print(f"  📊 BÁO CÁO ĐỘ CHÍNH XÁC DỰ ĐOÁN")
    print(f"{'='*60}")
    print(f"  Tổng dự đoán BUY: {n_total} | Đã có kết quả: {n_resolved}")

    if score.get("score") is None:
        print("  Chưa đủ dữ liệu để tính điểm (cần ít nhất 5 ngày).")
        return

    s = score["score"]
    if s >= 70:
        grade = "🟢 Rất tốt"
    elif s >= 60:
        grade = "🟡 Tốt"
    elif s >= 50:
        grade = "🟠 Trung bình"
    else:
        grade = "🔴 Cần cải thiện"

    print(f"\n  ⭐ Điểm Dự Đoán: {s:.1f}/100  {grade}")
    raw = score.get("raw", {})
    print(f"     Hit Rate:   {raw.get('hit_rate', 0):.1%}  (dự đoán BUY tăng ≥3% đúng bao nhiêu %)")
    print(f"     Avg Return: {raw.get('avg_return', 0):+.2%}  (lãi/lỗ TB thực tế mỗi lần BUY)")
    n = raw.get('n_resolved', 0)
    print(f"     Mẫu:        {n} BUY signals đã có kết quả")

    # Confidence breakdown
    conf = analysis.get("by_confidence", [])
    if conf:
        print(f"\n  📈 Theo độ chắc chắn:")
        for b in conf:
            bar = "█" * int(b["hit_rate"] * 20)
            print(f"     {b['band']:8s} │{bar:<20s}│ {b['hit_rate']:.0%}  ({b['n']} lệnh, avg {b['avg_return']:+.1%})")

    # Regime
    regime = analysis.get("by_regime", [])
    if regime:
        print(f"\n  🌍 Theo xu hướng thị trường:")
        for r in regime:
            icon = "🟢" if r["regime"] == "BULL" else "🔴"
            print(f"     {icon} {r['regime']:5s}: {r['hit_rate']:.0%}  ({r['n']} lệnh)")

    # Tags
    tags = analysis.get("by_tag", [])
    meaningful = [t for t in tags if abs(t["delta"]) >= 0.05]
    if meaningful:
        print(f"\n  🏷️  Ảnh hưởng của các tín hiệu bổ sung:")
        for t in meaningful[:5]:
            delta_str = f"{t['delta']:+.0%}"
            direction = "↑ tốt hơn" if t["delta"] > 0 else "↓ kém hơn"
            print(f"     {t['tag']:<28s}: {t['hit_with']:.0%} vs {t['hit_without']:.0%}  ({delta_str} {direction})")

    # Sector
    sectors = analysis.get("by_sector", [])
    if sectors:
        print(f"\n  🏭 Theo ngành:")
        for s in sectors[:5]:
            bar = "█" * int(s["hit_rate"] * 20)
            print(f"     {s['sector']:12s} │{bar:<20s}│ {s['hit_rate']:.0%}  ({s['n']} lệnh)")

    # Trend
    trend = analysis.get("trend", {})
    if trend.get("recent_30d_hit_rate") is not None:
        overall = trend["overall_hit_rate"]
        recent  = trend["recent_30d_hit_rate"]
        delta   = recent - overall
        arrow   = "↑" if delta > 0 else "↓"
        print(f"\n  📅 Xu hướng gần đây: {recent:.0%}  (tổng thể {overall:.0%})  {arrow}{abs(delta):.0%}")

    # Suggestions
    high_prio = [s for s in suggestions if s["priority"] == "high"]
    if high_prio:
        print(f"\n  🔧 Gợi ý cải thiện (ưu tiên cao):")
        for s in high_prio[:3]:
            print(f"\n     ❗ {s['title']}")
            print(f"        {s['detail']}")
            print(f"        → {s['action']}")

    other = [s for s in suggestions if s["priority"] != "high"][:2]
    if other:
        print(f"\n  💡 Gợi ý thêm:")
        for s in other:
            print(f"     • {s['title']}: {s['action']}")

    print(f"\n{'='*60}\n")


# ─── 7. MAIN ─────────────────────────────────────────────────────────────────

def run(signals: pd.DataFrame | None = None, verbose: bool = True) -> dict:
    """
    Entry point: record + resolve + report.
    Gọi từ signal_generator sau khi tạo signals.
    """
    # Ghi dự đoán mới
    if signals is not None and not signals.empty:
        n_new = record_predictions(signals)
        if verbose and n_new > 0:
            print(f"  📝 Đã ghi {n_new} dự đoán mới vào tracker")

    # Resolve dự đoán cũ đủ 5 ngày
    n_resolved = resolve_predictions()
    if verbose and n_resolved > 0:
        print(f"  ✅ Đã kiểm tra kết quả {n_resolved} dự đoán từ {FORWARD_DAYS} ngày trước")

    # Tạo báo cáo
    return generate_report(verbose=verbose)


if __name__ == "__main__":
    report = run(verbose=True)
    if report:
        print(f"\nReport saved → {REPORT_FILE}")
