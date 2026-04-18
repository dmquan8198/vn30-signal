"""
dashboard.py — Generate HTML dashboard từ signals + backtest data
Output: dashboard/index.html
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
SIGNALS_DIR = PROJECT_DIR / "signals"
BACKTEST_DIR = PROJECT_DIR / "backtest"
NEWS_DIR = PROJECT_DIR / "data" / "news"
DASHBOARD_DIR = PROJECT_DIR / "dashboard"
DASHBOARD_DIR.mkdir(exist_ok=True)


def load_latest_signals() -> pd.DataFrame:
    files = sorted(SIGNALS_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


def load_backtest() -> pd.DataFrame:
    path = BACKTEST_DIR / "trades.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_latest_news() -> pd.DataFrame:
    files = sorted(NEWS_DIR.glob("*[0-9].csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1], index_col=0)


def load_latest_articles() -> dict:
    files = sorted(NEWS_DIR.glob("*_articles.json"))
    if not files:
        return {}
    return json.loads(files[-1].read_text(encoding="utf-8"))


def load_tracker_report() -> dict:
    path = PROJECT_DIR / "data" / "tracker" / "latest_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_monthly_pnl(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"labels": [], "values": [], "colors": []}
    buy = trades[trades["direction"] == "BUY"].copy()
    buy["month"] = buy["date"].dt.to_period("M").astype(str)
    monthly = buy.groupby("month")["pnl"].sum().reset_index()
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in monthly["pnl"]]
    return {
        "labels": monthly["month"].tolist(),
        "values": (monthly["pnl"] / 1_000_000).round(2).tolist(),
        "colors": colors,
    }


def build_cumulative_pnl(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"labels": [], "values": []}
    buy = trades[trades["direction"] == "BUY"].sort_values("date").copy()
    buy["cum"] = buy["pnl"].cumsum() / 1_000_000
    return {
        "labels": buy["date"].dt.strftime("%Y-%m-%d").tolist(),
        "values": buy["cum"].round(2).tolist(),
    }


def build_confidence_dist(signals: pd.DataFrame) -> dict:
    bins = ["<40%", "40-50%", "50-60%", "60-70%", "70-80%", ">80%"]
    edges = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    counts = []
    for i in range(len(edges) - 1):
        n = ((signals["confidence"] >= edges[i]) & (signals["confidence"] < edges[i+1])).sum()
        counts.append(int(n))
    return {"labels": bins, "values": counts}


def build_tracker_section(report: dict) -> str:
    """Render tracker section HTML từ latest_report.json."""
    if not report or not report.get("prediction_score", {}).get("score"):
        return """
        <div class="card" style="margin-bottom:24px">
          <div class="card-title">🎯 Đánh giá độ chính xác dự đoán</div>
          <div style="color:#64748b;font-size:13px;padding:8px 0">
            Chưa có dữ liệu — hệ thống cần ít nhất 5 ngày giao dịch để đánh giá kết quả.
          </div>
        </div>"""

    score_data  = report["prediction_score"]
    analysis    = report["analysis"]
    suggestions = report["suggestions"]
    s = score_data["score"]
    raw = score_data.get("raw", {})

    if s >= 70:
        grade_color, grade_text = "#10b981", "Rất tốt 🟢"
    elif s >= 60:
        grade_color, grade_text = "#f59e0b", "Tốt 🟡"
    elif s >= 50:
        grade_color, grade_text = "#f97316", "Trung bình 🟠"
    else:
        grade_color, grade_text = "#ef4444", "Cần cải thiện 🔴"

    hit_rate   = raw.get("hit_rate", 0)
    avg_return = raw.get("avg_return", 0)
    n_resolved = raw.get("n_resolved", 0)
    ret_color  = "#10b981" if avg_return >= 0 else "#ef4444"

    # Confidence breakdown
    conf_rows = ""
    for b in analysis.get("by_confidence", []):
        pct = b["hit_rate"] * 100
        bar_color = "#10b981" if b["hit_rate"] >= 0.4 else "#f59e0b" if b["hit_rate"] >= 0.28 else "#ef4444"
        conf_rows += f"""
        <div style="display:flex;align-items:center;gap:10px;padding:5px 0">
          <span style="width:64px;font-size:12px;color:#94a3b8">{b['band']}</span>
          <div style="flex:1;height:8px;background:#1e3a5f;border-radius:4px;overflow:hidden">
            <div style="width:{min(pct,100):.0f}%;height:100%;background:{bar_color};border-radius:4px"></div>
          </div>
          <span style="width:36px;text-align:right;font-size:12px;font-weight:600;color:{bar_color}">{b['hit_rate']:.0%}</span>
          <span style="width:50px;text-align:right;font-size:11px;color:#64748b">{b['n']} lệnh</span>
        </div>"""

    # Suggestions
    sugg_html = ""
    high = [s2 for s2 in suggestions if s2["priority"] == "high"][:3]
    med  = [s2 for s2 in suggestions if s2["priority"] == "medium"][:2]
    for s2 in high:
        sugg_html += f"""
        <div style="padding:10px 12px;background:rgba(239,68,68,0.07);border-left:3px solid #ef4444;border-radius:0 6px 6px 0;margin-bottom:8px">
          <div style="font-size:13px;font-weight:600;color:#f87171;margin-bottom:3px">❗ {s2['title']}</div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.5">{s2['detail']}</div>
          <div style="font-size:11px;color:#64748b;margin-top:4px">→ {s2['action']}</div>
        </div>"""
    for s2 in med:
        sugg_html += f"""
        <div style="padding:10px 12px;background:rgba(245,158,11,0.07);border-left:3px solid #f59e0b;border-radius:0 6px 6px 0;margin-bottom:8px">
          <div style="font-size:13px;font-weight:600;color:#fbbf24;margin-bottom:3px">💡 {s2['title']}</div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.5">{s2['detail']}</div>
          <div style="font-size:11px;color:#64748b;margin-top:4px">→ {s2['action']}</div>
        </div>"""

    # Trend
    trend = analysis.get("trend", {})
    trend_html = ""
    if trend.get("recent_30d_hit_rate") is not None:
        r30 = trend["recent_30d_hit_rate"]
        overall = trend["overall_hit_rate"]
        delta = r30 - overall
        arrow = "↑" if delta > 0 else "↓"
        d_color = "#10b981" if delta > 0 else "#ef4444"
        trend_html = f"""
        <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-top:1px solid #1e3a5f;margin-top:8px">
          <span style="font-size:12px;color:#64748b">30 ngày gần nhất</span>
          <span style="font-size:13px;font-weight:600;color:{d_color}">{r30:.0%}  <span style="font-size:11px">{arrow} {abs(delta):.0%} so với tổng thể</span></span>
        </div>"""

    return f"""
    <div class="card" style="margin-bottom:24px">
      <div class="card-title">🎯 Đánh giá độ chính xác dự đoán</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">

        <!-- Left: Score + stats -->
        <div>
          <div style="display:flex;align-items:flex-end;gap:12px;margin-bottom:16px">
            <div style="font-size:56px;font-weight:800;color:{grade_color};line-height:1">{s:.0f}</div>
            <div>
              <div style="font-size:13px;color:#94a3b8;margin-bottom:2px">Điểm Dự Đoán / 100</div>
              <div style="font-size:14px;font-weight:600;color:{grade_color}">{grade_text}</div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px">
            <div style="background:#0f172a;border-radius:8px;padding:10px">
              <div style="font-size:11px;color:#64748b;margin-bottom:3px">Tỷ lệ đúng</div>
              <div style="font-size:20px;font-weight:700;color:#f8fafc">{hit_rate:.0%}</div>
              <div style="font-size:10px;color:#64748b">trong {n_resolved} lệnh BUY</div>
            </div>
            <div style="background:#0f172a;border-radius:8px;padding:10px">
              <div style="font-size:11px;color:#64748b;margin-bottom:3px">Lãi TB thực tế</div>
              <div style="font-size:20px;font-weight:700;color:{ret_color}">{avg_return:+.1%}</div>
              <div style="font-size:10px;color:#64748b">mỗi lần mua</div>
            </div>
          </div>
          <div style="font-size:12px;color:#64748b;margin-bottom:6px;font-weight:500">Độ chính xác theo mức tin cậy:</div>
          {conf_rows}
          {trend_html}
        </div>

        <!-- Right: Suggestions -->
        <div>
          <div style="font-size:12px;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px">Gợi ý cải thiện</div>
          {sugg_html if sugg_html else '<div style="color:#64748b;font-size:13px">Chưa đủ dữ liệu để đưa ra gợi ý cụ thể.</div>'}
        </div>
      </div>
    </div>"""


def generate_html(signals: pd.DataFrame, trades: pd.DataFrame, news: pd.DataFrame, articles: dict, tracker_report: dict = None) -> str:
    date_str = signals["date"].iloc[0] if not signals.empty else datetime.today().strftime("%Y-%m-%d")
    market_bull = int(signals["vni_bull"].iloc[0]) if not signals.empty and "vni_bull" in signals.columns else 0
    market_label = "Thị trường đang tăng 🟢" if market_bull else "Thị trường đang giảm 🔴"
    market_color = "#10b981" if market_bull else "#ef4444"

    buy_signals = signals[signals["signal"] == "BUY"]
    high_conf = buy_signals[buy_signals["confidence"] >= 0.6]
    n_buy = len(buy_signals)
    n_actionable = len(high_conf)

    if not trades.empty:
        buy_trades = trades[trades["direction"] == "BUY"]
        total_pnl = buy_trades["pnl"].sum() / 1_000_000
        win_rate = (buy_trades["net_return"] > 0).mean() * 100
        total_trades = len(buy_trades)
        avg_ret = buy_trades["net_return"].mean()
    else:
        total_pnl = win_rate = total_trades = avg_ret = 0

    mkt_sent = float(news["market_sentiment_1d"].iloc[0]) if not news.empty and "market_sentiment_1d" in news.columns else 0
    mkt_sent_label = f"{mkt_sent:+.2f}"
    mkt_sent_color = "#10b981" if mkt_sent > 0 else "#ef4444" if mkt_sent < 0 else "#6b7280"

    monthly = build_monthly_pnl(trades)
    cumulative = build_cumulative_pnl(trades)
    conf_dist = build_confidence_dist(signals)
    tracker_section_html = build_tracker_section(tracker_report or {})

    # Sentiment icon
    if mkt_sent > 0.3:
        sent_icon = "😊"
        sent_text = "Tích cực"
    elif mkt_sent > 0:
        sent_icon = "😐"
        sent_text = "Trung lập"
    elif mkt_sent < -0.3:
        sent_icon = "😟"
        sent_text = "Tiêu cực"
    else:
        sent_icon = "😐"
        sent_text = "Trung lập"

    def signal_rows(df):
        rows = ""
        for _, r in df.iterrows():
            sig = r["signal"]
            conf = r["confidence"]
            tag = str(r.get("news_tag", "") or "")
            ticker = r["ticker"]

            # News badge — clickable to show articles popup
            news_badges = ""
            if tag:
                parts = [p.strip() for p in tag.split("  ") if p.strip()]
                for p in parts:
                    news_badges += f'<span class="news-badge news-badge-link" data-ticker="{ticker}">{p}</span> '

            if sig == "BUY" and conf >= 0.6:
                row_class = "row-buy-strong"
                sig_badge = '<span class="badge badge-buy">⭐ NÊN MUA</span>'
            elif sig == "BUY":
                row_class = "row-buy"
                sig_badge = '<span class="badge badge-buy-low">MUA</span>'
            else:
                row_class = "row-hold"
                sig_badge = '<span class="badge badge-hold">CHỜ</span>'

            conf_pct = conf * 100
            conf_color = "#10b981" if conf >= 0.6 else "#f59e0b"
            conf_bar = (
                f'<div class="conf-bar">'
                f'<div class="conf-track"><div class="conf-fill" style="width:{conf_pct:.0f}%;background:{conf_color}"></div></div>'
                f'<span style="color:{conf_color};font-weight:600">{conf:.0%}</span>'
                f'</div>'
            )

            rsi = r.get("rsi14", 0)
            if rsi > 70:
                rsi_color = "#ef4444"
                rsi_label = f'<span title="Quá mua — giá có thể sắp điều chỉnh" style="color:{rsi_color};cursor:help">{rsi:.0f} 🔥</span>'
            elif rsi < 30:
                rsi_color = "#10b981"
                rsi_label = f'<span title="Quá bán — giá có thể sắp phục hồi" style="color:{rsi_color};cursor:help">{rsi:.0f} 💚</span>'
            else:
                rsi_label = f'<span style="color:#94a3b8">{rsi:.0f}</span>'

            ret5 = r.get("ret_5d", 0)
            ret_color = "#10b981" if ret5 > 0 else "#ef4444"

            # Ticker cell — links to Vietstock detail page
            ticker_cell = f'<a class="ticker-link" href="https://finance.vietstock.vn/{ticker}/thong-tin-co-ban.htm" target="_blank" rel="noopener">{ticker}</a>'

            rows += f"""
            <tr class="{row_class}" data-ticker="{ticker}">
                <td class="ticker-cell">{ticker_cell}</td>
                <td>{sig_badge} {news_badges}</td>
                <td class="num">{r['close']:,g}đ</td>
                <td>{conf_bar}</td>
                <td class="num">{rsi_label}</td>
                <td class="num" style="color:{ret_color};font-weight:600">{ret5:+.1f}%</td>
            </tr>"""
        return rows

    all_rows = signal_rows(signals)
    monthly_j = json.dumps(monthly)
    cumulative_j = json.dumps(cumulative)
    conf_dist_j = json.dumps(conf_dist)
    articles_j = json.dumps(articles, ensure_ascii=False)

    # News coverage section
    news_items = ""
    if not news.empty:
        for t in news.index:
            count = int(news.loc[t, "news_count_1d"]) if "news_count_1d" in news.columns else 0
            if count == 0:
                continue
            sent = float(news.loc[t, "news_sentiment_1d"]) if "news_sentiment_1d" in news.columns else 0
            s_color = "#10b981" if sent > 0 else "#ef4444" if sent < 0 else "#64748b"
            s_icon = "📈" if sent > 0 else "📉" if sent < 0 else "➡️"
            news_items += (
                f'<div class="news-ticker-row" data-ticker="{t}">'
                f'<span class="news-ticker-name ticker-link" data-ticker="{t}">{t}</span>'
                f'<span style="font-size:12px;color:{s_color}">{s_icon} {sent:+.2f} · {count} bài</span>'
                f'</div>'
            )
    if not news_items:
        news_items = '<div style="color:#64748b;font-size:13px;padding:12px 0">Không có tin tức VN30 trong 24h qua</div>'

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VN30 Signal Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#0f172a; color:#e2e8f0; min-height:100vh; }}

  /* Header */
  .header {{ background:linear-gradient(135deg,#1e293b,#0f172a); border-bottom:1px solid #1e3a5f; padding:20px 32px; display:flex; align-items:center; justify-content:space-between; }}
  .header h1 {{ font-size:22px; font-weight:700; color:#f8fafc; letter-spacing:-0.5px; }}
  .header h1 span {{ color:#3b82f6; }}
  .meta {{ font-size:13px; color:#64748b; margin-top:4px; }}
  .container {{ max-width:1400px; margin:0 auto; padding:24px 32px; }}

  /* Stat cards */
  .stats-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:16px; margin-bottom:24px; }}
  .stat-card {{ background:#1e293b; border:1px solid #1e3a5f; border-radius:12px; padding:20px; position:relative; }}
  .stat-label {{ font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; display:flex; align-items:center; gap:5px; }}
  .stat-value {{ font-size:28px; font-weight:700; color:#f8fafc; line-height:1; }}
  .stat-sub {{ font-size:12px; color:#64748b; margin-top:6px; }}

  /* Tooltip */
  .tip {{ display:inline-flex; align-items:center; justify-content:center; width:15px; height:15px; border-radius:50%; background:#1e3a5f; color:#64748b; font-size:10px; cursor:help; position:relative; flex-shrink:0; }}
  .tip:hover::after {{ content:attr(data-tip); position:absolute; bottom:calc(100% + 6px); left:50%; transform:translateX(-50%); background:#1e3a5f; border:1px solid #334155; color:#e2e8f0; font-size:12px; padding:8px 12px; border-radius:8px; white-space:nowrap; max-width:260px; white-space:normal; z-index:100; line-height:1.5; text-transform:none; letter-spacing:0; font-weight:400; }}
  .tip:hover::before {{ content:''; position:absolute; bottom:calc(100% + 1px); left:50%; transform:translateX(-50%); border:5px solid transparent; border-top-color:#334155; }}

  /* Layout */
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }}
  .grid-3 {{ display:grid; grid-template-columns:2fr 1fr; gap:20px; margin-bottom:24px; }}
  .card {{ background:#1e293b; border:1px solid #1e3a5f; border-radius:12px; padding:20px; }}
  .card-title {{ font-size:14px; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:16px; display:flex; align-items:center; gap:8px; }}
  .chart-wrap {{ position:relative; height:200px; }}
  .chart-wrap-tall {{ position:relative; height:240px; }}

  /* Table */
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; padding:10px 12px; color:#64748b; font-weight:500; font-size:11px; text-transform:uppercase; border-bottom:1px solid #1e3a5f; }}
  th .tip {{ margin-left:4px; }}
  td {{ padding:10px 12px; border-bottom:1px solid #0f172a; vertical-align:middle; }}
  .row-buy-strong {{ background:rgba(16,185,129,0.06); }}
  .row-buy {{ background:rgba(59,130,246,0.04); }}
  tr:hover {{ background:rgba(255,255,255,0.04); cursor:pointer; }}
  .ticker-cell {{ font-weight:700; font-size:14px; }}
  .ticker-link {{ color:#60a5fa; cursor:pointer; text-decoration:none; border-bottom:1px dashed #334155; }}
  .ticker-link:hover {{ color:#93c5fd; border-bottom-color:#60a5fa; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; }}

  /* Badges */
  .badge {{ display:inline-block; padding:3px 8px; border-radius:6px; font-size:11px; font-weight:600; }}
  .badge-buy {{ background:rgba(16,185,129,0.2); color:#10b981; border:1px solid rgba(16,185,129,0.3); }}
  .badge-buy-low {{ background:rgba(59,130,246,0.15); color:#60a5fa; border:1px solid rgba(59,130,246,0.2); }}
  .badge-hold {{ background:rgba(100,116,139,0.15); color:#94a3b8; border:1px solid rgba(100,116,139,0.2); }}
  .news-badge {{ display:inline-block; padding:2px 6px; border-radius:4px; font-size:10px; background:rgba(245,158,11,0.1); color:#f59e0b; }}
  .news-badge-link {{ cursor:pointer; border-bottom:1px dashed rgba(245,158,11,0.4); }}
  .news-badge-link:hover {{ background:rgba(245,158,11,0.2); }}

  /* Confidence bar */
  .conf-bar {{ display:flex; align-items:center; gap:8px; min-width:110px; }}
  .conf-track {{ flex:1; height:6px; background:#1e3a5f; border-radius:3px; overflow:hidden; min-width:60px; }}
  .conf-fill {{ height:100%; border-radius:3px; transition:width 0.3s; }}

  /* Regime badge */
  .regime-badge {{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; }}
  .market-sent-big {{ font-size:28px; font-weight:700; }}

  /* News popup */
  .news-popup {{ display:none; position:fixed; z-index:1000; background:#1e293b; border:1px solid #334155; border-radius:12px; padding:16px; max-width:380px; width:90%; box-shadow:0 20px 60px rgba(0,0,0,0.6); }}
  .news-popup.visible {{ display:block; }}
  .news-popup-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
  .news-popup-ticker {{ font-size:18px; font-weight:700; color:#f8fafc; }}
  .news-popup-close {{ color:#64748b; cursor:pointer; font-size:18px; line-height:1; padding:4px; }}
  .news-popup-close:hover {{ color:#f8fafc; }}
  .news-article {{ padding:10px 0; border-bottom:1px solid #1e3a5f; }}
  .news-article:last-child {{ border-bottom:none; }}
  .news-article a {{ color:#93c5fd; text-decoration:none; font-size:13px; line-height:1.5; display:block; }}
  .news-article a:hover {{ color:#60a5fa; text-decoration:underline; }}
  .news-article-meta {{ font-size:11px; color:#64748b; margin-top:3px; display:flex; gap:8px; }}
  .sent-pos {{ color:#10b981; }}
  .sent-neg {{ color:#ef4444; }}
  .no-news-msg {{ color:#64748b; font-size:13px; padding:8px 0; }}
  .popup-overlay {{ display:none; position:fixed; inset:0; z-index:999; }}
  .popup-overlay.visible {{ display:block; }}

  /* News coverage */
  .news-ticker-row {{ display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1e3a5f; cursor:pointer; }}
  .news-ticker-row:last-child {{ border-bottom:none; }}
  .news-ticker-row:hover {{ background:rgba(255,255,255,0.03); border-radius:6px; padding-left:4px; }}
  .news-ticker-name {{ font-weight:600; color:#f8fafc; }}

  /* Explain boxes */
  .explain-box {{ background:#162032; border:1px solid #1e3a5f; border-radius:8px; padding:12px 16px; font-size:12px; color:#94a3b8; line-height:1.6; margin-bottom:16px; }}
  .explain-box strong {{ color:#cbd5e1; }}

  @media (max-width:900px) {{
    .stats-grid {{ grid-template-columns:repeat(2,1fr); }}
    .grid-2, .grid-3 {{ grid-template-columns:1fr; }}
  }}
</style>
</head>
<body>

<!-- News Popup -->
<div class="popup-overlay" id="popupOverlay"></div>
<div class="news-popup" id="newsPopup">
  <div class="news-popup-header">
    <span class="news-popup-ticker" id="popupTicker">—</span>
    <span class="news-popup-close" id="popupClose">✕</span>
  </div>
  <div id="popupContent"></div>
</div>

<div class="header">
  <div>
    <h1>VN30 <span>Signal</span> Dashboard</h1>
    <div class="meta">Cập nhật: {date_str} · Mô hình AI phân tích 45 chỉ số kỹ thuật</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:11px;color:#64748b;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px">Xu hướng thị trường</div>
    <span class="regime-badge" style="background:rgba({"16,185,129" if market_bull else "239,68,68"},0.15);color:{market_color};border:1px solid rgba({"16,185,129" if market_bull else "239,68,68"},0.3)">{market_label}</span>
  </div>
</div>

<div class="container">

  <!-- Stats -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">
        Cơ hội mua hôm nay
        <span class="tip" data-tip="Số mã cổ phiếu mà AI đánh giá là có khả năng tăng giá trong 5 ngày tới. ⭐ là những mã AI tự tin nhất (≥60%).">?</span>
      </div>
      <div class="stat-value" style="color:#10b981">{n_buy}</div>
      <div class="stat-sub">{n_actionable} mã AI tự tin cao ⭐</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">
        Lợi nhuận backtest
        <span class="tip" data-tip="Tổng lợi nhuận giả định nếu mua 10 triệu/lần theo tín hiệu BUY từ năm 2023 đến nay, đã trừ phí giao dịch 0.15%/chiều.">?</span>
      </div>
      <div class="stat-value" style="color:{'#10b981' if total_pnl>=0 else '#ef4444'}">{total_pnl:+.1f}M</div>
      <div class="stat-sub">Kiểm nghiệm 2023–2026</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">
        Tỷ lệ thắng
        <span class="tip" data-tip="Tỷ lệ các lần mua theo tín hiệu AI có lời sau 5 ngày. Ví dụ: 65% nghĩa là cứ 10 lần mua thì 6-7 lần có lãi.">?</span>
      </div>
      <div class="stat-value" style="color:#3b82f6">{win_rate:.1f}%</div>
      <div class="stat-sub">{total_trades} lần mua trong backtest</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">
        Lãi TB mỗi lần mua
        <span class="tip" data-tip="Lợi nhuận trung bình mỗi lần mua theo tín hiệu AI, tính theo % giá cổ phiếu sau 5 ngày nắm giữ, đã trừ phí.">?</span>
      </div>
      <div class="stat-value" style="color:{'#10b981' if avg_ret>=0 else '#ef4444'}">{avg_ret:+.2f}%</div>
      <div class="stat-sub">Sau phí giao dịch</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">
        Cảm xúc tin tức
        <span class="tip" data-tip="AI phân tích các bài báo tài chính từ Vietstock trong 24h để đo 'không khí' thị trường. +1.0 = toàn tin tốt, -1.0 = toàn tin xấu.">?</span>
      </div>
      <div class="market-sent-big" style="color:{mkt_sent_color}">{sent_icon} {sent_text}</div>
      <div class="stat-sub" style="color:{mkt_sent_color}">{mkt_sent_label} · Vietstock RSS</div>
    </div>
  </div>

  <!-- Charts row -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title">
        📈 Tăng trưởng vốn theo thời gian
        <span class="tip" data-tip="Đồ thị mô phỏng nếu bạn đầu tư 10 triệu/lệnh theo tất cả tín hiệu MUA của AI từ năm 2023. Đường đi lên = chiến lược đang có lời.">?</span>
      </div>
      <div class="chart-wrap-tall"><canvas id="cumChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">
        📅 Lợi nhuận theo tháng
        <span class="tip" data-tip="Lãi/lỗ mỗi tháng trong backtest. Màu xanh = tháng có lời, đỏ = tháng lỗ. Đơn vị: triệu VND.">?</span>
      </div>
      <div class="chart-wrap-tall"><canvas id="monthChart"></canvas></div>
    </div>
  </div>

  <!-- Tracker / Accuracy Report -->
  {tracker_section_html}

  <!-- Signals table + right sidebar -->
  <div class="grid-3">
    <div class="card">
      <div class="card-title">🎯 Khuyến nghị hôm nay — {date_str}</div>
      <div class="explain-box">
        <strong>Cách đọc bảng:</strong>
        ⭐ <strong>NÊN MUA</strong> = AI tự tin cao, nên cân nhắc vào lệnh ·
        <strong>MUA</strong> = tín hiệu tốt nhưng chắc chắn vừa · <strong>CHỜ</strong> = chưa có tín hiệu rõ.
        Nhấn vào tên mã để xem tin tức liên quan.
      </div>
      <table>
        <thead>
          <tr>
            <th>Mã CK</th>
            <th>Khuyến nghị</th>
            <th class="num">Giá đóng cửa</th>
            <th>
              Độ chắc chắn
              <span class="tip" data-tip="AI tự tin bao nhiêu % về khuyến nghị này. ≥60% (xanh) là đáng tin cậy. Dưới 60% (vàng) nên theo dõi thêm.">?</span>
            </th>
            <th class="num">
              Sức mạnh giá
              <span class="tip" data-tip="Chỉ số RSI (0-100): dưới 30 💚 = giá đang rẻ, có thể phục hồi. Trên 70 🔥 = giá đang cao, cẩn thận mua đỉnh. 30-70 = bình thường.">?</span>
            </th>
            <th class="num">
              ±5 ngày
              <span class="tip" data-tip="Giá cổ phiếu đã tăng hay giảm bao nhiêu % trong 5 ngày vừa qua.">?</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {all_rows}
        </tbody>
      </table>
    </div>

    <div style="display:flex;flex-direction:column;gap:20px;">
      <div class="card">
        <div class="card-title">
          📊 Phân bố độ chắc chắn
          <span class="tip" data-tip="Biểu đồ cho thấy bao nhiêu mã đang có độ chắc chắn ở mức nào. Cột màu xanh/teal (≥60%) là vùng đáng hành động.">?</span>
        </div>
        <div class="chart-wrap"><canvas id="confChart"></canvas></div>
      </div>

      <div class="card">
        <div class="card-title">
          📰 Tin tức trong 24h
          <span class="tip" data-tip="Các mã VN30 được báo chí nhắc đến trong 24h qua. Nhấn vào tên mã để đọc tiêu đề các bài báo.">?</span>
        </div>
        {news_items}
      </div>

      <div class="card">
        <div class="card-title">💡 Hướng dẫn sử dụng</div>
        <div style="font-size:12px;color:#94a3b8;line-height:1.8">
          <div style="margin-bottom:8px">📌 <strong style="color:#cbd5e1">Ai nên dùng dashboard này?</strong><br>Nhà đầu tư mới muốn có gợi ý ban đầu khi chọn cổ phiếu trong rổ VN30.</div>
          <div style="margin-bottom:8px">⚠️ <strong style="color:#cbd5e1">Lưu ý quan trọng</strong><br>AI chỉ là công cụ hỗ trợ. Không có mô hình nào đúng 100%. Luôn quản lý rủi ro và chỉ đầu tư số tiền bạn sẵn sàng mất.</div>
          <div>🕓 <strong style="color:#cbd5e1">Cập nhật lúc nào?</strong><br>Dashboard tự động chạy sau 15:05 mỗi ngày giao dịch, sau khi thị trường đóng cửa.</div>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
const monthly = {monthly_j};
const cumulative = {cumulative_j};
const confDist = {conf_dist_j};
const tickerArticles = {articles_j};

// Chart defaults
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#1e3a5f';
Chart.defaults.font = {{ family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", size: 11 }};

// Cumulative P&L
new Chart(document.getElementById('cumChart'), {{
  type: 'line',
  data: {{
    labels: cumulative.labels,
    datasets: [{{
      data: cumulative.values,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.08)',
      fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => ' ' + ctx.parsed.y.toFixed(1) + ' triệu VND' }} }} }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 6 }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ callback: v => v + 'M' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// Monthly P&L
new Chart(document.getElementById('monthChart'), {{
  type: 'bar',
  data: {{
    labels: monthly.labels,
    datasets: [{{
      data: monthly.values,
      backgroundColor: monthly.colors,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => ' ' + ctx.parsed.y.toFixed(1) + ' triệu VND' }} }} }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 8 }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ callback: v => v + 'M' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// Confidence distribution
new Chart(document.getElementById('confChart'), {{
  type: 'bar',
  data: {{
    labels: confDist.labels,
    datasets: [{{
      label: 'Số mã',
      data: confDist.values,
      backgroundColor: ['#334155','#334155','#334155','#3b82f6','#10b981','#10b981'],
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => ' ' + ctx.parsed.y + ' mã' }} }} }},
    scales: {{
      x: {{ grid: {{ display: false }} }},
      y: {{ ticks: {{ stepSize: 1 }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// News popup
function showNewsPopup(ticker, event) {{
  const popup = document.getElementById('newsPopup');
  const overlay = document.getElementById('popupOverlay');
  const titleEl = document.getElementById('popupTicker');
  const contentEl = document.getElementById('popupContent');

  titleEl.textContent = ticker + ' — Tin tức gần đây';
  const arts = tickerArticles[ticker] || [];

  if (arts.length === 0) {{
    contentEl.innerHTML = '<p class="no-news-msg">Chưa có tin tức nào về ' + ticker + ' trong 3 ngày qua.</p>';
  }} else {{
    contentEl.innerHTML = arts.map(a => {{
      const sentClass = a.sentiment > 0 ? 'sent-pos' : a.sentiment < 0 ? 'sent-neg' : '';
      const sentIcon = a.sentiment > 0 ? '📈' : a.sentiment < 0 ? '📉' : '➡️';
      return `<div class="news-article">
        <a href="${{a.link}}" target="_blank" rel="noopener">${{a.title}}</a>
        <div class="news-article-meta">
          <span>${{a.published}}</span>
          <span class="${{sentClass}}">${{sentIcon}} ${{a.sentiment > 0 ? 'Tích cực' : a.sentiment < 0 ? 'Tiêu cực' : 'Trung lập'}}</span>
        </div>
      </div>`;
    }}).join('');
  }}

  // Position popup near click
  const vw = window.innerWidth, vh = window.innerHeight;
  let x = event.clientX + 12, y = event.clientY + 12;
  popup.style.left = Math.min(x, vw - 400) + 'px';
  popup.style.top = Math.min(y, vh - 300) + 'px';

  popup.classList.add('visible');
  overlay.classList.add('visible');
}}

document.getElementById('popupClose').addEventListener('click', closePopup);
document.getElementById('popupOverlay').addEventListener('click', closePopup);
function closePopup() {{
  document.getElementById('newsPopup').classList.remove('visible');
  document.getElementById('popupOverlay').classList.remove('visible');
}}

document.addEventListener('click', function(e) {{
  const badge = e.target.closest('.news-badge-link');
  if (badge) {{
    e.preventDefault();
    showNewsPopup(badge.dataset.ticker, e);
    return;
  }}
  // news section ticker-link still opens popup
  const el = e.target.closest('.news-ticker-name');
  if (el) {{
    e.preventDefault();
    showNewsPopup(el.dataset.ticker, e);
  }}
}});
</script>
</body>
</html>"""


def build_dashboard() -> Path:
    print("Loading data...")
    signals = load_latest_signals()
    trades = load_backtest()
    news = load_latest_news()
    articles = load_latest_articles()
    tracker_report = load_tracker_report()

    print(f"  Signals: {len(signals)} rows")
    print(f"  Trades:  {len(trades)} rows")
    print(f"  News:    {len(news)} tickers")
    print(f"  Articles: {sum(len(v) for v in articles.values())} total")
    has_score = bool(tracker_report.get("prediction_score", {}).get("score"))
    print(f"  Tracker: {'score=' + str(tracker_report['prediction_score']['score']) if has_score else 'no data yet'}")

    html = generate_html(signals, trades, news, articles, tracker_report)
    path = DASHBOARD_DIR / "index.html"
    path.write_text(html, encoding="utf-8")
    print(f"\n✅ Dashboard → {path}")
    return path


if __name__ == "__main__":
    path = build_dashboard()
    import subprocess
    subprocess.run(["open", str(path)])
