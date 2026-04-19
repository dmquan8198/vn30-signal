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
    colors = ["#22C55E" if v >= 0 else "#ef4444" for v in monthly["pnl"]]
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
    tickers = []
    for i in range(len(edges) - 1):
        mask = (signals["confidence"] >= edges[i]) & (signals["confidence"] < edges[i+1])
        counts.append(int(mask.sum()))
        tickers.append(signals[mask]["ticker"].tolist() if "ticker" in signals.columns else [])
    return {"labels": bins, "values": counts, "tickers": tickers}


def build_tracker_section(report: dict) -> str:
    """Compact tracker — chỉ hiển thị 3 con số quan trọng nhất."""
    if not report or not report.get("prediction_score", {}).get("score"):
        return ""

    score_data = report["prediction_score"]
    s = score_data["score"]
    raw = score_data.get("raw", {})
    hit_rate = raw.get("hit_rate", 0)
    avg_return = raw.get("avg_return", 0)
    n_resolved = raw.get("n_resolved", 0)

    if s >= 70:
        grade_color = "#10b981"
    elif s >= 60:
        grade_color = "#f59e0b"
    elif s >= 50:
        grade_color = "#f97316"
    else:
        grade_color = "#ef4444"

    ret_color = "#10b981" if avg_return >= 0 else "#ef4444"

    return f"""
    <div class="card" style="margin-bottom:24px">
      <div class="card-title">
        📡 Hiệu quả dự đoán gần đây
        <span class="tip" data-tip="Đánh giá dựa trên {n_resolved} tín hiệu BUY đã có kết quả thực tế (khác với backtest: đây là kết quả live từ khi hệ thống vận hành).">?</span>
      </div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:stretch">
        <div style="background:#0f172a;border-radius:8px;padding:12px 20px;text-align:center;min-width:100px">
          <div style="font-size:11px;color:#64748b;margin-bottom:4px">Điểm tổng thể</div>
          <div style="font-size:36px;font-weight:800;color:{grade_color};line-height:1">{s:.0f}<span style="font-size:13px;color:#64748b">/100</span></div>
        </div>
        <div style="background:#0f172a;border-radius:8px;padding:12px 20px;text-align:center;min-width:100px">
          <div style="font-size:11px;color:#64748b;margin-bottom:4px">Tỷ lệ đúng</div>
          <div style="font-size:28px;font-weight:700;color:#f8fafc">{hit_rate:.0%}</div>
          <div style="font-size:11px;color:#64748b">{n_resolved} tín hiệu</div>
        </div>
        <div style="background:#0f172a;border-radius:8px;padding:12px 20px;text-align:center;min-width:100px">
          <div style="font-size:11px;color:#64748b;margin-bottom:4px">Lãi TB thực tế</div>
          <div style="font-size:28px;font-weight:700;color:{ret_color}">{avg_return:+.1%}</div>
          <div style="font-size:11px;color:#64748b">mỗi tín hiệu</div>
        </div>
      </div>
    </div>"""


def build_regime_badge(signals: pd.DataFrame) -> str:
    """Build regime badge HTML — tính trực tiếp từ VNINDEX nếu signals chưa có regime cols."""
    regime_state = 2
    trend = 0.0
    vol_z = 0.0

    if not signals.empty and "regime_state" in signals.columns:
        regime_state = int(signals["regime_state"].iloc[0])
        trend = float(signals.get("trend_strength", pd.Series([0])).iloc[0])
        vol_z = float(signals.get("regime_volatility_z", pd.Series([0])).iloc[0])
    else:
        # Fallback: compute from VNINDEX directly
        try:
            from src.fetch import load_index
            from src.regime import detect_regime, get_current_regime
            vni = load_index("VNINDEX")
            regime_df = detect_regime(vni)
            cur = get_current_regime(regime_df)
            regime_state = cur["regime_state"]
            trend = cur["trend_strength"]
            vol_z = cur["regime_volatility_z"]
        except Exception:
            pass

    regime_names = {
        0: ("BEAR — Thị trường giảm", "#ef4444", "🔴"),
        1: ("SIDEWAYS — Đi ngang", "#6b7280", "⚪"),
        2: ("BULL — Thị trường tăng", "#10b981", "🟢"),
        3: ("BREAKOUT — Bứt phá", "#f59e0b", "🚀"),
    }
    name, color, icon = regime_names.get(regime_state, ("BULL — Thị trường tăng", "#10b981", "🟢"))

    if trend > 0.02:
        trend_label, trend_color = "Tăng ↑", "#10b981"
    elif trend < -0.02:
        trend_label, trend_color = "Giảm ↓", "#ef4444"
    else:
        trend_label, trend_color = "Đi ngang →", "#94a3b8"

    if vol_z > 1.5:
        vol_label, vol_color = "Cao ⚠️", "#ef4444"
    elif vol_z < -1.0:
        vol_label, vol_color = "Thấp", "#10b981"
    else:
        vol_label, vol_color = "Bình thường", "#94a3b8"

    return f"""
    <div style="display:inline-flex;align-items:center;gap:10px;background:rgba(255,255,255,0.05);
                border:1px solid {color}44;border-radius:10px;padding:8px 16px;margin-left:12px;flex-wrap:wrap">
      <div>
        <div style="font-size:10px;color:#64748b;font-weight:500;margin-bottom:2px">Trạng thái thị trường</div>
        <div style="font-size:13px;font-weight:700;color:{color}">{icon} {name}</div>
      </div>
      <div style="border-left:1px solid #1e3a5f;padding-left:10px">
        <div style="font-size:10px;color:#64748b;margin-bottom:2px" title="Xu hướng giá dựa trên MA20 so với MA50 của VNINDEX">Xu hướng</div>
        <div style="font-size:12px;font-weight:600;color:{trend_color}">{trend_label}</div>
      </div>
      <div style="border-left:1px solid #1e3a5f;padding-left:10px">
        <div style="font-size:10px;color:#64748b;margin-bottom:2px" title="Mức độ biến động so với lịch sử: cao = thị trường đang có nhiều biến động">Biến động</div>
        <div style="font-size:12px;font-weight:600;color:{vol_color}">{vol_label}</div>
      </div>
    </div>"""


def build_portfolio_section(signals: pd.DataFrame) -> str:
    """Build portfolio allocation section using PortfolioSizer."""
    try:
        from portfolio.sizing import PortfolioSizer
        sizer = PortfolioSizer(total_capital=50_000_000)
        sized = sizer.select_trades(signals)
        summary = sizer.summary(sized)
    except Exception:
        return ""

    if not summary["positions"]:
        return ""

    rows = ""
    for p in summary["positions"]:
        size_m = p["position_size_vnd"] / 1_000_000
        rows += f"""
        <tr>
          <td style="font-weight:700;color:#f8fafc">{p['ticker']}</td>
          <td style="color:#10b981">{p['confidence']:.0%}</td>
          <td style="color:#94a3b8">{p['sector']}</td>
          <td style="color:#f59e0b;font-weight:600">{size_m:.1f}M</td>
        </tr>"""

    util = summary["utilization_pct"]
    return f"""
    <div class="card" style="margin-bottom:24px">
      <div class="card-title">💼 Đề xuất phân bổ vốn hôm nay</div>
      <div style="display:flex;gap:20px;margin-bottom:14px">
        <div style="background:#0f172a;border-radius:8px;padding:10px 16px">
          <div style="font-size:11px;color:#64748b">Vị thế</div>
          <div style="font-size:22px;font-weight:700;color:#f8fafc">{summary['n_positions']}<span style="font-size:13px;color:#64748b">/{summary['max_positions']}</span></div>
        </div>
        <div style="background:#0f172a;border-radius:8px;padding:10px 16px">
          <div style="font-size:11px;color:#64748b">Vốn triển khai</div>
          <div style="font-size:22px;font-weight:700;color:#f59e0b">{summary['deployed_vnd']/1_000_000:.0f}M</div>
        </div>
        <div style="background:#0f172a;border-radius:8px;padding:10px 16px">
          <div style="font-size:11px;color:#64748b">Sử dụng</div>
          <div style="font-size:22px;font-weight:700;color:#{'10b981' if util>=80 else '94a3b8'}">{util:.0f}%</div>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead>
          <tr style="color:#64748b;font-size:11px;text-transform:uppercase;border-bottom:1px solid #1e3a5f">
            <th style="text-align:left;padding:6px 4px">Mã CK</th>
            <th style="text-align:left;padding:6px 4px">Conf</th>
            <th style="text-align:left;padding:6px 4px">Sector</th>
            <th style="text-align:left;padding:6px 4px">Vốn</th>
          </tr>
        </thead>
        <tbody style="color:#94a3b8">
          {rows}
        </tbody>
      </table>
      <div style="font-size:11px;color:#475569;margin-top:8px">* Dựa trên tổng vốn 50M VND, tối đa 5 vị thế, scaling theo confidence.</div>
    </div>"""


def generate_html(signals: pd.DataFrame, trades: pd.DataFrame, news: pd.DataFrame, articles: dict, tracker_report: dict = None) -> str:
    date_str = signals["date"].iloc[0] if not signals.empty else datetime.today().strftime("%Y-%m-%d")
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")
    market_bull = int(signals["vni_bull"].iloc[0]) if not signals.empty and "vni_bull" in signals.columns else 0
    market_label = "Thị trường đang tăng 🟢" if market_bull else "Thị trường đang giảm 🔴"
    market_color = "#10b981" if market_bull else "#ef4444"
    regime_badge = build_regime_badge(signals)
    portfolio_section = build_portfolio_section(signals)

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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  /* ── Design tokens ─────────────────────────────────── */
  :root {{
    --bg:          #020617;
    --bg-card:     #0E1223;
    --bg-hover:    #131929;
    --border:      #334155;
    --border-sub:  #1e293b;
    --accent:      #22C55E;
    --accent-dim:  rgba(34,197,94,0.15);
    --accent-glow: rgba(34,197,94,0.25);
    --blue:        #60a5fa;
    --blue-dim:    rgba(96,165,250,0.15);
    --red:         #ef4444;
    --red-dim:     rgba(239,68,68,0.12);
    --text-primary:   #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted:     #475569;
    --font-ui:   'Fira Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'Fira Code', 'Fira Mono', monospace;
    --radius:    12px;
    --ease-out:  150ms cubic-bezier(0.16,1,0.3,1);
  }}

  /* ── Reset ─────────────────────────────────────────── */
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:var(--font-ui); background:var(--bg); color:var(--text-primary); min-height:100dvh; line-height:1.5; }}

  /* ── Accessibility ─────────────────────────────────── */
  :focus-visible {{ outline:2px solid var(--accent); outline-offset:3px; border-radius:4px; }}
  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ transition-duration:0ms !important; animation-duration:0ms !important; }}
  }}

  /* ── Header ────────────────────────────────────────── */
  .header {{ background:linear-gradient(135deg,#0E1223,var(--bg)); border-bottom:1px solid var(--border); padding:20px 32px; display:flex; align-items:center; justify-content:space-between; }}
  .header h1 {{ font-size:22px; font-weight:700; color:var(--text-primary); letter-spacing:-0.5px; }}
  .header h1 span {{ color:var(--accent); }}
  .meta {{ font-size:13px; color:var(--text-muted); margin-top:4px; }}
  .container {{ max-width:1400px; margin:0 auto; padding:24px 32px; }}

  /* ── Stat cards ────────────────────────────────────── */
  .stats-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:16px; margin-bottom:24px; }}
  .stat-card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:20px; position:relative; transition:border-color var(--ease-out), background var(--ease-out); }}
  .stat-card:hover {{ background:var(--bg-hover); border-color:#4b6480; }}
  .stat-label {{ font-size:11px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.6px; margin-bottom:8px; display:flex; align-items:center; gap:5px; font-weight:500; }}
  .stat-value {{ font-size:28px; font-weight:700; color:var(--text-primary); line-height:1; font-variant-numeric:tabular-nums; font-family:var(--font-mono); }}
  .stat-sub {{ font-size:12px; color:var(--text-muted); margin-top:8px; }}

  /* ── Tooltip ───────────────────────────────────────── */
  .tip {{ display:inline-flex; align-items:center; justify-content:center; width:15px; height:15px; border-radius:50%; background:#1e293b; color:var(--text-muted); font-size:10px; cursor:help; position:relative; flex-shrink:0; transition:background var(--ease-out); }}
  .tip:hover {{ background:#2d3f55; }}
  .tip:hover::after {{ content:attr(data-tip); position:absolute; bottom:calc(100% + 6px); left:50%; transform:translateX(-50%); background:#1a2438; border:1px solid var(--border); color:var(--text-primary); font-size:12px; padding:10px 14px; border-radius:8px; white-space:normal; max-width:340px; min-width:180px; z-index:100; line-height:1.6; text-transform:none; letter-spacing:0; font-weight:400; box-shadow:0 8px 32px rgba(0,0,0,0.5); }}
  .tip:hover::before {{ content:''; position:absolute; bottom:calc(100% + 1px); left:50%; transform:translateX(-50%); border:5px solid transparent; border-top-color:var(--border); }}

  /* ── Layout ────────────────────────────────────────── */
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }}
  .grid-3 {{ display:grid; grid-template-columns:2fr 1fr; gap:20px; margin-bottom:24px; }}
  .card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:20px; transition:border-color var(--ease-out); }}
  .card:hover {{ border-color:#4b6480; }}
  .card-title {{ font-size:12px; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.7px; margin-bottom:16px; display:flex; align-items:center; gap:8px; }}
  .chart-wrap {{ position:relative; height:200px; }}
  .chart-wrap-tall {{ position:relative; height:240px; }}

  /* ── Table ─────────────────────────────────────────── */
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; padding:10px 12px; color:var(--text-muted); font-weight:500; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; border-bottom:1px solid var(--border); }}
  th .tip {{ margin-left:4px; }}
  td {{ padding:10px 12px; border-bottom:1px solid var(--border-sub); vertical-align:middle; font-variant-numeric:tabular-nums; }}
  .row-buy-strong {{ background:rgba(34,197,94,0.05); }}
  .row-buy {{ background:rgba(96,165,250,0.04); }}
  tr {{ transition:background var(--ease-out); }}
  tr:hover {{ background:rgba(255,255,255,0.04); cursor:pointer; }}
  .ticker-cell {{ font-weight:700; font-size:14px; }}
  .ticker-link {{ color:var(--blue); cursor:pointer; text-decoration:none; border-bottom:1px dashed var(--border); transition:color var(--ease-out), border-color var(--ease-out); }}
  .ticker-link:hover {{ color:#93c5fd; border-bottom-color:var(--blue); }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; font-family:var(--font-mono); }}

  /* ── Badges ────────────────────────────────────────── */
  .badge {{ display:inline-block; padding:3px 8px; border-radius:6px; font-size:11px; font-weight:600; }}
  .badge-buy {{ background:var(--accent-dim); color:var(--accent); border:1px solid var(--accent-glow); }}
  .badge-buy-low {{ background:var(--blue-dim); color:var(--blue); border:1px solid rgba(96,165,250,0.2); }}
  .badge-hold {{ background:rgba(100,116,139,0.15); color:var(--text-secondary); border:1px solid rgba(100,116,139,0.2); }}
  .news-badge {{ display:inline-block; padding:2px 6px; border-radius:4px; font-size:10px; background:rgba(245,158,11,0.1); color:#f59e0b; }}
  .news-badge-link {{ cursor:pointer; border-bottom:1px dashed rgba(245,158,11,0.4); transition:background var(--ease-out); }}
  .news-badge-link:hover {{ background:rgba(245,158,11,0.2); }}

  /* ── Confidence bar ────────────────────────────────── */
  .conf-bar {{ display:flex; align-items:center; gap:8px; min-width:110px; }}
  .conf-track {{ flex:1; height:6px; background:#1e293b; border-radius:3px; overflow:hidden; min-width:60px; }}
  .conf-fill {{ height:100%; border-radius:3px; transition:width 300ms cubic-bezier(0.16,1,0.3,1); }}

  /* ── Regime badge ──────────────────────────────────── */
  .regime-badge {{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; }}
  .market-sent-big {{ font-size:28px; font-weight:700; font-family:var(--font-mono); font-variant-numeric:tabular-nums; }}

  /* ── News popup ────────────────────────────────────── */
  .news-popup {{ display:none; position:fixed; z-index:1000; background:#111827; border:1px solid var(--border); border-radius:var(--radius); padding:16px; max-width:380px; width:90%; box-shadow:0 24px 64px rgba(0,0,0,0.7); }}
  .news-popup.visible {{ display:block; }}
  .news-popup-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
  .news-popup-ticker {{ font-size:18px; font-weight:700; color:var(--text-primary); font-family:var(--font-mono); }}
  .news-popup-close {{ color:var(--text-muted); cursor:pointer; font-size:18px; line-height:1; padding:4px; transition:color var(--ease-out); border-radius:4px; }}
  .news-popup-close:hover {{ color:var(--text-primary); }}
  .news-article {{ padding:10px 0; border-bottom:1px solid var(--border-sub); }}
  .news-article:last-child {{ border-bottom:none; }}
  .news-article a {{ color:#93c5fd; text-decoration:none; font-size:13px; line-height:1.5; display:block; transition:color var(--ease-out); }}
  .news-article a:hover {{ color:var(--blue); text-decoration:underline; }}
  .news-article-meta {{ font-size:11px; color:var(--text-muted); margin-top:3px; display:flex; gap:8px; font-variant-numeric:tabular-nums; }}
  .sent-pos {{ color:var(--accent); }}
  .sent-neg {{ color:var(--red); }}
  .no-news-msg {{ color:var(--text-muted); font-size:13px; padding:8px 0; }}
  .popup-overlay {{ display:none; position:fixed; inset:0; z-index:999; }}
  .popup-overlay.visible {{ display:block; }}

  /* ── News coverage ─────────────────────────────────── */
  .news-ticker-row {{ display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid var(--border-sub); cursor:pointer; transition:background var(--ease-out), padding-left var(--ease-out); border-radius:0; }}
  .news-ticker-row:last-child {{ border-bottom:none; }}
  .news-ticker-row:hover {{ background:rgba(255,255,255,0.03); border-radius:6px; padding-left:6px; }}
  .news-ticker-name {{ font-weight:600; color:var(--text-primary); }}

  /* ── Explain boxes ─────────────────────────────────── */
  .explain-box {{ background:#0a1120; border:1px solid var(--border-sub); border-radius:8px; padding:12px 16px; font-size:12px; color:var(--text-secondary); line-height:1.6; margin-bottom:16px; }}
  .explain-box strong {{ color:#cbd5e1; }}

  /* ── Responsive ────────────────────────────────────── */
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
    <div class="meta">Tín hiệu ngày {date_str} · Tạo lúc {generated_at} · Mô hình AI phân tích 60 chỉ số kỹ thuật</div>
  </div>
  <div style="text-align:right;display:flex;align-items:center;gap:10px">
    <span class="regime-badge" style="background:rgba({"16,185,129" if market_bull else "239,68,68"},0.15);color:{market_color};border:1px solid rgba({"16,185,129" if market_bull else "239,68,68"},0.3)">{market_label}</span>
    {regime_badge}
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
      <div class="stat-value" style="color:var(--accent)">{n_buy}</div>
      <div class="stat-sub">{n_actionable} mã AI tự tin cao ⭐</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">
        Lợi nhuận backtest
        <span class="tip" data-tip="Tổng lợi nhuận giả định nếu mua 10 triệu/lần theo tín hiệu BUY từ năm 2023 đến nay, đã trừ phí giao dịch 0.15%/chiều.">?</span>
      </div>
      <div class="stat-value" style="color:{'var(--accent)' if total_pnl>=0 else 'var(--red)'}">{total_pnl:+.1f}M</div>
      <div class="stat-sub">Kiểm nghiệm 2023–2026</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">
        Tỷ lệ thắng
        <span class="tip" data-tip="Tỷ lệ các lần mua theo tín hiệu AI có lời sau 5 ngày — tính trên toàn bộ backtest lịch sử 2023–2026 ({total_trades} lệnh). Xem 'Hiệu quả dự đoán gần đây' để biết kết quả live.">?</span>
      </div>
      <div class="stat-value" style="color:var(--blue)">{win_rate:.1f}%</div>
      <div class="stat-sub">{total_trades} lần mua trong backtest</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">
        Lãi TB mỗi lần mua
        <span class="tip" data-tip="Lợi nhuận trung bình mỗi lần mua theo tín hiệu AI, tính theo % giá cổ phiếu sau 5 ngày nắm giữ, đã trừ phí.">?</span>
      </div>
      <div class="stat-value" style="color:{'var(--accent)' if avg_ret>=0 else 'var(--red)'}">{avg_ret:+.2f}%</div>
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

  <!-- Portfolio Allocation -->
  {portfolio_section}

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
Chart.defaults.borderColor = '#1e293b';
Chart.defaults.font = {{ family: "'Fira Sans', -apple-system, BlinkMacSystemFont, sans-serif", size: 11 }};

// Cumulative P&L
new Chart(document.getElementById('cumChart'), {{
  type: 'line',
  data: {{
    labels: cumulative.labels,
    datasets: [{{
      data: cumulative.values,
      borderColor: '#22C55E',
      backgroundColor: 'rgba(34,197,94,0.07)',
      fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => ' ' + ctx.parsed.y.toFixed(1) + ' triệu VND' }} }} }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 6 }}, grid: {{ color: '#0f172a' }} }},
      y: {{ ticks: {{ callback: v => v + 'M' }}, grid: {{ color: '#0f172a' }} }}
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
      x: {{ ticks: {{ maxTicksLimit: 8 }}, grid: {{ color: '#0f172a' }} }},
      y: {{ ticks: {{ callback: v => v + 'M' }}, grid: {{ color: '#0f172a' }} }}
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
      backgroundColor: ['#1e293b','#1e293b','#1e293b','#60a5fa','#22C55E','#22C55E'],
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => {{
            const n = ctx.parsed.y;
            const tickers = (confDist.tickers || [])[ctx.dataIndex] || [];
            if (n === 0) return ' Không có mã nào';
            return tickers.length ? ' ' + tickers.join(', ') : ' ' + n + ' mã';
          }}
        }}
      }}
    }},
    scales: {{
      x: {{ grid: {{ display: false }} }},
      y: {{ ticks: {{ stepSize: 1 }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// News popup
const geoRisk = tickerArticles['_geo_risk'] || {{}};

function showGeoPopup(event) {{
  const popup = document.getElementById('newsPopup');
  const overlay = document.getElementById('popupOverlay');
  document.getElementById('popupTicker').textContent = '🌍 Cảnh báo địa chính trị';
  const heads = (geoRisk.headlines || []);
  const lvlMap = {{1:'🟡 Thấp', 2:'🟠 Trung bình', 3:'🔴 Cao'}};
  const lvlText = lvlMap[geoRisk.level] || '';
  let html = `<div style="font-size:12px;color:#f59e0b;margin-bottom:10px;padding:6px 10px;background:rgba(245,158,11,0.1);border-radius:6px">
    Mức độ: <strong>${{lvlText}}</strong> &nbsp;·&nbsp; ${{geoRisk.summary || ''}}
  </div>`;
  if (heads.length === 0) {{
    html += '<p class="no-news-msg">Không tìm thấy tiêu đề cụ thể.</p>';
  }} else {{
    html += heads.map(a => `<div class="news-article">
      ${{a.link ? `<a href="${{a.link}}" target="_blank" rel="noopener">${{a.title}}</a>` : `<span style="color:#e2e8f0">${{a.title}}</span>`}}
      <div class="news-article-meta"><span>${{a.published}}</span><span style="color:#64748b">${{a.source}}</span></div>
    </div>`).join('');
  }}
  document.getElementById('popupContent').innerHTML = html;
  positionAndShow(event);
}}

function showNewsPopup(ticker, event) {{
  const popup = document.getElementById('newsPopup');
  document.getElementById('popupTicker').textContent = ticker + ' — Tin tức gần đây';
  const arts = (tickerArticles[ticker] || []).filter(a => !a._geo);

  if (arts.length === 0) {{
    document.getElementById('popupContent').innerHTML =
      '<p class="no-news-msg">Chưa có tin tức nào về ' + ticker + ' trong 3 ngày qua.</p>';
  }} else {{
    document.getElementById('popupContent').innerHTML = arts.map(a => {{
      const sentClass = a.sentiment > 0 ? 'sent-pos' : a.sentiment < 0 ? 'sent-neg' : '';
      const sentIcon  = a.sentiment > 0 ? '📈' : a.sentiment < 0 ? '📉' : '➡️';
      const srcBadge  = a.source === 'intl' ? '<span style="color:#3b82f6;font-size:10px">🌐 Quốc tế</span>' : '';
      return `<div class="news-article">
        <a href="${{a.link}}" target="_blank" rel="noopener">${{a.title}}</a>
        <div class="news-article-meta">
          <span>${{a.published}}</span>
          <span class="${{sentClass}}">${{sentIcon}} ${{a.sentiment > 0 ? 'Tích cực' : a.sentiment < 0 ? 'Tiêu cực' : 'Trung lập'}}</span>
          ${{srcBadge}}
        </div>
      </div>`;
    }}).join('');
  }}
  positionAndShow(event);
}}

function positionAndShow(event) {{
  const popup = document.getElementById('newsPopup');
  const vw = window.innerWidth, vh = window.innerHeight;
  let x = event.clientX + 12, y = event.clientY + 12;
  popup.style.left = Math.min(x, vw - 420) + 'px';
  popup.style.top  = Math.min(y, vh - 320) + 'px';
  popup.classList.add('visible');
  document.getElementById('popupOverlay').classList.add('visible');
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
    const tag = badge.textContent || '';
    if (tag.includes('geo:') || tag.includes('🌍')) {{
      showGeoPopup(e);
    }} else {{
      showNewsPopup(badge.dataset.ticker, e);
    }}
    return;
  }}
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
