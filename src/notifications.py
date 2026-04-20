"""Email notifications for VN30 Signal — uses Gmail SMTP with App Password."""

import smtplib
import os
import json
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SENDER    = os.getenv("NOTIFY_EMAIL_SENDER")
PASSWORD  = os.getenv("NOTIFY_EMAIL_PASSWORD")
RECEIVERS = os.getenv("NOTIFY_EMAIL_TO", "").split(",")

DASHBOARD_URL = "https://dmquan8198.github.io/vn30-signal/dashboard/"

# ── Design tokens (nhất quán với dashboard) ────────────────────────────────
BG         = "#020617"
BG_CARD    = "#0E1223"
BG_CARD2   = "#131929"
BORDER     = "#334155"
ACCENT     = "#22C55E"   # green — BUY / positive
BLUE       = "#60a5fa"
RED        = "#ef4444"
AMBER      = "#f59e0b"
TEXT       = "#f1f5f9"
TEXT_SUB   = "#94a3b8"
TEXT_MUTED = "#475569"


def _load_tracker() -> dict:
    p = Path(__file__).parent.parent / "data" / "tracker" / "latest_report.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _market_summary_section(df, tracker: dict) -> str:
    """Phần tóm tắt thị trường — hiển thị ngay dưới header."""
    # Regime
    regime_state = int(df["regime_state"].iloc[0]) if "regime_state" in df.columns and not df.empty else -1
    regime_map = {
        0: ("BEAR", "🔴", RED,    "Thị trường đang giảm — thận trọng"),
        1: ("SIDEWAYS", "⚪", TEXT_SUB, "Thị trường đi ngang — chờ tín hiệu rõ"),
        2: ("BULL", "🟢", ACCENT, "Thị trường đang tăng — tích cực"),
        3: ("BREAKOUT", "🚀", BLUE, "Breakout — động lực mạnh"),
    }
    regime_name, regime_icon, regime_color, regime_desc = regime_map.get(
        regime_state, ("—", "—", TEXT_MUTED, "Chưa xác định")
    )

    # Market sentiment
    mkt_sent = 0.0
    if not df.empty and "market_sentiment_1d" in df.columns:
        try:
            mkt_sent = float(df["market_sentiment_1d"].iloc[0])
        except Exception:
            pass
    if mkt_sent > 0.3:
        sent_icon, sent_label, sent_color = "😊", "Tích cực", ACCENT
    elif mkt_sent < -0.3:
        sent_icon, sent_label, sent_color = "😟", "Tiêu cực", RED
    else:
        sent_icon, sent_label, sent_color = "😐", "Trung lập", TEXT_SUB

    # Geo risk
    geo_level = 0
    if not df.empty and "geo_risk_level" in df.columns:
        try:
            geo_level = int(df["geo_risk_level"].iloc[0])
        except Exception:
            pass
    geo_map = {0: ("—", TEXT_MUTED), 1: ("🟡 Thấp", AMBER), 2: ("🟠 Trung bình", AMBER), 3: ("🔴 Cao", RED)}
    geo_label, geo_color = geo_map.get(geo_level, ("—", TEXT_MUTED))

    # Signal counts
    n_buy_strong = len(df[(df["signal"] == "BUY") & (df["confidence"] >= 0.6)]) if not df.empty else 0
    n_buy_low    = len(df[(df["signal"] == "BUY") & (df["confidence"] < 0.6)]) if not df.empty else 0
    n_hold       = len(df[df["signal"] == "HOLD"]) if not df.empty else 0

    # Tracker
    ps = tracker.get("prediction_score", {})
    raw = ps.get("raw", {})
    score     = ps.get("score", None)
    hit_rate  = raw.get("hit_rate", None)
    avg_ret   = raw.get("avg_return", None)
    n_resolved = raw.get("n_resolved", 0)
    score_color = ACCENT if score and score >= 55 else AMBER if score and score >= 40 else RED
    score_str = f"{score:.0f}/100" if score is not None else "—"
    hit_str   = f"{hit_rate:.1%}" if hit_rate is not None else "—"
    ret_str   = f"{avg_ret*100:+.2f}%" if avg_ret is not None else "—"
    ret_color = ACCENT if avg_ret and avg_ret > 0 else RED

    # Dynamic confidence threshold
    threshold_col = AMBER if regime_state == 0 else ACCENT
    threshold_val = "73%" if regime_state == 0 else "60%"

    return f"""
  <!-- Market Summary -->
  <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px">
    <tr>
      <td colspan="2" style="padding-bottom:10px">
        <div style="background:{BG_CARD2};border:1px solid {BORDER};border-radius:10px;padding:16px">

          <!-- Regime row -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:12px">
            <tr>
              <td>
                <span style="font-size:18px;font-weight:700;color:{regime_color}">{regime_icon} {regime_name}</span>
                <span style="font-size:12px;color:{TEXT_MUTED};margin-left:8px">{regime_desc}</span>
              </td>
              <td align="right">
                <span style="font-size:11px;color:{TEXT_MUTED}">Threshold: </span>
                <span style="font-size:13px;font-weight:700;color:{threshold_col}">{threshold_val}</span>
              </td>
            </tr>
          </table>

          <!-- 4 metric boxes -->
          <table width="100%" cellpadding="0" cellspacing="6">
            <tr>
              <!-- Signal counts -->
              <td width="25%" style="background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;padding:10px 12px;text-align:center">
                <div style="font-size:10px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">BUY hôm nay</div>
                <div style="font-size:22px;font-weight:700;color:{ACCENT if n_buy_strong > 0 else TEXT_SUB}">{n_buy_strong}</div>
                <div style="font-size:11px;color:{TEXT_MUTED}">⭐ tự tin cao</div>
                {f'<div style="font-size:11px;color:{BLUE};margin-top:2px">+{n_buy_low} theo dõi</div>' if n_buy_low > 0 else ''}
              </td>
              <!-- Sentiment -->
              <td width="25%" style="background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;padding:10px 12px;text-align:center">
                <div style="font-size:10px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Tin tức</div>
                <div style="font-size:22px;font-weight:700;color:{sent_color}">{sent_icon}</div>
                <div style="font-size:11px;color:{sent_color}">{sent_label}</div>
                <div style="font-size:11px;color:{TEXT_MUTED};margin-top:2px">{mkt_sent:+.2f}</div>
              </td>
              <!-- Geo risk -->
              <td width="25%" style="background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;padding:10px 12px;text-align:center">
                <div style="font-size:10px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Địa chính trị</div>
                <div style="font-size:16px;font-weight:700;color:{geo_color};margin:4px 0">{geo_label}</div>
                <div style="font-size:11px;color:{TEXT_MUTED}">Risk level {geo_level}/3</div>
              </td>
              <!-- Tracker score -->
              <td width="25%" style="background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;padding:10px 12px;text-align:center">
                <div style="font-size:10px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Độ chính xác AI</div>
                <div style="font-size:22px;font-weight:700;color:{score_color}">{score_str}</div>
                <div style="font-size:11px;color:{TEXT_MUTED}">hit {hit_str} · avg <span style="color:{ret_color}">{ret_str}</span></div>
              </td>
            </tr>
          </table>

        </div>
      </td>
    </tr>
  </table>"""


def _build_html(df, date_str: str, regime_state: int, tracker: dict) -> str:
    import pandas as pd

    buy  = df[df["signal"] == "BUY"].sort_values("confidence", ascending=False)
    hold = df[df["signal"] == "HOLD"]

    regime_map = {0: ("BEAR 🔴", RED), 1: ("SIDEWAYS ⚪", TEXT_SUB), 2: ("BULL 🟢", ACCENT), 3: ("BREAKOUT 🚀", BLUE)}
    regime_label, regime_color = regime_map.get(regime_state, ("—", TEXT_MUTED))

    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")

    market_summary = _market_summary_section(df, tracker)

    # ── BUY table rows ──────────────────────────────────────────────────────
    def _conf_bar(conf: float) -> str:
        color = ACCENT if conf >= 0.6 else AMBER
        pct = int(conf * 100)
        filled = int(pct / 10)  # 10 blocks max
        bar = "█" * filled + "░" * (10 - filled)
        return f'<span style="font-family:monospace;color:{color};font-size:12px">{bar}</span> <span style="color:{color};font-weight:700">{conf:.0%}</span>'

    rows_html = ""
    for _, r in buy.iterrows():
        conf    = float(r["confidence"])
        close   = float(r["close"])
        ret5    = float(r.get("ret_5d", 0))
        tag     = str(r.get("news_tag", "") or "")
        ticker  = r["ticker"]
        ret_col = ACCENT if ret5 > 0 else RED
        tag_parts = "  ".join(p.strip() for p in tag.split("  ") if p.strip())
        badge = "⭐ NÊN MUA" if conf >= 0.6 else "MUA"
        badge_bg = f"rgba(34,197,94,0.15)" if conf >= 0.6 else f"rgba(96,165,250,0.15)"
        badge_color = ACCENT if conf >= 0.6 else BLUE

        rows_html += f"""
        <tr style="border-bottom:1px solid {BORDER}">
          <td style="padding:10px 12px">
            <a href="https://finance.vietstock.vn/{ticker}/thong-tin-co-ban.htm"
               style="color:{BLUE};font-weight:700;font-size:14px;text-decoration:none">{ticker}</a>
            <div style="margin-top:4px">
              <span style="display:inline-block;background:{badge_bg};color:{badge_color};
                           border:1px solid {badge_color};border-radius:4px;
                           padding:2px 7px;font-size:10px;font-weight:600">{badge}</span>
              {"<br><span style='font-size:11px;color:" + AMBER + ";margin-top:4px;display:inline-block'>" + tag_parts + "</span>" if tag_parts else ""}
            </div>
          </td>
          <td style="padding:10px 12px;text-align:right;font-family:monospace;color:{TEXT};font-size:13px">{close:,.0f}đ</td>
          <td style="padding:10px 12px">{_conf_bar(conf)}</td>
          <td style="padding:10px 12px;text-align:right;color:{ret_col};font-weight:600;font-family:monospace">{ret5:+.1f}%</td>
        </tr>"""

    buy_section = f"""
  <h3 style="color:{TEXT_SUB};font-size:12px;text-transform:uppercase;letter-spacing:0.7px;margin:0 0 10px 0">
    ⭐ BUY Signals — {len(buy)} mã
  </h3>
  {"<p style='color:" + TEXT_MUTED + ";font-size:13px;padding:12px 0'>Không có tín hiệu BUY hôm nay. Thị trường chưa rõ tín hiệu.</p>" if buy.empty else f'''
  <table width="100%" cellpadding="0" cellspacing="0"
         style="border-collapse:collapse;font-size:13px;border:1px solid {BORDER};border-radius:8px;overflow:hidden">
    <thead>
      <tr style="background:{BG_CARD2}">
        <th style="padding:8px 12px;text-align:left;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase">Mã / Khuyến nghị</th>
        <th style="padding:8px 12px;text-align:right;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase">Giá đóng cửa</th>
        <th style="padding:8px 12px;text-align:left;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase">Độ chắc chắn</th>
        <th style="padding:8px 12px;text-align:right;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase">±5 ngày</th>
      </tr>
    </thead>
    <tbody style="background:{BG_CARD}">
      {rows_html}
    </tbody>
  </table>'''}"""

    # ── Portfolio section ────────────────────────────────────────────────────
    portfolio_section = ""
    try:
        from portfolio.sizing import PortfolioSizer
        sizer  = PortfolioSizer(total_capital=50_000_000)
        sized  = sizer.select_trades(df)
        summary = sizer.summary(sized)
        if summary["positions"]:
            p_rows = ""
            for p in summary["positions"]:
                vnd = p["position_size_vnd"] / 1_000_000
                p_rows += f"""
                <tr style="border-bottom:1px solid {BORDER}">
                  <td style="padding:8px 12px;font-weight:700;color:{BLUE}">{p['ticker']}</td>
                  <td style="padding:8px 12px;color:{ACCENT}">{p['confidence']:.0%}</td>
                  <td style="padding:8px 12px;color:{TEXT_SUB}">{p.get('sector','—')}</td>
                  <td style="padding:8px 12px;text-align:right;color:{AMBER};font-weight:600;font-family:monospace">{vnd:.1f}M</td>
                </tr>"""
            portfolio_section = f"""
  <h3 style="color:{TEXT_SUB};font-size:12px;text-transform:uppercase;letter-spacing:0.7px;margin:20px 0 10px 0">
    💼 Đề xuất phân bổ vốn (50M)
  </h3>
  <table width="100%" cellpadding="0" cellspacing="0"
         style="border-collapse:collapse;font-size:13px;border:1px solid {BORDER};border-radius:8px;overflow:hidden">
    <thead>
      <tr style="background:{BG_CARD2}">
        <th style="padding:8px 12px;text-align:left;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase">Mã</th>
        <th style="padding:8px 12px;text-align:left;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase">Conf</th>
        <th style="padding:8px 12px;text-align:left;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase">Sector</th>
        <th style="padding:8px 12px;text-align:right;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase">Vốn</th>
      </tr>
    </thead>
    <tbody style="background:{BG_CARD}">{p_rows}</tbody>
  </table>
  <p style="font-size:11px;color:{TEXT_MUTED};margin-top:6px">* Tối đa 5 vị thế, scaling theo confidence</p>"""
    except Exception:
        pass

    # ── HOLD summary ─────────────────────────────────────────────────────────
    hold_tickers = ", ".join(hold["ticker"].tolist()) if not hold.empty else "—"

    # ── Watch list (HOLD + tin tốt) ───────────────────────────────────────────
    watch_list = ""
    if "news_tag" in df.columns:
        watch = df[(df["signal"] == "HOLD") & (df["news_tag"].str.contains("watch|tin tốt", na=False, case=False))]
        if not watch.empty:
            watch_items = " &nbsp;·&nbsp; ".join(
                f'<span style="color:{BLUE};font-weight:600">{r["ticker"]}</span>'
                for _, r in watch.iterrows()
            )
            watch_list = f"""
  <div style="margin-top:12px;padding:10px 14px;background:rgba(96,165,250,0.08);
              border:1px solid rgba(96,165,250,0.2);border-radius:8px;font-size:13px">
    👀 <strong style="color:{BLUE}">WATCH — HOLD có tin tốt:</strong> {watch_items}
  </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>VN30 Signal — {date_str}</title>
</head>
<body style="background:{BG};color:{TEXT};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;margin:0;padding:24px">
  <div style="max-width:620px;margin:0 auto">

    <!-- Header -->
    <div style="border-bottom:1px solid {BORDER};padding-bottom:16px;margin-bottom:20px">
      <h1 style="margin:0;font-size:20px;font-weight:700;color:{TEXT}">
        VN30 <span style="color:{ACCENT}">Signal</span>
      </h1>
      <div style="font-size:13px;color:{TEXT_MUTED};margin-top:4px">
        Tín hiệu ngày <strong style="color:{TEXT}">{date_str}</strong>
        &nbsp;·&nbsp; Gửi lúc {generated_at}
        &nbsp;·&nbsp; <span style="color:{regime_color}">{regime_label}</span>
      </div>
    </div>

    {market_summary}

    <!-- BUY signals -->
    {buy_section}

    {portfolio_section}

    <!-- HOLD -->
    <div style="margin-top:16px;padding:10px 14px;background:{BG_CARD};
                border:1px solid {BORDER};border-radius:8px;font-size:13px;color:{TEXT_MUTED}">
      ⚪ <strong style="color:{TEXT_SUB}">HOLD ({len(hold)} mã):</strong> {hold_tickers}
    </div>

    {watch_list}

    <!-- CTA -->
    <div style="margin-top:24px;text-align:center">
      <a href="{DASHBOARD_URL}"
         style="display:inline-block;background:{ACCENT};color:#000;text-decoration:none;
                padding:12px 28px;border-radius:8px;font-size:14px;font-weight:700;
                letter-spacing:0.2px">
        📊 Xem Dashboard đầy đủ →
      </a>
    </div>

    <!-- Footer -->
    <div style="border-top:1px solid {BORDER};margin-top:24px;padding-top:14px;
                text-align:center;font-size:11px;color:{TEXT_MUTED}">
      VN30 Signal &nbsp;·&nbsp; Cập nhật sau 14:45 mỗi ngày giao dịch
      &nbsp;·&nbsp; Chỉ mang tính tham khảo, không phải tư vấn đầu tư
    </div>

  </div>
</body>
</html>"""


def send_signal_email(df, subject: str | None = None) -> bool:
    """Gửi email tóm tắt tín hiệu. Trả về True nếu thành công."""
    if not SENDER or not PASSWORD or not any(RECEIVERS):
        print("  [notify] Bỏ qua — chưa cấu hình NOTIFY_EMAIL_* trong .env")
        return False

    date_str = str(df["date"].iloc[0]) if not df.empty else str(date.today())
    regime_state = int(df["regime_state"].iloc[0]) if "regime_state" in df.columns and not df.empty else 2
    regime_names = {0: "BEAR 🔴", 1: "SIDEWAYS ⚪", 2: "BULL 🟢", 3: "BREAKOUT 🚀"}
    regime = regime_names.get(regime_state, "BULL 🟢")

    tracker = _load_tracker()

    subject = subject or f"[VN30 Signal] {date_str} — {regime}"
    html_body = _build_html(df, date_str, regime_state, tracker)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SENDER
    msg["To"]      = ", ".join(r.strip() for r in RECEIVERS)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SENDER, PASSWORD)
            server.sendmail(SENDER, [r.strip() for r in RECEIVERS], msg.as_string())
        print(f"  [notify] ✅ Đã gửi email tới: {msg['To']}")
        return True
    except Exception as e:
        print(f"  [notify] ❌ Gửi email thất bại: {e}")
        return False
