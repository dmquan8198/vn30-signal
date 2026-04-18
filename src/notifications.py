"""Email notifications for VN30 Signal — uses Gmail SMTP with App Password."""

import smtplib
import os
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SENDER    = os.getenv("NOTIFY_EMAIL_SENDER")   # your Gmail address
PASSWORD  = os.getenv("NOTIFY_EMAIL_PASSWORD")  # Gmail App Password (16 chars)
RECEIVERS = os.getenv("NOTIFY_EMAIL_TO", "").split(",")  # comma-separated


def _build_html(df, date_str: str, regime: str, regime_state: int = 2) -> str:
    buy = df[df["signal"] == "BUY"].sort_values("confidence", ascending=False)
    hold = df[df["signal"] == "HOLD"]

    regime_color = "#10b981" if regime_state >= 2 else "#6b7280" if regime_state == 1 else "#ef4444"
    regime_icons = {0: "🔴 BEAR", 1: "⚪ SIDEWAYS", 2: "🟢 BULL", 3: "🚀 BREAKOUT"}
    regime_full = regime_icons.get(regime_state, regime)

    def price_fmt(v):
        return f"{v:,g}".replace(",", ".")

    rows_html = ""
    for _, r in buy.iterrows():
        conf_pct = f"{r['confidence']:.1%}"
        close    = price_fmt(r["close"])
        ret5     = r["ret_5d"]
        ret_col  = "#10b981" if ret5 > 0 else "#ef4444"
        ret_str  = f"{ret5:+.1f}%"
        news_tag = r.get("news_tag", "") or ""
        rows_html += f"""
        <tr>
          <td style="padding:6px 12px;font-weight:700;color:#facc15">{r['ticker']}</td>
          <td style="padding:6px 12px;text-align:right">{close}đ</td>
          <td style="padding:6px 12px;text-align:right">{conf_pct}</td>
          <td style="padding:6px 12px;text-align:right;color:{ret_col}">{ret_str}</td>
          <td style="padding:6px 12px;color:#94a3b8;font-size:12px">{news_tag}</td>
        </tr>"""

    # Portfolio allocation section
    portfolio_rows = ""
    try:
        from portfolio.sizing import PortfolioSizer
        sizer = PortfolioSizer(total_capital=50_000_000)
        sized = sizer.select_trades(df)
        summary = sizer.summary(sized)
        if summary["positions"]:
            for p in summary["positions"]:
                portfolio_rows += f"""
                <tr>
                  <td style="padding:4px 8px;color:#facc15;font-weight:700">{p['ticker']}</td>
                  <td style="padding:4px 8px;color:#10b981">{p['confidence']:.0%}</td>
                  <td style="padding:4px 8px;color:#94a3b8">{p['sector']}</td>
                  <td style="padding:4px 8px;color:#f59e0b;font-weight:600">{p['position_size_vnd']/1_000_000:.1f}M</td>
                </tr>"""
    except Exception:
        pass

    portfolio_section = ""
    if portfolio_rows:
        portfolio_section = f"""
    <h3 style="color:#a78bfa;margin-top:24px">💼 Đề xuất phân bổ vốn</h3>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead>
        <tr style="background:#1e293b;color:#94a3b8;font-size:11px">
          <th style="padding:4px 8px;text-align:left">Mã</th>
          <th style="padding:4px 8px;text-align:left">Conf</th>
          <th style="padding:4px 8px;text-align:left">Sector</th>
          <th style="padding:4px 8px;text-align:left">Vốn</th>
        </tr>
      </thead>
      <tbody style="background:#0f172a">{portfolio_rows}</tbody>
    </table>
    <p style="font-size:11px;color:#475569;margin-top:4px">* Tổng 50M VND, tối đa 5 vị thế</p>"""

    hold_tickers = ", ".join(hold["ticker"].tolist()) if not hold.empty else "—"

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"/></head>
<body style="background:#0f172a;color:#e2e8f0;font-family:Arial,sans-serif;padding:24px">
  <div style="max-width:640px;margin:0 auto">
    <h2 style="color:#facc15;margin-bottom:4px">VN30 Signal — {date_str}</h2>
    <div style="display:flex;gap:12px;align-items:center;margin-bottom:4px">
      <span style="color:{regime_color};font-size:14px;font-weight:700">Regime: {regime_full}</span>
    </div>

    <h3 style="color:#a78bfa;margin-top:24px">⭐ BUY Signals ({len(buy)} mã)</h3>
    {"<p style='color:#64748b'>Không có tín hiệu BUY hôm nay.</p>" if buy.empty else f'''
    <table style="width:100%;border-collapse:collapse;font-size:14px">
      <thead>
        <tr style="background:#1e293b;color:#94a3b8;font-size:12px">
          <th style="padding:6px 12px;text-align:left">Ticker</th>
          <th style="padding:6px 12px;text-align:right">Giá</th>
          <th style="padding:6px 12px;text-align:right">Conf</th>
          <th style="padding:6px 12px;text-align:right">Ret 5d</th>
          <th style="padding:6px 12px;text-align:left">News</th>
        </tr>
      </thead>
      <tbody style="background:#0f172a">
        {rows_html}
      </tbody>
    </table>'''}

    {portfolio_section}

    <p style="margin-top:20px;color:#64748b;font-size:13px">
      ⚪ HOLD: {hold_tickers}
    </p>
    <hr style="border-color:#1e293b;margin-top:24px"/>
    <p style="color:#475569;font-size:11px">VN30 Signal — tự động gửi lúc {date_str}</p>
  </div>
</body>
</html>"""


def send_signal_email(df, subject: str | None = None) -> bool:
    """Gửi email tóm tắt tín hiệu. Trả về True nếu thành công."""
    if not SENDER or not PASSWORD or not any(RECEIVERS):
        print("  [notify] Bỏ qua — chưa cấu hình NOTIFY_EMAIL_* trong .env")
        return False

    import pandas as pd
    date_str = str(df["date"].iloc[0]) if not df.empty else str(date.today())
    regime_state = int(df["regime_state"].iloc[0]) if "regime_state" in df.columns and not df.empty else 2
    regime_names = {0: "BEAR 🔴", 1: "SIDEWAYS ⚪", 2: "BULL 🟢", 3: "BREAKOUT 🚀"}
    regime = regime_names.get(regime_state, "BULL 🟢")

    subject = subject or f"[VN30 Signal] {date_str} — {regime}"
    html_body = _build_html(df, date_str, regime, regime_state)

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
