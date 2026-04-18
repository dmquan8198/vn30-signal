"""
live_overlay.py — Live data overlays (không dùng cho training, chỉ cho live signal)
1. Foreign Flow: price_board API → foreign net volume hôm nay
2. Insider Trading: RSS → insider buy/sell trong 30 ngày gần nhất
"""

import re
import time
import feedparser
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from vnstock import Vnstock

from src.fetch import VN30
from src.earnings import in_earnings_window, EARNINGS_SENSITIVE

# Earnings event filter: avoid entering positions 2 days before/after earnings period
EARNINGS_BUFFER_DAYS = 2

# ─── 1. FOREIGN FLOW ──────────────────────────────────────────────────────────

def fetch_foreign_flow(tickers: list[str] = VN30) -> pd.DataFrame:
    """
    Lấy foreign buy/sell volume hôm nay từ price_board.
    Returns DataFrame indexed by ticker.
    """
    try:
        stock = Vnstock().stock(symbol=tickers[0], source='KBS')
        board = stock.trading.price_board(tickers)

        cols = ["foreign_buy_volume", "foreign_sell_volume", "volume_accumulated"]
        if "foreign_room" in board.columns:
            cols.append("foreign_room")
        board = board.set_index("symbol")[cols].copy()

        board["foreign_net"] = board["foreign_buy_volume"] - board["foreign_sell_volume"]
        board["foreign_net_pct"] = (
            board["foreign_net"] / board["volume_accumulated"].replace(0, np.nan)
        ).fillna(0)

        # Classify net flow
        def classify(pct):
            if pct > 0.10:  return "strong_buy"
            elif pct > 0.03: return "buy"
            elif pct < -0.10: return "strong_sell"
            elif pct < -0.03: return "sell"
            else:             return "neutral"

        board["foreign_signal"] = board["foreign_net_pct"].apply(classify)

        # Room status: 0 = kín room (khối ngoại không thể mua thêm)
        if "foreign_room" in board.columns:
            board["room_full"] = (board["foreign_room"] == 0).astype(int)
        else:
            board["room_full"] = 0

        return board

    except Exception as e:
        print(f"  ⚠️  Foreign flow fetch error: {e}")
        return pd.DataFrame(
            index=tickers,
            columns=["foreign_net", "foreign_net_pct", "foreign_signal"]
        ).fillna(0)


# ─── 2. INSIDER TRADING ───────────────────────────────────────────────────────

INSIDER_RSS = "https://vietstock.vn/739/chung-khoan/giao-dich-noi-bo.rss"

BUY_KEYWORDS  = ["mua vào", "mua thêm", "mua thành công", "hoàn tất mua",
                  "đăng ký mua", "gom", "tăng sở hữu", "mua lại"]
SELL_KEYWORDS = ["bán ra", "bán thành công", "hoàn tất bán", "đăng ký bán",
                 "thoái vốn", "giảm sở hữu", "bán hết", "bán toàn bộ"]

# Insider types có weight cao hơn
HIGH_VALUE_INSIDERS = ["chủ tịch", "tổng giám đốc", "ceo", "phó tổng", "cfo",
                       "thành viên hđqt", "hội đồng quản trị", "ban giám đốc"]


def parse_insider_transactions(lookback_days: int = 30) -> pd.DataFrame:
    """
    Parse RSS giao dịch nội bộ, extract:
    - ticker, direction (buy/sell), weight (insider seniority), pubDate
    Returns: DataFrame với insider score mỗi ticker trong lookback_days
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows = []

    try:
        feed = feedparser.parse(INSIDER_RSS)
        for entry in feed.entries:
            pub = entry.get("published_parsed") or entry.get("updated_parsed")
            if not pub:
                continue
            pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
            if pub_dt < cutoff:
                continue

            title = entry.get("title", "")
            summary = entry.get("summary", "")
            text = (title + " " + summary).lower()

            # Extract tickers (VN30 only)
            tickers_found = [
                t for t in VN30
                if re.search(rf'\b{t}\b', (title + " " + summary).upper())
            ]
            if not tickers_found:
                continue

            # Direction
            is_buy  = any(kw in text for kw in BUY_KEYWORDS)
            is_sell = any(kw in text for kw in SELL_KEYWORDS)
            if not is_buy and not is_sell:
                continue
            direction = 1 if is_buy else -1

            # Weight: senior insider = 2x
            weight = 2.0 if any(kw in text for kw in HIGH_VALUE_INSIDERS) else 1.0

            # Try to extract volume (số cổ phiếu)
            vol_match = re.search(r'(\d[\d,.]+)\s*(triệu|nghìn|cp|cổ phiếu)', text)
            vol_scale = 1
            if vol_match:
                vol_str = vol_match.group(1).replace(",", "").replace(".", "")
                unit = vol_match.group(2)
                try:
                    vol = float(vol_str)
                    if "triệu" in unit: vol *= 1_000_000
                    elif "nghìn" in unit: vol *= 1_000
                    vol_scale = min(vol / 1_000_000, 3.0)  # cap at 3x for 1M+ shares
                except:
                    pass

            for t in tickers_found:
                rows.append({
                    "ticker": t,
                    "direction": direction,
                    "weight": weight * (1 + vol_scale),
                    "date": pub_dt,
                    "title": title[:80],
                })

    except Exception as e:
        print(f"  ⚠️  Insider RSS error: {e}")

    if not rows:
        return pd.DataFrame({"ticker": VN30, "insider_score": 0.0,
                             "insider_buy_flag": 0, "insider_sell_flag": 0}).set_index("ticker")

    df = pd.DataFrame(rows)

    # Aggregate: net score per ticker
    summary = df.groupby("ticker").apply(
        lambda g: pd.Series({
            "insider_score": (g["direction"] * g["weight"]).sum(),
            "insider_buy_flag": int((g["direction"] == 1).any()),
            "insider_sell_flag": int((g["direction"] == -1).any()),
            "insider_events": len(g),
        })
    ).reindex(VN30).fillna(0)

    return summary


# ─── 3. COMBINED OVERLAY ──────────────────────────────────────────────────────

def apply_live_overlay(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng foreign flow + insider trading lên signals.
    """
    print("  Fetching foreign flow (price_board)...")
    foreign = fetch_foreign_flow()

    print("  Parsing insider transactions (RSS 30d)...")
    insider = parse_insider_transactions(lookback_days=30)

    df = signals.copy()
    df = df.join(foreign[["foreign_net_pct", "foreign_signal", "room_full"]], on="ticker", how="left")
    df = df.join(insider[["insider_score", "insider_buy_flag", "insider_sell_flag"]], on="ticker", how="left")

    df["foreign_net_pct"] = df["foreign_net_pct"].fillna(0)
    df["foreign_signal"] = df["foreign_signal"].fillna("neutral")
    df["room_full"] = df["room_full"].fillna(0).astype(int)
    df["insider_score"] = df["insider_score"].fillna(0)
    df["insider_buy_flag"] = df["insider_buy_flag"].fillna(0).astype(int)
    df["insider_sell_flag"] = df["insider_sell_flag"].fillna(0).astype(int)

    if "news_tag" not in df.columns:
        df["news_tag"] = ""

    for idx, row in df.iterrows():
        tags = [str(row["news_tag"])] if row.get("news_tag") else []
        signal = row["signal"]
        f_sig = row["foreign_signal"]
        room_full = row["room_full"]
        ins_buy = row["insider_buy_flag"]
        ins_sell = row["insider_sell_flag"]

        # P2.7: ML SELL as negative filter — nếu model muốn SELL mà signal là BUY → downgrade
        ml_sell = int(row.get("ml_sell_flag", 0))
        if ml_sell and signal == "BUY":
            df.at[idx, "signal"] = "HOLD"
            signal = "HOLD"
            tags.append("⚠️ ML cảnh báo bán (negative filter)")

        # Room status — kín room ảnh hưởng đến cách đọc dòng ngoại
        if room_full:
            tags.append("🔒 kín room ngoại")
            # Khi kín room: foreign sell = ngoại đang rút → hạ tín hiệu
            if f_sig in ("strong_sell", "sell") and signal == "BUY":
                df.at[idx, "signal"] = "HOLD"
                tags.append("⚠️ ngoại rút vốn")
        else:
            # Còn room: dòng ngoại vào/ra có ý nghĩa
            if f_sig == "strong_buy":
                tags.append("🌊 ngoại mua mạnh")
            elif f_sig == "buy" and signal == "BUY":
                tags.append("🌊 ngoại mua")
            elif f_sig == "strong_sell" and signal == "BUY":
                df.at[idx, "signal"] = "HOLD"
                tags.append("🌊 ngoại bán mạnh")
            elif f_sig == "sell" and signal == "BUY":
                tags.append("⚠️ ngoại bán ròng")

        # Chuỗi sàn liên tiếp — cảnh báo bán tháo
        floor_streak = row.get("floor_streak", 0)
        ceil_streak = row.get("ceil_streak", 0)
        if floor_streak >= 2 and signal == "BUY":
            df.at[idx, "signal"] = "HOLD"
            tags.append(f"📉 sàn {int(floor_streak)} ngày liên tiếp")
        elif floor_streak == 1 and signal == "BUY":
            tags.append("⚠️ vừa chạm sàn")
        if ceil_streak >= 2:
            tags.append(f"📈 trần {int(ceil_streak)} ngày liên tiếp")

        # Insider tags
        if ins_buy and signal == "BUY":
            tags.append("👤 nội bộ đang mua")
        elif ins_buy and signal == "HOLD":
            tags.append("👤 nội bộ mua (theo dõi)")
        if ins_sell and signal == "BUY":
            tags.append("👤 nội bộ đang bán")

        # P3.4: Earnings event filter — tránh mua 2 ngày quanh mùa KQKD cho earnings-sensitive stocks
        today = pd.Timestamp.today()
        near_earnings = False
        for offset in range(-EARNINGS_BUFFER_DAYS, EARNINGS_BUFFER_DAYS + 1):
            check_date = today + pd.DateOffset(days=offset)
            if in_earnings_window(check_date):
                near_earnings = True
                break
        if near_earnings and row["ticker"] in EARNINGS_SENSITIVE and signal == "BUY":
            df.at[idx, "signal"] = "HOLD"
            signal = "HOLD"
            tags.append("📅 mùa KQKD — tránh vào lệnh")

        df.at[idx, "news_tag"] = "  ".join(t for t in tags if t)

    return df


if __name__ == "__main__":
    print("=== Foreign Flow ===")
    ff = fetch_foreign_flow(["VCB", "FPT", "HPG", "VIC", "MBB"])
    print(ff[["foreign_net", "foreign_net_pct", "foreign_signal"]].to_string())

    print("\n=== Insider Transactions (30d) ===")
    ins = parse_insider_transactions(30)
    active = ins[ins["insider_score"] != 0]
    if active.empty:
        print("  Không có giao dịch nội bộ VN30 trong 30 ngày qua")
    else:
        print(active.to_string())
