"""
news.py — Fetch Vietstock + quốc tế RSS, phân tích sentiment + địa chính trị
Output: daily sentiment overlay + geo risk overlay cho từng mã VN30
"""

import re
import time
import feedparser
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.fetch import VN30

NEWS_DIR = Path(__file__).parent.parent / "data" / "news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)

# === RSS Feeds trong nước ===
RSS_FEEDS = {
    "co_phieu":         "https://vietstock.vn/830/chung-khoan/co-phieu.rss",
    "thi_truong":       "https://vietstock.vn/1636/nhan-dinh-phan-tich/nhan-dinh-thi-truong.rss",
    "kinh_doanh":       "https://vietstock.vn/737/doanh-nghiep/hoat-dong-kinh-doanh.rss",
    "co_tuc":           "https://vietstock.vn/738/doanh-nghiep/co-tuc.rss",
    "giao_dich_noi_bo": "https://vietstock.vn/739/chung-khoan/giao-dich-noi-bo.rss",
    "phan_tich_kt":     "https://vietstock.vn/585/nhan-dinh-phan-tich/phan-tich-ky-thuat.rss",
    "tang_von_ma":      "https://vietstock.vn/764/doanh-nghiep/tang-von-m-a.rss",
    "cafef_ck":         "https://cafef.vn/chung-khoan.rss",
    "cafef_doanh_nghiep": "https://cafef.vn/doanh-nghiep.rss",
    "vnexpress_kd":     "https://vnexpress.net/rss/kinh-doanh.rss",
}

# === RSS Feeds quốc tế (địa chính trị + kinh tế thế giới) ===
INTL_RSS_FEEDS = {
    "reuters_world":    "https://feeds.reuters.com/reuters/worldNews",
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "cafef_the_gioi":   "https://cafef.vn/kinh-te-the-gioi.rss",
    "vnexpress_tg":     "https://vnexpress.net/rss/the-gioi.rss",
    "vneconomy_tg":     "https://vneconomy.vn/the-gioi.rss",
}

# === Sentiment keywords (Vietnamese financial) ===

POSITIVE_WORDS = [
    # Tăng trưởng
    "tăng trưởng", "tăng mạnh", "tăng vọt", "bứt phá", "phục hồi", "tăng tốc",
    "kỷ lục", "cao nhất", "vượt kế hoạch", "vượt kỳ vọng", "vượt mục tiêu",
    # Lợi nhuận / doanh thu
    "lợi nhuận tăng", "doanh thu tăng", "lãi ròng tăng", "lãi lớn", "lãi kỷ lục",
    "kết quả tốt", "hiệu quả cao", "sinh lời", "hoàn thành kế hoạch",
    # Cổ tức / chia thưởng
    "cổ tức", "chia thưởng", "phát hành cổ phiếu thưởng", "chia cổ tức",
    "tỷ lệ cổ tức cao", "thưởng cổ phiếu",
    # Mua ròng / dòng tiền
    "mua ròng", "khối ngoại mua", "tự doanh mua", "dòng tiền vào",
    "tích lũy", "gom mạnh", "đẩy mạnh mua",
    # Hợp đồng / đơn hàng
    "ký kết hợp đồng", "trúng thầu", "đơn hàng mới", "hợp tác chiến lược",
    "mở rộng thị trường", "thị phần tăng",
    # Nâng hạng / tích cực
    "nâng hạng", "khuyến nghị mua", "outperform", "tích cực", "triển vọng tốt",
    "tăng giá mục tiêu", "rating tích cực",
    # M&A tích cực
    "mua lại", "thâu tóm", "sáp nhập thành công", "mở rộng quy mô",
]

NEGATIVE_WORDS = [
    # Giảm / thua lỗ
    "lỗ", "thua lỗ", "lỗ ròng", "lợi nhuận giảm", "doanh thu giảm",
    "giảm mạnh", "sụt giảm", "lao dốc", "giảm sâu", "dưới kế hoạch",
    # Vi phạm / cảnh báo
    "vi phạm", "cảnh báo", "kiểm soát đặc biệt", "đình chỉ", "thu hồi",
    "bị phạt", "xử phạt", "khởi tố", "điều tra", "thanh tra",
    "chậm công bố", "chậm nộp", "không đúng hạn",
    # Nợ / thanh khoản
    "nợ xấu", "mất khả năng thanh toán", "vỡ nợ", "nợ đến hạn",
    "áp lực thanh khoản", "thiếu vốn", "cắt giảm", "thu hẹp",
    # Bán ròng / thoái vốn
    "bán ròng", "khối ngoại bán", "thoái vốn", "rút vốn", "xả hàng",
    "bán tháo", "chốt lời mạnh",
    # Rủi ro
    "rủi ro", "bất ổn", "khó khăn", "thách thức lớn", "suy giảm",
    "lo ngại", "áp lực", "giảm giá mục tiêu", "underperform",
    # Hủy niêm yết / delisting
    "hủy niêm yết", "tạm dừng giao dịch", "đình chỉ giao dịch",
    "diện cảnh báo", "diện kiểm soát",
]

# === English sentiment keywords (cho Reuters / quốc tế) ===
INTL_POSITIVE_WORDS = [
    "growth", "profit", "record high", "beat expectations", "upgrade",
    "buy rating", "outperform", "strong demand", "expansion", "rally",
    "surge", "gain", "recovery", "contract win", "deal", "partnership",
    "rate cut", "stimulus", "easing", "optimism", "rebound",
]
INTL_NEGATIVE_WORDS = [
    "loss", "decline", "recession", "default", "bankruptcy", "downgrade",
    "sell rating", "underperform", "inflation", "rate hike", "slowdown",
    "crisis", "collapse", "crash", "plunge", "selloff", "risk-off",
]

# === Địa chính trị — từ khóa → loại rủi ro ===
GEO_RISK_KEYWORDS: dict[str, str] = {
    # Xung đột quân sự
    "war": "conflict", "invasion": "conflict", "military": "conflict",
    "missile": "conflict", "airstrike": "conflict", "nuclear": "conflict",
    "attack": "conflict", "offensive": "conflict", "cease-fire": "conflict",
    "chiến tranh": "conflict", "xung đột": "conflict", "tấn công": "conflict",
    "quân sự": "conflict", "tên lửa": "conflict",
    # Trừng phạt / cấm vận
    "sanctions": "sanctions", "embargo": "sanctions", "ban export": "sanctions",
    "blacklist": "sanctions", "asset freeze": "sanctions",
    "trừng phạt": "sanctions", "cấm vận": "sanctions", "phong tỏa tài sản": "sanctions",
    # Thương mại / thuế quan
    "tariff": "trade", "trade war": "trade", "import duty": "trade",
    "export ban": "trade", "trade restriction": "trade", "protectionism": "trade",
    "thuế quan": "trade", "chiến tranh thương mại": "trade",
    "hạn chế xuất khẩu": "trade", "rào cản thương mại": "trade",
    # Năng lượng
    "oil price": "energy", "energy crisis": "energy", "gas shortage": "energy",
    "opec": "energy", "oil embargo": "energy", "lng": "energy",
    "khủng hoảng năng lượng": "energy", "giá dầu tăng": "energy",
    # Chuỗi cung ứng
    "supply chain": "supply", "chip shortage": "supply", "shortage": "supply",
    "disruption": "supply", "logistics crisis": "supply",
    "đứt gãy chuỗi cung ứng": "supply", "thiếu hụt": "supply",
    # Tổng quát
    "geopolitical": "general", "tension": "general", "uncertainty": "general",
    "instability": "general", "địa chính trị": "general", "bất ổn": "general",
    "khủng hoảng": "general",
}

# Các ngành VN30 bị ảnh hưởng theo loại rủi ro địa chính trị
# conflict → năng lượng + thép bị tác động giá nguyên liệu
# sanctions → ngân hàng + xuất khẩu bị chặn dòng tiền
# trade → xuất khẩu + thép + tiêu dùng
# energy → dầu khí + điện + tiêu dùng
# supply → xây dựng + thép + công nghệ
GEO_SECTOR_EXPOSURE: dict[str, list[str]] = {
    "conflict":  ["GAS", "PVS", "PLX", "HPG", "HSG", "NKG", "POW"],
    "sanctions": ["VCB", "BID", "CTG", "MBB", "TCB", "VPB", "HDB", "STB"],
    "trade":     ["HPG", "HSG", "NKG", "SAB", "MSN", "VIC", "VHM", "FPT"],
    "energy":    ["GAS", "PLX", "PVS", "POW", "REE", "HPG"],
    "supply":    ["HPG", "HSG", "VIC", "VHM", "REE", "FPT", "MWG"],
    "general":   [],  # ảnh hưởng toàn thị trường
}

# Từ có trọng số cao hơn (x2)
HIGH_WEIGHT_POSITIVE = {
    "kỷ lục", "bứt phá", "vượt kế hoạch", "lợi nhuận tăng", "cổ tức",
    "trúng thầu", "khuyến nghị mua",
}
HIGH_WEIGHT_NEGATIVE = {
    "vi phạm", "cảnh báo", "khởi tố", "điều tra", "thua lỗ",
    "diện cảnh báo", "hủy niêm yết", "bị phạt",
}


def _parse_feed(name: str, url: str, is_intl: bool = False) -> list[dict]:
    """Parse một RSS feed, trả về list articles."""
    results = []
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
        for entry in feed.entries:
            pub = entry.get("published_parsed") or entry.get("updated_parsed")
            pub_dt = datetime(*pub[:6], tzinfo=timezone.utc) if pub else datetime.now(timezone.utc)
            results.append({
                "feed": name,
                "title": entry.get("title", ""),
                "summary": re.sub(r"<[^>]+>", " ", entry.get("summary", "")),
                "published": pub_dt,
                "link": entry.get("link", ""),
                "is_intl": is_intl,
            })
    except Exception as e:
        print(f"  ⚠️  Feed {name}: {e}")
    return results


def fetch_all_feeds(delay: float = 0.5) -> list[dict]:
    """Fetch tất cả RSS feeds (trong nước + quốc tế), trả về list of articles."""
    articles = []
    all_feeds = {**{k: (v, False) for k, v in RSS_FEEDS.items()},
                 **{k: (v, True)  for k, v in INTL_RSS_FEEDS.items()}}

    for name, (url, is_intl) in all_feeds.items():
        batch = _parse_feed(name, url, is_intl)
        articles.extend(batch)
        time.sleep(delay)

    domestic = sum(1 for a in articles if not a["is_intl"])
    intl = sum(1 for a in articles if a["is_intl"])
    print(f"  Fetched {len(articles)} articles ({domestic} trong nước, {intl} quốc tế)")
    return articles


def score_intl_sentiment(text: str) -> float:
    """Score sentiment cho bài báo tiếng Anh (Reuters, Bloomberg...)."""
    text_lower = text.lower()
    pos = sum(1.0 for w in INTL_POSITIVE_WORDS if w in text_lower)
    neg = sum(1.0 for w in INTL_NEGATIVE_WORDS if w in text_lower)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


def detect_geo_risk(articles: list[dict], lookback_hours: int = 48) -> dict:
    """
    Phân tích tin quốc tế để phát hiện rủi ro địa chính trị.

    Trả về dict:
      {
        "risk_types": {"conflict": 3, "trade": 1, ...},  # số lần xuất hiện
        "risk_level": 0–3,                                # 0=none,1=low,2=med,3=high
        "exposed_tickers": ["HPG","GAS",...],             # mã bị ảnh hưởng
        "top_headlines": ["...", "..."],                   # tiêu đề đặc trưng nhất
        "summary": "conflict (3), trade (1)",
      }
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    risk_counts: dict[str, int] = {}
    headlines: list[str] = []

    for art in articles:
        if art["published"] < cutoff:
            continue
        text = (art["title"] + " " + art["summary"]).lower()
        hit = False
        for keyword, risk_type in GEO_RISK_KEYWORDS.items():
            if keyword in text:
                risk_counts[risk_type] = risk_counts.get(risk_type, 0) + 1
                if not hit:
                    headlines.append(art["title"][:120])
                    hit = True

    total_hits = sum(risk_counts.values())
    if total_hits == 0:
        risk_level = 0
    elif total_hits <= 3:
        risk_level = 1
    elif total_hits <= 8:
        risk_level = 2
    else:
        risk_level = 3

    exposed: set[str] = set()
    for risk_type, tickers in GEO_SECTOR_EXPOSURE.items():
        if risk_type in risk_counts:
            exposed.update(tickers)
    if "general" in risk_counts:
        exposed.update(VN30)

    summary_parts = [f"{k} ({v})" for k, v in sorted(risk_counts.items(), key=lambda x: -x[1])]

    return {
        "risk_types": risk_counts,
        "risk_level": risk_level,
        "exposed_tickers": sorted(exposed),
        "top_headlines": headlines[:5],
        "summary": ", ".join(summary_parts) if summary_parts else "none",
    }


def extract_tickers(text: str, tickers: list[str] = VN30) -> list[str]:
    """Tìm các mã VN30 được nhắc đến trong text."""
    found = []
    text_upper = text.upper()
    for ticker in tickers:
        # Match ticker as standalone word (tránh nhầm ACB trong "HACB")
        if re.search(rf'\b{ticker}\b', text_upper):
            found.append(ticker)
    return found


def score_sentiment(text: str, is_intl: bool = False) -> float:
    """
    Tính sentiment score từ -1.0 đến +1.0.
    Tự động chọn bộ từ khóa tiếng Việt hay tiếng Anh.
    """
    text_lower = text.lower()
    pos_score = 0.0
    neg_score = 0.0

    if is_intl:
        pos_score = sum(1.0 for w in INTL_POSITIVE_WORDS if w in text_lower)
        neg_score = sum(1.0 for w in INTL_NEGATIVE_WORDS if w in text_lower)
    else:
        for word in POSITIVE_WORDS:
            if word in text_lower:
                pos_score += 2.0 if word in HIGH_WEIGHT_POSITIVE else 1.0
        for word in NEGATIVE_WORDS:
            if word in text_lower:
                neg_score += 2.0 if word in HIGH_WEIGHT_NEGATIVE else 1.0

    total = pos_score + neg_score
    return (pos_score - neg_score) / total if total > 0 else 0.0


def build_ticker_sentiment(
    articles: list[dict],
    tickers: list[str] = VN30,
    lookback_days: int = 3,
) -> pd.DataFrame:
    """
    Với mỗi ticker, tính:
    - sentiment_1d / sentiment_3d: score trung bình bài nhắc đến mã đó
    - news_count_1d / news_count_3d: số bài trong 24h / 3 ngày
    - negative_flag: có tin xấu nặng (vi phạm, khởi tố...)
    - market_sentiment_1d: sentiment chung thị trường
    - geo_risk_score: mức độ rủi ro địa chính trị (0.0–1.0)
    - geo_risk_tag: mô tả loại rủi ro địa chính trị nếu có
    """
    import numpy as np

    now = datetime.now(timezone.utc)
    cutoff_1d = now - timedelta(days=1)
    cutoff_3d = now - timedelta(days=3)

    STRONG_NEGATIVE = {"vi phạm", "cảnh báo", "khởi tố", "điều tra",
                       "diện cảnh báo", "hủy niêm yết", "bị phạt", "đình chỉ giao dịch"}

    # Phát hiện rủi ro địa chính trị từ tin quốc tế
    geo = detect_geo_risk(articles, lookback_hours=48)
    geo_level = geo["risk_level"]          # 0-3
    geo_exposed = set(geo["exposed_tickers"])
    geo_summary = geo["summary"]

    rows = []
    for ticker in tickers:
        ticker_arts_1d, ticker_arts_3d = [], []
        negative_flag = 0

        for art in articles:
            text = art["title"] + " " + art["summary"]
            if not extract_tickers(text, [ticker]):
                continue
            is_intl = art.get("is_intl", False)
            s = score_sentiment(text, is_intl=is_intl)
            if any(w in text.lower() for w in STRONG_NEGATIVE):
                negative_flag = 1
            if art["published"] >= cutoff_1d:
                ticker_arts_1d.append(s)
            if art["published"] >= cutoff_3d:
                ticker_arts_3d.append(s)

        # Geo risk score cho ticker này
        if ticker in geo_exposed and geo_level > 0:
            geo_risk_score = geo_level / 3.0   # normalize to 0–1
            geo_risk_tag = f"🌍 geo:{geo_summary[:40]}"
        else:
            geo_risk_score = 0.0
            geo_risk_tag = ""

        rows.append({
            "ticker": ticker,
            "news_sentiment_1d": float(np.mean(ticker_arts_1d)) if ticker_arts_1d else 0.0,
            "news_sentiment_3d": float(np.mean(ticker_arts_3d)) if ticker_arts_3d else 0.0,
            "news_count_1d": len(ticker_arts_1d),
            "news_count_3d": len(ticker_arts_3d),
            "negative_flag": negative_flag,
            "geo_risk_score": geo_risk_score,
            "geo_risk_tag": geo_risk_tag,
        })

    df = pd.DataFrame(rows).set_index("ticker")

    # Market-wide sentiment (feed thi_truong trong nước)
    market_arts_1d = [
        score_sentiment(a["title"] + " " + a["summary"])
        for a in articles
        if a["feed"] == "thi_truong" and a["published"] >= cutoff_1d
    ]
    df["market_sentiment_1d"] = float(np.mean(market_arts_1d)) if market_arts_1d else 0.0
    df["geo_risk_level"] = geo_level   # 0-3, thị trường chung

    if geo_level > 0:
        print(f"  🌍 Geo risk level={geo_level}/3 — {geo_summary}")
        for hl in geo["top_headlines"][:3]:
            print(f"     · {hl}")

    return df


def apply_news_overlay(signals: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng news overlay lên model signals:
    - BUY + negative_flag=1 → HOLD + tag "⚠️ tin xấu"
    - BUY + sentiment_1d < -0.3 → HOLD + tag "📰 sentiment âm"
    - BUY + sentiment_1d > 0.2 → tag "📰 news confirmed"
    - BUY + geo_risk_score >= 0.67 (level 2+) → HOLD + tag "🌍 geo risk cao"
    - BUY + geo_risk_score > 0 (level 1) → giữ BUY nhưng tag cảnh báo
    - HOLD + sentiment_1d > 0.4 → tag "👀 watch: tin tốt"
    """
    df = signals.copy()
    df = df.join(sentiment, on="ticker", how="left")

    for col, fill in [
        ("news_sentiment_1d", 0.0), ("news_sentiment_3d", 0.0),
        ("market_sentiment_1d", 0.0), ("geo_risk_score", 0.0),
        ("geo_risk_tag", ""), ("news_count_1d", 0), ("negative_flag", 0),
    ]:
        if col not in df.columns:
            df[col] = fill
        else:
            df[col] = df[col].fillna(fill)

    df["news_count_1d"] = df["news_count_1d"].astype(int)
    df["negative_flag"] = df["negative_flag"].astype(int)
    df["news_tag"] = ""

    for idx, row in df.iterrows():
        signal = row["signal"]
        sent = row["news_sentiment_1d"]
        neg = int(row["negative_flag"])
        count = int(row["news_count_1d"])
        geo = float(row["geo_risk_score"])
        geo_tag = str(row.get("geo_risk_tag", ""))

        if signal == "BUY":
            if neg == 1:
                df.at[idx, "signal"] = "HOLD"
                df.at[idx, "news_tag"] = "⚠️ tin xấu"
            elif sent < -0.3:
                df.at[idx, "signal"] = "HOLD"
                df.at[idx, "news_tag"] = "📰 sentiment âm"
            elif geo >= 0.67:
                # Geo risk level 2-3: đủ cao để hạ xuống HOLD
                df.at[idx, "signal"] = "HOLD"
                df.at[idx, "news_tag"] = geo_tag or "🌍 geo risk cao"
            elif geo > 0:
                # Geo risk level 1: cảnh báo nhưng giữ BUY
                df.at[idx, "news_tag"] = geo_tag or "🌍 geo risk"
            elif sent > 0.2:
                df.at[idx, "news_tag"] = "📰 news confirmed"
        elif signal == "HOLD":
            if sent > 0.4 and count >= 1:
                df.at[idx, "news_tag"] = "👀 watch: tin tốt"
            elif geo > 0 and not df.at[idx, "news_tag"]:
                df.at[idx, "news_tag"] = geo_tag or "🌍 geo risk"

    return df


def get_today_sentiment(verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """
    Main function: fetch RSS → (sentiment DataFrame, articles list).
    Trả về tuple để caller có thể dùng articles cho save_articles().
    """
    if verbose:
        print("  Fetching RSS feeds (trong nước + quốc tế)...")
    articles = fetch_all_feeds(delay=0.5)
    sentiment = build_ticker_sentiment(articles)
    if verbose:
        covered = (sentiment["news_count_1d"] > 0).sum()
        geo_level = int(sentiment["geo_risk_level"].iloc[0]) if "geo_risk_level" in sentiment.columns else 0
        print(f"  {covered}/{len(sentiment)} tickers có tin tức trong 24h")
        print(f"  Market sentiment: {sentiment['market_sentiment_1d'].iloc[0]:+.2f}")
        if geo_level > 0:
            geo_icons = {1: "🟡 Thấp", 2: "🟠 Trung bình", 3: "🔴 Cao"}
            print(f"  Geo risk: {geo_icons.get(geo_level, str(geo_level))}")
    return sentiment, articles


def save_sentiment(df: pd.DataFrame, date: str | None = None) -> Path:
    if date is None:
        date = pd.Timestamp.today().strftime("%Y-%m-%d")
    path = NEWS_DIR / f"{date}.csv"
    df.to_csv(path)
    return path


def save_articles(articles: list[dict], date: str | None = None) -> Path:
    """Lưu mapping ticker → bài báo + geo risk headlines để dùng trong dashboard."""
    import json
    if date is None:
        date = pd.Timestamp.today().strftime("%Y-%m-%d")
    path = NEWS_DIR / f"{date}_articles.json"

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=3)

    ticker_articles: dict[str, list] = {t: [] for t in VN30}
    for art in articles:
        if art["published"] < cutoff:
            continue
        text = art["title"] + " " + art["summary"]
        tickers_found = extract_tickers(text)
        for t in tickers_found:
            ticker_articles[t].append({
                "title": art["title"][:120],
                "link": art.get("link", ""),
                "published": art["published"].strftime("%d/%m %H:%M"),
                "sentiment": score_sentiment(text, is_intl=art.get("is_intl", False)),
                "source": "intl" if art.get("is_intl") else "domestic",
            })

    for t in ticker_articles:
        ticker_articles[t] = ticker_articles[t][:5]

    # Lưu geo risk headlines riêng để dashboard hiển thị khi click tag 🌍
    geo = detect_geo_risk(articles, lookback_hours=48)
    geo_headlines = []
    for art in articles:
        if art["published"] < now - timedelta(hours=48):
            continue
        text = (art["title"] + " " + art["summary"]).lower()
        if any(kw in text for kw in GEO_RISK_KEYWORDS):
            geo_headlines.append({
                "title": art["title"][:140],
                "link": art.get("link", ""),
                "published": art["published"].strftime("%d/%m %H:%M"),
                "source": art["feed"],
            })
    ticker_articles["_geo_risk"] = {
        "level": geo["risk_level"],
        "summary": geo["summary"],
        "headlines": geo_headlines[:8],
    }

    path.write_text(json.dumps(ticker_articles, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    sentiment, articles = get_today_sentiment(verbose=True)
    print("\n=== Sentiment by Ticker ===")
    active = sentiment[sentiment["news_count_1d"] > 0].sort_values(
        "news_sentiment_1d", ascending=False
    )
    if active.empty:
        print("  Không có mã VN30 nào được nhắc đến trong 24h gần nhất.")
    else:
        print(f"  {'Ticker':<8} {'Sent1d':>8} {'Sent3d':>8} {'Count1d':>8} {'NegFlag':>8} {'Tag'}")
        print(f"  {'-'*60}")
        for ticker, row in active.iterrows():
            flag = "🚨" if row["negative_flag"] else ""
            print(f"  {ticker:<8} {row['news_sentiment_1d']:>+8.2f} {row['news_sentiment_3d']:>+8.2f} "
                  f"{int(row['news_count_1d']):>8} {flag:>8}")

    path = save_sentiment(sentiment)
    print(f"\n✅ Saved → {path}")
