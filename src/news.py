"""
news.py — Fetch Vietstock RSS, extract ticker mentions, score sentiment
Output: daily sentiment overlay cho từng mã VN30
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

# === RSS Feeds ===
RSS_FEEDS = {
    "co_phieu":         "https://vietstock.vn/830/chung-khoan/co-phieu.rss",
    "thi_truong":       "https://vietstock.vn/1636/nhan-dinh-phan-tich/nhan-dinh-thi-truong.rss",
    "kinh_doanh":       "https://vietstock.vn/737/doanh-nghiep/hoat-dong-kinh-doanh.rss",
    "co_tuc":           "https://vietstock.vn/738/doanh-nghiep/co-tuc.rss",
    "giao_dich_noi_bo": "https://vietstock.vn/739/chung-khoan/giao-dich-noi-bo.rss",
    "phan_tich_kt":     "https://vietstock.vn/585/nhan-dinh-phan-tich/phan-tich-ky-thuat.rss",
    "tang_von_ma":      "https://vietstock.vn/764/doanh-nghiep/tang-von-m-a.rss",
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

# Từ có trọng số cao hơn (x2)
HIGH_WEIGHT_POSITIVE = {
    "kỷ lục", "bứt phá", "vượt kế hoạch", "lợi nhuận tăng", "cổ tức",
    "trúng thầu", "khuyến nghị mua",
}
HIGH_WEIGHT_NEGATIVE = {
    "vi phạm", "cảnh báo", "khởi tố", "điều tra", "thua lỗ",
    "diện cảnh báo", "hủy niêm yết", "bị phạt",
}


def fetch_all_feeds(delay: float = 1.0) -> list[dict]:
    """Fetch tất cả RSS feeds, trả về list of articles."""
    articles = []
    for name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub:
                    pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
                else:
                    pub_dt = datetime.now(timezone.utc)

                articles.append({
                    "feed": name,
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "published": pub_dt,
                    "link": entry.get("link", ""),
                })
            time.sleep(delay)
        except Exception as e:
            print(f"  ⚠️  Feed {name}: {e}")

    print(f"  Fetched {len(articles)} articles from {len(RSS_FEEDS)} feeds")
    return articles


def extract_tickers(text: str, tickers: list[str] = VN30) -> list[str]:
    """Tìm các mã VN30 được nhắc đến trong text."""
    found = []
    text_upper = text.upper()
    for ticker in tickers:
        # Match ticker as standalone word (tránh nhầm ACB trong "HACB")
        if re.search(rf'\b{ticker}\b', text_upper):
            found.append(ticker)
    return found


def score_sentiment(text: str) -> float:
    """
    Tính sentiment score từ -1.0 đến +1.0
    Dựa trên keyword matching, có trọng số
    """
    text_lower = text.lower()
    pos_score = 0.0
    neg_score = 0.0

    for word in POSITIVE_WORDS:
        if word in text_lower:
            weight = 2.0 if word in HIGH_WEIGHT_POSITIVE else 1.0
            pos_score += weight

    for word in NEGATIVE_WORDS:
        if word in text_lower:
            weight = 2.0 if word in HIGH_WEIGHT_NEGATIVE else 1.0
            neg_score += weight

    total = pos_score + neg_score
    if total == 0:
        return 0.0

    # Score từ -1 đến +1
    return (pos_score - neg_score) / total


def build_ticker_sentiment(
    articles: list[dict],
    tickers: list[str] = VN30,
    lookback_days: int = 3,
) -> pd.DataFrame:
    """
    Với mỗi ticker, tính:
    - sentiment_1d: score trung bình articles trong 24h gần nhất
    - sentiment_3d: score trung bình articles trong 3 ngày
    - news_count_1d: số bài nhắc đến trong 24h
    - news_count_3d: số bài nhắc đến trong 3 ngày
    - negative_flag: có tin xấu nặng không (vi phạm, cảnh báo, khởi tố...)
    - market_sentiment_1d: sentiment chung thị trường (không phân biệt mã)
    """
    now = datetime.now(timezone.utc)
    cutoff_1d = now - timedelta(days=1)
    cutoff_3d = now - timedelta(days=3)

    STRONG_NEGATIVE = {"vi phạm", "cảnh báo", "khởi tố", "điều tra",
                       "diện cảnh báo", "hủy niêm yết", "bị phạt", "đình chỉ giao dịch"}

    rows = []

    for ticker in tickers:
        ticker_arts_1d = []
        ticker_arts_3d = []
        negative_flag = 0

        for art in articles:
            text = art["title"] + " " + art["summary"]
            mentioned = extract_tickers(text, [ticker])
            if not mentioned:
                continue

            score = score_sentiment(text)
            text_lower = text.lower()

            # Check strong negative
            if any(w in text_lower for w in STRONG_NEGATIVE):
                negative_flag = 1

            if art["published"] >= cutoff_1d:
                ticker_arts_1d.append(score)
            if art["published"] >= cutoff_3d:
                ticker_arts_3d.append(score)

        import numpy as np
        rows.append({
            "ticker": ticker,
            "news_sentiment_1d": float(np.mean(ticker_arts_1d)) if ticker_arts_1d else 0.0,
            "news_sentiment_3d": float(np.mean(ticker_arts_3d)) if ticker_arts_3d else 0.0,
            "news_count_1d": len(ticker_arts_1d),
            "news_count_3d": len(ticker_arts_3d),
            "negative_flag": negative_flag,
        })

    df = pd.DataFrame(rows).set_index("ticker")

    # Market-wide sentiment (từ feed thi_truong)
    market_arts_1d = [
        score_sentiment(a["title"] + " " + a["summary"])
        for a in articles
        if a["feed"] == "thi_truong" and a["published"] >= cutoff_1d
    ]
    import numpy as np
    market_sent = float(np.mean(market_arts_1d)) if market_arts_1d else 0.0
    df["market_sentiment_1d"] = market_sent

    return df


def apply_news_overlay(signals: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng news overlay lên model signals:
    - BUY + negative_flag=1 → downgrade HOLD
    - BUY + sentiment_1d < -0.3 → downgrade HOLD
    - BUY + sentiment_1d > 0.2 → thêm tag ⭐ news_confirmed
    - HOLD + sentiment_1d > 0.4 + news_count >= 2 → flag WATCH
    """
    df = signals.copy()
    df = df.join(sentiment, on="ticker", how="left")

    # Fill NaN (ticker không có tin tức)
    df["news_sentiment_1d"] = df["news_sentiment_1d"].fillna(0.0)
    df["news_sentiment_3d"] = df["news_sentiment_3d"].fillna(0.0)
    df["news_count_1d"] = df["news_count_1d"].fillna(0).astype(int)
    df["negative_flag"] = df["negative_flag"].fillna(0).astype(int)
    df["market_sentiment_1d"] = df["market_sentiment_1d"].fillna(0.0)

    df["news_tag"] = ""

    for idx, row in df.iterrows():
        signal = row["signal"]
        sent = row["news_sentiment_1d"]
        neg = row["negative_flag"]
        count = row["news_count_1d"]

        if signal == "BUY":
            if neg == 1:
                df.at[idx, "signal"] = "HOLD"
                df.at[idx, "news_tag"] = "⚠️ tin xấu"
            elif sent < -0.3:
                df.at[idx, "signal"] = "HOLD"
                df.at[idx, "news_tag"] = "📰 sentiment âm"
            elif sent > 0.2:
                df.at[idx, "news_tag"] = "📰 news confirmed"

        elif signal == "HOLD":
            if sent > 0.4 and count >= 1:
                df.at[idx, "news_tag"] = "👀 watch: tin tốt"

    return df


def get_today_sentiment(verbose: bool = True) -> pd.DataFrame:
    """Main function: fetch RSS → sentiment DataFrame."""
    if verbose:
        print("  Fetching Vietstock RSS feeds...")
    articles = fetch_all_feeds(delay=0.5)
    sentiment = build_ticker_sentiment(articles)
    if verbose:
        covered = (sentiment["news_count_1d"] > 0).sum()
        print(f"  {covered}/{len(sentiment)} tickers có tin tức trong 24h")
        print(f"  Market sentiment: {sentiment['market_sentiment_1d'].iloc[0]:+.2f}")
    return sentiment


def save_sentiment(df: pd.DataFrame, date: str | None = None) -> Path:
    if date is None:
        date = pd.Timestamp.today().strftime("%Y-%m-%d")
    path = NEWS_DIR / f"{date}.csv"
    df.to_csv(path)
    return path


def save_articles(articles: list[dict], date: str | None = None) -> Path:
    """Lưu mapping ticker → danh sách bài báo (title + link) để dùng trong dashboard."""
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
                "sentiment": score_sentiment(text),
            })

    # Keep max 5 latest per ticker
    for t in ticker_articles:
        ticker_articles[t] = ticker_articles[t][:5]

    path.write_text(json.dumps(ticker_articles, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    sentiment = get_today_sentiment(verbose=True)
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
