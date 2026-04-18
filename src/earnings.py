"""
earnings.py — Earnings season calendar features
VN reporting schedule:
  Q4 results: Jan 15 – Feb 28  (high impact, full year)
  Q1 results: Apr 15 – May 15
  Q2 results: Jul 15 – Aug 15
  Q3 results: Oct 15 – Nov 15
"""

import numpy as np
import pandas as pd

# Earnings windows (month_start, day_start, month_end, day_end)
EARNINGS_WINDOWS = [
    (1, 15, 2, 28),   # Q4 full year — highest impact
    (4, 15, 5, 15),   # Q1
    (7, 15, 8, 15),   # Q2
    (10, 15, 11, 15), # Q3
]

# Stocks with historically strong earnings reactions (upgrade weight)
EARNINGS_SENSITIVE = {
    "FPT", "MWG", "TCB", "VCB", "HPG", "VHM",
    "ACB", "MBB", "VPB", "CTG", "BID",
}


def days_to_next_earnings(date: pd.Timestamp) -> int:
    """Số ngày đến mùa KQKD tiếp theo."""
    year = date.year
    candidates = []
    for m_start, d_start, m_end, d_end in EARNINGS_WINDOWS:
        start = pd.Timestamp(year, m_start, d_start)
        end = pd.Timestamp(year, m_end, d_end)
        if date <= end:
            candidates.append((start - date).days)
        # Next year Q4
        next_start = pd.Timestamp(year + 1, m_start, d_start)
        candidates.append((next_start - date).days)

    positives = [d for d in candidates if d >= 0]
    return min(positives) if positives else 90


def in_earnings_window(date: pd.Timestamp) -> int:
    """1 nếu đang trong mùa công bố KQKD."""
    m, d = date.month, date.day
    for m_start, d_start, m_end, d_end in EARNINGS_WINDOWS:
        if (m == m_start and d >= d_start) or \
           (m == m_end and d <= d_end) or \
           (m_start < m < m_end):
            return 1
    return 0


def earnings_season_type(date: pd.Timestamp) -> int:
    """
    0 = không trong mùa
    1 = Q1/Q2/Q3 season (impact bình thường)
    2 = Q4 season (full year, impact cao nhất)
    """
    m, d = date.month, date.day
    # Q4 full year
    if (m == 1 and d >= 15) or (m == 2 and d <= 28):
        return 2
    # Other quarters
    for m_start, d_start, m_end, d_end in EARNINGS_WINDOWS[1:]:
        if (m == m_start and d >= d_start) or \
           (m == m_end and d <= d_end) or \
           (m_start < m < m_end):
            return 1
    return 0


def add_earnings_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Thêm earnings calendar features vào stock dataframe."""
    df = df.copy()

    dates = pd.to_datetime(df.index)

    df["in_earnings_season"] = dates.map(in_earnings_window)
    df["earnings_season_type"] = dates.map(earnings_season_type)
    df["days_to_earnings"] = dates.map(days_to_next_earnings)

    # Normalize days_to_earnings: 0 = ngay hôm nay bắt đầu mùa, 1 = 90+ ngày
    df["days_to_earnings_norm"] = (df["days_to_earnings"].clip(0, 90) / 90)

    # Bonus: earnings-sensitive stocks
    df["earnings_sensitive"] = 1 if ticker in EARNINGS_SENSITIVE else 0

    # Interaction: sensitive stock + in season = strong signal
    df["earnings_boost"] = (
        df["in_earnings_season"] * df["earnings_sensitive"]
    ).astype(int)

    return df


EARNINGS_FEATURE_COLS = [
    "in_earnings_season",
    "earnings_season_type",
    "days_to_earnings_norm",
    "earnings_sensitive",
    "earnings_boost",
]


if __name__ == "__main__":
    # Sanity check
    test_dates = [
        "2024-01-20",  # Q4 season
        "2024-04-20",  # Q1 season
        "2024-07-01",  # Not in season
        "2024-10-20",  # Q3 season
    ]
    for d in test_dates:
        ts = pd.Timestamp(d)
        print(f"{d}: in_season={in_earnings_window(ts)} "
              f"type={earnings_season_type(ts)} "
              f"days_to_next={days_to_next_earnings(ts)}")
