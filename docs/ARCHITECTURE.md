# VN30 Signal System — Architecture

> Generated từ code thực tế (không phải từ docs). Last updated: 2026-04-18.

---

## Dataflow tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│                     run_daily.sh (cron)                     │
│              09:00 & 13:30, weekdays (Mon-Fri)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              src/signal_generator.py  (__main__)            │
│                                                             │
│  1. get_signal_today(refresh_data)                          │
│  2. fetch_all_feeds() → build_ticker_sentiment()            │
│  3. apply_news_overlay(df, sentiment)                       │
│  4. apply_live_overlay(df)                                  │
│  5. print_signals(df) + save_signals(df)                    │
│  6. tracker.run(signals=df)                                 │
│  7. send_signal_email(df)                                   │
└──┬──────────────────────────────────────────────────────────┘
   │
   │  get_signal_today()
   ├──────────────────────────────────────────────────────────┐
   │                                                          │
   ▼                                                          ▼
┌──────────────────┐                              ┌───────────────────────┐
│  src/fetch.py    │                              │  src/features.py      │
│                  │                              │                       │
│ fetch_ticker()   │──OHLCV per ticker──────────▶│ add_indicators()      │
│ fetch_indices()  │──VNINDEX,VN30──────────────▶│ build_market_context()│
│ load_all()       │                              │ add_market_context()  │
│ load_ticker()    │                              │ add_relative_strength │
│                  │                              │ add_ceiling_floor_features│
│ Source: vnstock  │                              │ add_target() (train)  │
│ Format: parquet  │                              │                       │
│ Price unit: 1000đ│                              │ → FEATURE_COLS (48)   │
└──────────────────┘                              └───────────┬───────────┘
                                                              │
                    ┌─────────────────┐                      │
                    │  src/sector.py  │◀─────────────────────┤
                    │                 │  build_sector_returns │
                    │ 8 sectors       │  add_sector_features  │
                    │ 4 features      │                       │
                    └─────────────────┘                      │
                                                              │
                    ┌─────────────────┐                      │
                    │  src/earnings.py│◀─────────────────────┘
                    │                 │  add_earnings_features
                    │ 4 quarters      │  5 features
                    │ 11 sensitive    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────────────────┐
                    │       src/model.py           │
                    │                             │
                    │ ensemble_predict(            │
                    │   xgb_model,                │
                    │   lgb_model,                │
                    │   label_encoder,            │
                    │   X_live                    │
                    │ ) → (signal, confidence)    │
                    │                             │
                    │ Consensus: XGB == LGB ?     │
                    │   Yes → use pred            │
                    │   No  → HOLD                │
                    └─────────────────────────────┘
                              │
                              ▼ df (30 rows: ticker, signal, confidence, ...)
┌──────────────────────────────────────────────────────────────┐
│                     OVERLAY PIPELINE                         │
│                                                              │
│  src/news.py                                                 │
│  ├─ fetch_all_feeds()      ← 7 Vietstock RSS feeds          │
│  ├─ build_ticker_sentiment() ← scoring -1.0 to +1.0         │
│  └─ apply_news_overlay()   ← 4 rules: downgrade or tag      │
│                                                              │
│  src/live_overlay.py                                         │
│  ├─ foreign flow           ← vnstock price_board API         │
│  │   classify: strong_buy/buy/neutral/sell/strong_sell       │
│  └─ insider trading        ← RSS giao-dich-noi-bo, 30d      │
│       score = Σ(dir × weight × vol_scale)                   │
│       senior insiders: 2x weight                            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼ df enriched with tags + adjusted signals
┌──────────────────────────────────────────────────────────────┐
│                    OUTPUT PIPELINE                            │
│                                                              │
│  save_signals()    → signals/YYYY-MM-DD.csv                  │
│  tracker.run()     → data/tracker/predictions.parquet        │
│                       data/tracker/latest_report.json        │
│  send_signal_email() → Gmail SMTP (App Password)            │
│  dashboard.py      → dashboard/index.html → Vercel deploy   │
└──────────────────────────────────────────────────────────────┘
```

---

## Offline flows (chạy thủ công)

### Training pipeline
```
python -m src.features          # build_dataset() → data/features/features.parquet
python -m src.model             # run_walk_forward() → models/*.json + *.txt
python -m src.backtest          # run_backtest() → backtest/trades.csv
```

### Dashboard rebuild
```
python -m src.dashboard         # build_dashboard() → dashboard/index.html
```

### Tracker manual
```
python -m src.tracker           # run() → resolve + report
```

---

## File map

| File | Responsibility | Key outputs |
|------|---------------|-------------|
| `src/fetch.py` | Fetch & cache OHLCV từ vnstock | `data/raw/*.parquet`, `data/index/*.parquet` |
| `src/features.py` | 48 technical features + target | `data/features/features.parquet` |
| `src/sector.py` | Phân ngành 8 sectors, 4 features | In-memory DataFrame |
| `src/earnings.py` | Mùa KQKD, 5 features | In-memory DataFrame |
| `src/model.py` | XGB + LGB ensemble, walk-forward | `models/xgb_model.json`, `models/lgb_model.txt`, `models/label_classes.npy` |
| `src/signal_generator.py` | Orchestrator hàng ngày | `signals/YYYY-MM-DD.csv` |
| `src/news.py` | Sentiment từ 7 RSS feeds | `data/news/*.csv`, `data/news/*_articles.json` |
| `src/live_overlay.py` | Foreign flow + insider | Tags trên df |
| `src/backtest.py` | Walk-forward backtest | `backtest/trades.csv` |
| `src/tracker.py` | Theo dõi accuracy 5-ngày | `data/tracker/predictions.parquet`, `data/tracker/latest_report.json` |
| `src/dashboard.py` | HTML dashboard | `dashboard/index.html` |
| `src/notifications.py` | Gmail email | (gửi qua SMTP) |
| `run_daily.sh` | Cron entry point | Logs → `logs/YYYY-MM-DD.log` |

---

## Data storage

```
data/
├── raw/           {TICKER}.parquet     OHLCV, price unit = 1000 VND (*)
├── index/         VNINDEX.parquet      OHLCV index
│                  VN30.parquet
├── features/      features.parquet     All tickers, all dates, 48 features + target
├── news/          YYYY-MM-DD.csv       Sentiment per ticker
│                  YYYY-MM-DD_articles.json  Raw articles
└── tracker/       predictions.parquet  Signal history + resolve results
                   latest_report.json   Accuracy score + analysis

models/
├── xgb_model.json
├── lgb_model.txt
└── label_classes.npy

backtest/
└── trades.csv     date, ticker, direction, entry, exit, confidence, pnl

signals/
└── YYYY-MM-DD.csv

logs/
└── YYYY-MM-DD.log
```

> (*) vnstock API trả giá theo đơn vị nghìn VND. `close = 59.5` nghĩa là 59,500 VND.
> signal_generator.py nhân × 1000 khi lưu vào signals CSV và hiển thị.

---

## Constants quan trọng (có thể thành magic numbers cần extract)

| Constant | Value | File | Vấn đề tiềm ẩn |
|----------|-------|------|----------------|
| `FORWARD_DAYS` | 5 | features.py, tracker.py | Trùng nhau, nên centralize |
| `BUY_THRESHOLD` | 0.03 | features.py | Trùng với `BUY_TARGET` trong tracker.py |
| `SELL_THRESHOLD` | -0.02 | features.py | Asymmetric với BUY |
| `RANDOM_BASE` | 0.24 | tracker.py | Hardcode, cần time-varying |
| `HIGH_CONFIDENCE_THRESHOLD` | 0.60 | signal_generator.py | Cũng = `CONFIDENCE_THRESHOLD` trong backtest.py |
| `TRANSACTION_COST` | 0.0015 | backtest.py | Thiếu sell tax (0.1%) và slippage |
| `TRAIN_YEARS` | 3 | model.py | — |
| `TEST_MONTHS` | 6 | model.py | — |
| `CAPITAL_PER_TRADE` | 10,000,000 | backtest.py | Cố định, không liên quan tổng vốn |

---

## Known issues (từ code audit)

1. **Magic numbers phân tán:** `FORWARD_DAYS=5`, `BUY_THRESHOLD=0.03`, `CONFIDENCE_THRESHOLD=0.60` định nghĩa riêng ở nhiều file — nếu sửa 1 chỗ có thể quên chỗ kia.
2. **Không có config file:** Không có `config/constants.py` hay `config.yaml` — tất cả hard-coded trong từng module.
3. **Không có pytest:** Chưa có test infrastructure.
4. **Không có benchmark:** `backtest.py` chỉ track P&L tuyệt đối, không so với VN30 buy-and-hold.
5. **RANDOM_BASE=0.24 hardcode:** Không phản ánh regime thị trường hiện tại.
6. **SELL signals bị mute hoàn toàn:** Override tại inference, không track accuracy riêng.
7. **Model không retrain tự động:** Một lần train, dùng mãi.
8. **Không có circuit breaker:** Không có cơ chế dừng khi model degradation.
