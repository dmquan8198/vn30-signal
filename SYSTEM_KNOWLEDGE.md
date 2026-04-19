# VN30 Signal System — Tài liệu kỹ thuật đầy đủ

> Mục đích: Ghi lại toàn bộ cách hệ thống hoạt động để đánh giá phương pháp phân tích và cải thiện.

---

## 1. Tổng quan hệ thống

Hệ thống sinh tín hiệu BUY/HOLD hàng ngày cho 30 mã thuộc chỉ số VN30, dựa trên ensemble ML kết hợp phân tích kỹ thuật, sentiment tin tức, dòng tiền ngoại và giao dịch nội bộ.

**Luồng xử lý:**
```
Fetch OHLCV (vnstock API)
    → Tính 48 features (kỹ thuật + thị trường + ngành + kết quả KD)
    → Ensemble XGBoost + LightGBM → signal + confidence
    → Overlay: News sentiment + Foreign flow + Insider trading
    → Lưu signals CSV + Rebuild HTML dashboard
    → Gửi email thông báo
    → Tracker: ghi dự đoán, resolve sau 5 ngày, tính điểm accuracy
```

**Chạy:** 2 lần/ngày (9:00 và 13:30, weekdays) qua `run_daily.sh`.

---

## 2. Dữ liệu đầu vào

### Nguồn dữ liệu
- **Thư viện:** `vnstock` với source `KBS` (Kỹ Bảng Chứng Khoán)
- **Dữ liệu:** OHLCV (Open, High, Low, Close, Volume) hàng ngày
- **Lịch sử:** Từ 2020-01-01 đến hiện tại (~150 ngày gần nhất để tính features)
- **Lưu trữ:** `data/raw/{TICKER}.parquet`, `data/index/{SYMBOL}.parquet`

### 30 mã VN30
```
ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG,
MBB, MSN, MWG, PLX, POW, SAB, SHB, SSB, SSI, STB,
TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE
```

### Chỉ số thị trường
- VNINDEX (thị trường rộng)
- VN30 (top 30 theo vốn hóa)

---

## 3. Feature Engineering — 48 đặc trưng

### 3.1 Định nghĩa nhãn (target)
- **Horizon:** 5 ngày giao dịch (`FORWARD_DAYS = 5`)
- **BUY:** Lợi nhuận 5 ngày tới ≥ **+3%** (`BUY_THRESHOLD = 0.03`)
- **SELL:** Lợi nhuận 5 ngày tới ≤ **-2%** (`SELL_THRESHOLD = -0.02`)
- **HOLD:** Còn lại (khoảng -2% đến +3%)

> ⚠️ **Điểm chú ý:** Ngưỡng BUY (+3%) cao hơn SELL (2%) về tuyệt đối, tạo asymmetry: bias về phía thận trọng. Tuy nhiên nhãn SELL trong training nhưng bị loại hoàn toàn ở inference (xem mục 4).

### 3.2 Chỉ báo kỹ thuật (23 features)

| Nhóm | Features | Tham số |
|------|----------|---------|
| **Lợi nhuận** | ret_1d, ret_3d, ret_5d, ret_10d, ret_20d | — |
| **MA** | ma5, ma10, ma20, ma50 | Simple MA |
| **Vị trí vs MA** | close_vs_ma5/10/20/50 | (close/MA) - 1 |
| **Xu hướng** | ma20_slope | % thay đổi MA20 trong 5 ngày |
| **Biến động** | atr14, atr14_pct, bb_width | ATR(14), BB(20,2) |
| **Momentum** | rsi14, rsi7 | RSI 14 và 7 ngày |
| **MACD** | macd, macd_signal, macd_hist | (12,26,9) |
| **Stochastic** | stoch_k, stoch_d | (14,3,3) |
| **Khối lượng** | vol_ratio, obv, obv_ma10, obv_vs_ma | vol/vol_ma20 |
| **Vị thế giá** | price_pct_range20 | (close-min20)/(max20-min20) |
| **Nến** | body_size, upper_shadow, lower_shadow | so với open |

### 3.3 Thị trường (7 features)

| Feature | Ý nghĩa |
|---------|---------|
| `vni_bull_regime` | 1 nếu VNINDEX MA20 > MA50, else 0 |
| `vni_vs_ma50` | (VNINDEX / MA50) - 1 |
| `vni_ma20_slope` | Độ dốc MA20 VNINDEX (5 ngày) |
| `vni_vol20` | Std rolling 20 ngày của return VNINDEX |
| `vni_rsi14` | RSI(14) của VNINDEX |
| `vni_ret_5d` | Return 5 ngày của VNINDEX |
| `vni_vol_ratio` | Volume VNINDEX / vol_ma20 |

### 3.4 Sức mạnh tương đối (3 features)
- `rs_vs_vni_5d`, `rs_vs_vni_10d`, `rs_vs_vni_20d`: Return mã - Return VNINDEX

### 3.5 Trần/Sàn (6 features)
- Ngưỡng: `CEILING_THRESH = +6.8%`, `FLOOR_THRESH = -6.8%`
- `ceil_days_5d/10d`, `floor_days_5d/10d`: Số lần chạm trần/sàn trong cửa sổ
- `ceil_streak`, `floor_streak`: Số ngày liên tiếp chạm trần/sàn kết thúc hôm nay

### 3.6 Ngành (4 features)

| Feature | Ý nghĩa |
|---------|---------|
| `sector_ret_5d/10d` | Return trung bình ngành 5/10 ngày |
| `stock_vs_sector_5d` | Mã vượt trội ngành 5 ngày |
| `sector_bull` | 1 nếu ngành > MA10 |

**8 ngành:**
- Banking (13 mã): ACB, BID, CTG, HDB, MBB, SHB, SSB, STB, TCB, TPB, VCB, VIB, VPB
- Real Estate (4): BCM, VHM, VIC, VRE
- Energy (3): GAS, PLX, POW
- Material (2): GVR, HPG
- Consumer (4): MWG, MSN, SAB, VNM
- Financial (1): SSI | Aviation (1): VJC | Tech (1): FPT

### 3.7 Mùa kết quả kinh doanh (5 features)

| Feature | Ý nghĩa |
|---------|---------|
| `in_earnings_season` | 1 nếu đang trong mùa KQKD |
| `earnings_season_type` | 0=không, 1=Q1/Q2/Q3, 2=Q4 (cao nhất) |
| `days_to_earnings_norm` | Ngày đến mùa tiếp theo / 90, clip [0,1] |
| `earnings_sensitive` | 1 nếu là mã nhạy cảm KQ |
| `earnings_boost` | in_season × sensitive (interaction term) |

**Cửa sổ mùa KQKD:**
- Q4 (KQKD cả năm): 15/1 – 28/2 ← tác động lớn nhất
- Q1: 15/4 – 15/5 | Q2: 15/7 – 15/8 | Q3: 15/10 – 15/11

**11 mã nhạy cảm KQKD:** FPT, MWG, TCB, VCB, HPG, VHM, ACB, MBB, VPB, CTG, BID

---

## 4. Mô hình ML — Ensemble XGBoost + LightGBM

### 4.1 Chiến lược validation
- **Walk-forward với expanding window**
- Train: 3 năm (`TRAIN_YEARS = 3`)
- Test: 6 tháng (`TEST_MONTHS = 6`)
- Yêu cầu: ≥100 mẫu train, ≥20 mẫu test mỗi split

### 4.2 Tham số mô hình

| Tham số | XGBoost | LightGBM |
|---------|---------|---------|
| n_estimators | 400 | 400 |
| max_depth | 5 | 6 |
| learning_rate | 0.04 | 0.04 |
| num_leaves | — | 40 |
| subsample | 0.8 | 0.8 |
| colsample_bytree | 0.7 | 0.7 |
| reg_alpha (L1) | 0.05 | 0.05 |
| reg_lambda (L2) | 1.0 | 1.0 |
| min_child_weight | 3 | — |
| min_child_samples | — | 15 |

### 4.3 Ensemble & ra quyết định

```
proba_xgb = XGBoost.predict_proba(X)    # shape: [n, 3] (Sell, Hold, Buy)
proba_lgb = LightGBM.predict_proba(X)

proba_avg = (proba_xgb + proba_lgb) / 2   # trung bình đơn giản

prediction = argmax(proba_avg)
confidence = max(proba_avg)               # xác suất cao nhất

# Consensus rule: nếu 2 model bất đồng → mặc định HOLD
if XGB_pred != LGB_pred:
    prediction = HOLD
```

> ⚠️ **Điểm chú ý:** Ensemble weights bằng nhau (50/50), không có calibration theo performance thực tế. Có thể cân nhắc weighted ensemble dựa trên accuracy gần đây.

### 4.4 Override SELL → HOLD
- Tất cả dự đoán SELL đều bị chuyển thành HOLD ở inference
- Lý do: Backtest xác nhận tín hiệu SELL không đáng tin cậy

---

## 5. Overlay phi mô hình

### 5.1 Tin tức / Sentiment (news.py)

**Nguồn:** 7 RSS feeds từ Vietstock (cổ phiếu, thị trường, KD, cổ tức, nội bộ, PTKT, tăng vốn/M&A)

**Cách tính sentiment (-1.0 đến +1.0):**
- ~48 từ tích cực, ~40 từ tiêu cực
- Từ trọng số cao (2x dương): *kỷ lục, bứt phá, vượt kế hoạch, cổ tức, khuyến nghị mua...*
- Từ trọng số cao (2x âm): *vi phạm, cảnh báo, khởi tố, thua lỗ, hủy niêm yết...*
- Công thức: `(pos - neg) / (pos + neg)` (đã nhân trọng số)
- Lookback: 3 ngày (`lookback_days = 3`)

**Quy tắc điều chỉnh signal:**

| Điều kiện | Hành động |
|-----------|-----------|
| BUY + `negative_flag = 1` | → HOLD + tag "⚠️ tin xấu" |
| BUY + `sentiment_1d < -0.3` | → HOLD + tag "📰 sentiment âm" |
| BUY + `sentiment_1d > 0.2` | Giữ BUY + tag "📰 news confirmed" |
| HOLD + `sentiment_1d > 0.4` + có tin | Giữ HOLD + tag "👀 watch: tin tốt" |

### 5.2 Dòng tiền ngoại (live_overlay.py)

**Phân loại từ price_board Vnstock:**
```
foreign_net_pct = (buy_vol - sell_vol) / total_vol

> +10%  → strong_buy
+3-10%  → buy
-3-+3%  → neutral
-10--3% → sell
< -10%  → strong_sell
```

**Quy tắc:**

| Điều kiện | Hành động |
|-----------|-----------|
| BUY + room_full + ngoại bán/bán mạnh | → HOLD + "⚠️ ngoại rút vốn" |
| BUY + ngoại mua mạnh | Tag "🌊 ngoại mua mạnh" |
| BUY + ngoại bán mạnh | → HOLD + "🌊 ngoại bán mạnh" |
| floor_streak ≥ 2 + BUY | → HOLD + "📉 sàn N ngày liên tiếp" |
| floor_streak == 1 + BUY | Tag "⚠️ vừa chạm sàn" |
| ceil_streak ≥ 2 | Tag "📈 trần N ngày liên tiếp" |

### 5.3 Giao dịch nội bộ (live_overlay.py)

**Nguồn:** RSS feed `giao-dich-noi-bo`, lookback 30 ngày

**Tính điểm insider:**
- Phân biệt mua/bán qua keyword
- Senior insiders (Chủ tịch, TGĐ, CEO, CFO, HĐQT): trọng số x2
- Volume: trích xuất từ văn bản, chuẩn hóa theo đơn vị (triệu/nghìn CP)
- `insider_score = Σ (direction × weight × (1 + vol_scale))`

**Quy tắc:**

| Điều kiện | Hành động |
|-----------|-----------|
| insider_buy + BUY | Tag "👤 nội bộ đang mua" |
| insider_buy + HOLD | Tag "👤 nội bộ mua (theo dõi)" |
| insider_sell + BUY | Tag "👤 nội bộ đang bán" |

---

## 6. Backtest

**Tham số:**

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `CAPITAL_PER_TRADE` | 10,000,000 VND | Không đổi theo vốn tổng |
| `HOLD_DAYS` | 5 ngày | Nắm giữ cố định |
| `TRANSACTION_COST` | 0.15% / chiều | Tổng khứ hồi = 0.30% |
| `CONFIDENCE_THRESHOLD` | 60% | Chỉ trade khi conf ≥ 0.60 |

**Cách tính P&L:**
```
raw_return = (exit_price - entry_price) / entry_price
net_return = raw_return - 2 × 0.0015
pnl = 10,000,000 × net_return
```

**Metrics:**
- Win rate, avg return, Sharpe ratio (annualized với 252/5), max drawdown

> ⚠️ **Điểm chú ý:** Position size cố định (10M/lệnh), không có position sizing theo Kelly hay volatility. Capital_per_trade không gắn với tổng vốn → không biết % rủi ro thực tế.

---

## 7. Tracker — Đánh giá độ chính xác thực tế

### 7.1 Cách ghi nhận và resolve

- **Ghi nhận:** Mỗi ngày chạy, lưu tất cả signals vào `predictions.parquet`
- **Resolve:** Sau đúng 5 ngày giao dịch, so giá đóng cửa
- **Hit:** `actual_return ≥ 3%` → hit = 1
- Chỉ track BUY signals

### 7.2 Công thức điểm dự đoán (0–100)

```
# C1 - Hit Rate (trọng số 50%)
hit_rate = % BUY signals đạt ≥ 3% sau 5 ngày
C1 = max(0, min(100, (hit_rate - 0.24) / (1 - 0.24) × 100))
# Baseline: 24% ngày giao dịch có return ≥ 3% (random)

# C2 - Average Return (trọng số 30%)
avg_return = mean actual_return của BUY signals
C2 = max(0, min(100, avg_return × 400 + 40))
# Map: -10%→0, 0%→40, +10%→80, +20%→100

# C3 - Calibration (trọng số 20%)
# Spearman corr(confidence, hit) — nếu ≥10 mẫu
C3 = max(0, min(100, (corr + 1) / 2 × 100))

Score = 0.5×C1 + 0.3×C2 + 0.2×C3
```

**Thang điểm:**
- < 50: Tệ hơn random
- 50: Ngang random
- 60+: Tốt
- 70+: Xuất sắc

### 7.3 Phân tích tự động

Hệ thống phân tích hit rate theo:
- Nhóm confidence (<50%, 50-60%, 60-70%, ≥70%)
- Chế độ thị trường (BULL vs BEAR)
- Tags (news confirm, ngoại mua/bán, nội bộ, sàn streak)
- Ngành
- Xu hướng gần đây (30 ngày vs tổng thể)
- **Cảnh báo degradation:** Nếu hit rate 30 ngày gần nhất < tổng thể - 5%

---

## 8. Những điểm cần đánh giá / câu hỏi mở

### ❓ Về feature engineering
1. **Giá nhân 1000:** vnstock trả giá theo nghìn VND — đã sửa nhưng cần verify toàn bộ feature tính trên `close` có bị ảnh hưởng không (features tính ratio thì không sao).
2. **Look-ahead bias:** Features dùng `ret_5d` (5 ngày tiếp theo) — cần chắc chắn khi train, target và features được align đúng thứ tự thời gian.
3. **Data leakage qua sector/market context:** Nếu market context chứa forward info thì có leak.

### ❓ Về mô hình
4. **Ensemble 50/50** không có weighted — có thể một model tốt hơn model kia trong điều kiện nhất định.
5. **Consensus = HOLD khi bất đồng** — bảo thủ, nhưng bỏ lỡ nhiều cơ hội. Cần đo lường tỷ lệ bất đồng trong thực tế.
6. **Không có retrain định kỳ** — model train một lần, không cập nhật theo regime mới của thị trường.
7. **Class imbalance:** VN thị trường bull > bear, tỷ lệ nhãn BUY/HOLD/SELL trong training data ảnh hưởng lớn đến model.

### ❓ Về overlay
8. **Overlay là hard rule**, không phải soft adjustment — downgrade BUY → HOLD ngay lập tức khi có tin xấu/sàn streak. Chưa có backtest riêng cho overlay rules để đánh giá hiệu quả của từng rule.
9. **Sentiment từ Vietstock RSS** — nguồn này có thể có độ trễ (tin cũ vẫn còn trong feed). Lookback 3 ngày có thể quá ngắn hoặc quá dài tùy loại tin.

### ❓ Về backtest & đánh giá
10. **Không có benchmark:** Chưa so sánh với Buy-and-hold VN30 ETF (E1VFVN30) hay VNINDEX return trong cùng kỳ.
11. **Position sizing cố định 10M/lệnh:** Không phản ánh rủi ro thực. Nếu có 30 BUY signals cùng ngày → cần 300M vốn. Chưa có giới hạn max positions.
12. **Survivorship bias:** VN30 thay đổi thành phần định kỳ — dùng danh sách hiện tại có thể bị bias.
13. **SELL signals bị vô hiệu hóa** — nếu model học tốt short signal thì đang bỏ lãng phí. Cần phân tích riêng SELL signals có đúng không.

### ❓ Về vận hành
14. **Chạy 9:00 trước khi thị trường mở** — tín hiệu dùng giá đóng cửa ngày hôm trước, hợp lý. Nhưng lần 13:30 có dùng giá intraday không? (Hiện chỉ fetch daily OHLCV).
15. **Tracker resolve sau 5 ngày** — không tính đến slippage, không tính đến khả năng không mua được do thanh khoản thấp.

---

## 9. Tóm tắt tất cả ngưỡng quan trọng

| Tham số | Giá trị | File |
|---------|---------|------|
| Forward horizon | 5 ngày | features.py |
| BUY target | +3% | features.py |
| SELL target | -2% | features.py |
| Ceiling/Floor threshold | ±6.8% | features.py |
| High confidence (actionable) | ≥60% | signal_generator.py |
| News downgrade: negative flag | bất kỳ keyword mạnh | news.py |
| News downgrade: sentiment | < -0.3 | news.py |
| News confirm | > +0.2 | news.py |
| News watch | > +0.4 | news.py |
| Foreign strong buy/sell | ±10% | live_overlay.py |
| Foreign buy/sell | ±3% | live_overlay.py |
| Floor streak downgrade | ≥2 ngày liên tiếp | live_overlay.py |
| Insider lookback | 30 ngày | live_overlay.py |
| Backtest capital/lệnh | 10M VND | backtest.py |
| Backtest hold | 5 ngày | backtest.py |
| Backtest cost | 0.15%/chiều | backtest.py |
| Backtest min confidence | 60% | backtest.py |
| Tracker hit target | ≥+3% sau 5 ngày | tracker.py |
| Tracker baseline (random) | 24% | tracker.py |
| Tracker score tốt | ≥60 | tracker.py |
| Model XGB/LGB ensemble | 50/50 | model.py |
| Walk-forward train | 3 năm | model.py |
| Walk-forward test | 6 tháng | model.py |
