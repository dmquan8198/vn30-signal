# Survivorship Bias Assessment

**Date:** 2026-04-18
**Phase:** P1.4

---

## Vấn đề

Hệ thống hiện tại train và backtest trên **danh sách VN30 tại thời điểm 2026** (30 mã cố định). Tuy nhiên VN30 thay đổi thành phần 2 lần/năm (tháng 1 và tháng 7) do HOSE review định kỳ.

Training trên data từ 2020 với danh sách mã năm 2026 → **survivorship bias**: chỉ giữ lại các mã "sống sót" và đủ điều kiện vào VN30 ở thời điểm hiện tại, bỏ qua các mã đã bị loại ra.

---

## Ảnh hưởng ước tính

- Các nghiên cứu academic về survivorship bias trong equity: **optimistic 2-5%/năm** trên return
- Mức độ trong hệ thống này: **thấp hơn trung bình** vì:
  - VN30 là rổ blue-chip, ít bị xáo trộn mạnh (không như small-cap)
  - Chỉ có ~2-4 mã thay đổi mỗi kỳ review
  - Các mã bị loại thường vẫn là large-cap, không bị delisting

- **Ước tính bias: +1-3%/năm** trên backtest return (optimistic)

---

## Lịch sử thay đổi VN30 (ước tính từ public sources)

| Kỳ review | Mã vào | Mã ra | Ghi chú |
|-----------|--------|-------|---------|
| T1/2024 | BCM, SSB | (unknown) | HOSE cập nhật |
| T7/2023 | (unknown) | (unknown) | — |
| T1/2023 | (unknown) | (unknown) | — |
| ... | ... | ... | Cần lấy từ HOSE archive |

> **Giới hạn:** Không có nguồn dữ liệu miễn phí đầy đủ về lịch sử VN30 constituents.
> Fiingroup (có phí) hoặc HSX trực tiếp là nguồn chính xác nhất.

---

## Phân tích tác động

### Tại sao bias thấp trong trường hợp này:

1. **VN30 là rổ ổn định:** Gần như không có mã nào bị delisting (huỷ niêm yết). Các mã bị loại vẫn giao dịch bình thường, chỉ không đủ tiêu chí vốn hoá/thanh khoản.

2. **Mức độ xáo trộn thấp:** Trung bình 2-3 mã thay đổi/kỳ, tức ~4-6 mã/năm trên tổng số 30 → 13-20% thay đổi/năm.

3. **Data từ 2020:** Các mã hiện tại hầu hết đã có trong rổ từ đầu. BCM, SSB là các mã mới nhất (vào 2024).

### Mã hiện tại **không có trong VN30 toàn bộ giai đoạn 2020-2026:**
- **BCM** (Becamex): Niêm yết 2022, vào VN30 khoảng 2024
- **SSB** (SeABank): IPO 2021, vào VN30 khoảng 2022-2023

### Hành động ngay:
- **BCM:** Có thể dùng data từ ngày IPO (2022), bỏ qua period trước
- **SSB:** Tương tự

---

## Mitigation hiện tại

✅ **Đã apply:**
- Features.py dùng `dropna()` → các ngày không có data cho mã mới sẽ tự bị loại
- Walk-forward splits bắt đầu từ dữ liệu thực tế của từng mã

❌ **Chưa apply:**
- Không dùng historical VN30 constituents list
- Train trên toàn bộ 30 mã hiện tại bất kể thời điểm lịch sử

---

## Khuyến nghị

### Ngắn hạn (pre-live):
- Chấp nhận bias 1-3%/năm với disclaimer rõ ràng
- Ghi chú trong tất cả backtest reports: *"Results có thể optimistic ~1-3%/năm do survivorship bias (historical VN30 constituents không được apply)"*

### Dài hạn (Phase 3+):
- Lấy historical VN30 constituents từ HOSE hoặc Fiingroup
- Tạo `data/vn30_constituents_history.csv` với cột: `date, tickers_in_vn30`
- Modify `build_dataset()` trong features.py để tại mỗi ngày t, chỉ include mã thuộc VN30 tại ngày t

---

## Kết luận

**Mức độ rủi ro: THẤP-TRUNG BÌNH**

Survivorship bias tồn tại nhưng tác động hạn chế trong trường hợp VN30 vì:
- Rổ ổn định, ít xáo trộn
- Không có mã bị delisting
- Data đủ dài từ 2020

**Cần ghi chú trong mọi report:** *"Backtest results có survivorship bias ước tính +1-3%/năm"*
