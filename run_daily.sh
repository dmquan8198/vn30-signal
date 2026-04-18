#!/bin/bash
# run_daily.sh — Chạy tự động 2 lần/ngày giao dịch
# Cron 1: 0 9 * * 1-5  (09:00 — trước khi thị trường mở, lấy tín hiệu sáng)
# Cron 2: 30 13 * * 1-5 (13:30 — giữa phiên chiều, cập nhật foreign flow + news)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python3"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d).log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Run at: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

cd "$PROJECT_DIR"

# Fetch fresh data + generate signals + news overlay
"$PYTHON" -m src.signal_generator --refresh 2>&1 | tee -a "$LOG_FILE"

# Rebuild dashboard
"$PYTHON" -m src.dashboard 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Done: $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
