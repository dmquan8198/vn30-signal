#!/usr/bin/env bash
# [P2.1] weekly_retrain.sh — Tự động retrain model mỗi tuần (chạy Thứ Bảy ~21:00)
# Cron setup: 0 21 * * 6 /path/to/vn30-signal/scripts/weekly_retrain.sh >> logs/retrain.log 2>&1
#
# Pipeline:
#   1. Fetch fresh data for all 30 VN30 tickers
#   2. Rebuild features
#   3. Run walk-forward validation + save final model
#   4. Run drift detection (warn if PSI alert)
#   5. Run circuit breaker check
#   6. Run bootstrap CI on new backtest trades
#   7. Send email report (optional)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
REPORTS_DIR="$REPO_DIR/reports"
VENV="$REPO_DIR/.venv/bin/python"
DATE=$(date +%Y%m%d)

mkdir -p "$LOG_DIR" "$REPORTS_DIR"

echo ""
echo "=============================="
echo "  VN30 Weekly Retrain — $DATE"
echo "=============================="
echo "Repo: $REPO_DIR"
echo "Python: $VENV"
echo ""

cd "$REPO_DIR"

step() {
    echo ""
    echo "── $1"
}

# ── Step 1: Fetch data
step "1/7  Fetching VN30 price data..."
$VENV -c "
from src.fetch import fetch_all, fetch_indices
print('  Refreshing index data...')
fetch_indices(delay=1.5)
print('  Fetching all tickers...')
fetch_all(delay=3.5)
print('  Done.')
"

# ── Step 2: Rebuild features
step "2/7  Rebuilding features..."
$VENV -m src.features
echo "  Features rebuilt."

# ── Step 3: Walk-forward + retrain
step "3/7  Walk-forward validation + retrain..."
$VENV -m src.model
echo "  Model retrained and saved."

# ── Step 4: Drift detection
step "4/7  Drift detection (PSI)..."
DRIFT_OUT=$($VENV -c "
from monitoring.drift import run_drift_check
result = run_drift_check()
alerts = result.get('alerts', [])
if alerts:
    print('ALERT: ' + '; '.join(alerts))
else:
    print('OK: No drift detected')
" 2>&1)
echo "  $DRIFT_OUT"

# ── Step 5: Circuit breaker check
step "5/7  Circuit breaker check..."
CB_STATUS=$($VENV -c "
from monitoring.circuit_breaker import CircuitBreaker
cb = CircuitBreaker()
status = cb.check()
print(status['status'] + ': ' + status['reason'])
" 2>&1)
echo "  $CB_STATUS"

# If circuit breaker OPEN → abort signal generation today
if echo "$CB_STATUS" | grep -q "^OPEN:"; then
    echo ""
    echo "🚨 CIRCUIT BREAKER OPEN — halting retrain pipeline"
    echo "   Check reports/circuit_breaker.json for details"
    exit 1
fi

# ── Step 6: Bootstrap CI on latest backtest
step "6/7  Bootstrap CI (latest backtest)..."
TRADES="$REPO_DIR/backtest/trades.csv"
if [ -f "$TRADES" ]; then
    $VENV -c "
from backtest.bootstrap import main as run_bootstrap
report = run_bootstrap()
buy_ci = report['results'].get('buy', {}).get('go_live_criteria', {})
print('  BUY go-live:', buy_ci.get('result', 'N/A'))
" 2>&1 | grep -v "^Running\|^This may\|Bootstrapping"
else
    echo "  No trades.csv found — skipping bootstrap"
fi

# ── Step 7: Generate today's signals
step "7/7  Generating signals for next trading day..."
$VENV -m src.signal_generator
echo ""

echo "=============================="
echo "  Retrain complete — $DATE"
echo "=============================="
echo ""
