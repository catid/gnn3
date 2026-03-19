#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <exploit-config> <explore-config>" >&2
  exit 1
fi

EXPLOIT_CONFIG="$1"
EXPLORE_CONFIG="$2"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="artifacts/schedules/${STAMP}"
mkdir -p "${LOG_DIR}"

.venv/bin/python scripts/run_train.py --config "${EXPLOIT_CONFIG}" \
  > "${LOG_DIR}/exploit.log" 2>&1 &
EXPLOIT_PID=$!

.venv/bin/python scripts/run_train.py --config "${EXPLORE_CONFIG}" \
  > "${LOG_DIR}/explore.log" 2>&1 &
EXPLORE_PID=$!

echo "exploit_pid=${EXPLOIT_PID}"
echo "explore_pid=${EXPLORE_PID}"
echo "log_dir=${LOG_DIR}"

wait "${EXPLOIT_PID}"
wait "${EXPLORE_PID}"
