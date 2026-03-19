#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config> [nproc_per_node]" >&2
  exit 1
fi

CONFIG="$1"
NPROC_PER_NODE="${2:-2}"

"${ROOT_DIR}/.venv/bin/torchrun" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/run_train.py \
  --config "${CONFIG}"
