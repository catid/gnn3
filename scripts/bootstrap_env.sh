#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

uv venv --python 3.12 .venv
uv sync --all-groups
uv pip install --python .venv/bin/python --prerelease allow -r requirements/torch-nightly-cu130.txt
