#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

clone_if_missing() {
  local url="$1"
  local target="$2"
  if [[ -d "${target}/.git" ]]; then
    git -C "${target}" fetch --depth 1 origin
    git -C "${target}" reset --hard FETCH_HEAD
  else
    git clone --depth 1 "${url}" "${target}"
  fi
}

clone_if_missing https://github.com/state-spaces/mamba.git third_party/mamba
clone_if_missing https://github.com/navidivan/rsm.git third_party/rsm
clone_if_missing https://github.com/bowang-lab/Graph-Mamba.git third_party/Graph-Mamba
clone_if_missing https://github.com/LincanLi98/STG-Mamba.git third_party/STG-Mamba
clone_if_missing https://github.com/pyg-team/pytorch_geometric.git third_party/pytorch_geometric
