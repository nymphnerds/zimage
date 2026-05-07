#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ZIMAGE_INSTALL_ROOT="${ZIMAGE_INSTALL_ROOT:-$HOME/Z-Image}"
ZIMAGE_VENV_DIR="${ZIMAGE_VENV_DIR:-$ZIMAGE_INSTALL_ROOT/.venv-nunchaku}"
ZIMAGE_LOG_DIR="${ZIMAGE_LOG_DIR:-$ZIMAGE_INSTALL_ROOT/logs}"
ZIMAGE_PID_FILE="${ZIMAGE_PID_FILE:-$ZIMAGE_LOG_DIR/zimage.pid}"
Z_IMAGE_PORT="${Z_IMAGE_PORT:-8090}"
Z_IMAGE_HOST="${Z_IMAGE_HOST:-127.0.0.1}"
Z_IMAGE_SERVER_URL="${Z_IMAGE_SERVER_URL:-http://${Z_IMAGE_HOST}:${Z_IMAGE_PORT}}"

ZIMAGE_NUNCHAKU_REPO="${ZIMAGE_NUNCHAKU_REPO:-https://github.com/nymphnerds/nunchaku.git}"
ZIMAGE_NUNCHAKU_COMMIT="${ZIMAGE_NUNCHAKU_COMMIT:-a2a4f2444a092974ba53323ba0681a523ff98031}"
ZIMAGE_NUNCHAKU_SPEC="${ZIMAGE_NUNCHAKU_SPEC:-git+${ZIMAGE_NUNCHAKU_REPO}@${ZIMAGE_NUNCHAKU_COMMIT}}"
ZIMAGE_DIFFUSERS_SPEC="${ZIMAGE_DIFFUSERS_SPEC:-diffusers==0.37.1}"
ZIMAGE_TORCH_INDEX_URL="${ZIMAGE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

export Z_IMAGE_RUNTIME="${Z_IMAGE_RUNTIME:-nunchaku}"
export NYMPHS2D2_RUNTIME="${NYMPHS2D2_RUNTIME:-$Z_IMAGE_RUNTIME}"
export Z_IMAGE_MODEL_ID="${Z_IMAGE_MODEL_ID:-Tongyi-MAI/Z-Image-Turbo}"
export NYMPHS2D2_MODEL_ID="${NYMPHS2D2_MODEL_ID:-$Z_IMAGE_MODEL_ID}"
export Z_IMAGE_PORT
export NYMPHS2D2_PORT="${NYMPHS2D2_PORT:-$Z_IMAGE_PORT}"
export Z_IMAGE_NUNCHAKU_MODEL_REPO="${Z_IMAGE_NUNCHAKU_MODEL_REPO:-nunchaku-ai/nunchaku-z-image-turbo}"
export Z_IMAGE_NUNCHAKU_RANK="${Z_IMAGE_NUNCHAKU_RANK:-32}"
export Z_IMAGE_NUNCHAKU_PRECISION="${Z_IMAGE_NUNCHAKU_PRECISION:-auto}"
export Z_IMAGE_NUNCHAKU_IMG2IMG="${Z_IMAGE_NUNCHAKU_IMG2IMG:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ -d /usr/local/cuda-13.0 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${ZIMAGE_LOG_DIR}"

zimage_python() {
  printf '%s\n' "${ZIMAGE_VENV_DIR}/bin/python"
}

zimage_is_running() {
  if [[ -f "${ZIMAGE_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${ZIMAGE_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

zimage_probe_url() {
  local url="${1}"
  python3 - "${url}" <<'PY'
import sys
from urllib.request import urlopen

try:
    with urlopen(sys.argv[1], timeout=2) as response:
        print(response.read().decode("utf-8", errors="replace"))
except Exception as exc:
    raise SystemExit(str(exc))
PY
}
