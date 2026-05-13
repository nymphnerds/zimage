#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ZIMAGE_INSTALL_ROOT="${ZIMAGE_INSTALL_ROOT:-$HOME/Z-Image}"
ZIMAGE_VENV_DIR="${ZIMAGE_VENV_DIR:-$ZIMAGE_INSTALL_ROOT/.venv-nunchaku}"
NYMPHS_DATA_ROOT="${NYMPHS_DATA_ROOT:-$HOME/NymphsData}"
ZIMAGE_CONFIG_DIR="${ZIMAGE_CONFIG_DIR:-$NYMPHS_DATA_ROOT/config/zimage}"
ZIMAGE_PRESET_FILE="${ZIMAGE_PRESET_FILE:-$ZIMAGE_CONFIG_DIR/generation-preset.env}"
ZIMAGE_OUTPUT_DIR="${ZIMAGE_OUTPUT_DIR:-$NYMPHS_DATA_ROOT/outputs/zimage}"
ZIMAGE_LOG_DIR="${ZIMAGE_LOG_DIR:-$NYMPHS_DATA_ROOT/logs/zimage}"
ZIMAGE_PID_FILE="${ZIMAGE_PID_FILE:-$ZIMAGE_LOG_DIR/zimage.pid}"
Z_IMAGE_PORT="${Z_IMAGE_PORT:-8090}"
Z_IMAGE_HOST="${Z_IMAGE_HOST:-127.0.0.1}"
Z_IMAGE_SERVER_URL="${Z_IMAGE_SERVER_URL:-http://${Z_IMAGE_HOST}:${Z_IMAGE_PORT}}"
Z_IMAGE_MODEL_ID="${Z_IMAGE_MODEL_ID:-Tongyi-MAI/Z-Image-Turbo}"

ZIMAGE_NUNCHAKU_REPO="${ZIMAGE_NUNCHAKU_REPO:-https://github.com/nymphnerds/nunchaku.git}"
ZIMAGE_NUNCHAKU_COMMIT="${ZIMAGE_NUNCHAKU_COMMIT:-a2a4f2444a092974ba53323ba0681a523ff98031}"
ZIMAGE_NUNCHAKU_SPEC="${ZIMAGE_NUNCHAKU_SPEC:-git+${ZIMAGE_NUNCHAKU_REPO}@${ZIMAGE_NUNCHAKU_COMMIT}}"
ZIMAGE_DIFFUSERS_SPEC="${ZIMAGE_DIFFUSERS_SPEC:-diffusers==0.37.1}"
ZIMAGE_TORCH_INDEX_URL="${ZIMAGE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

preset_precision=""
preset_rank=""
if [[ -f "${ZIMAGE_PRESET_FILE}" ]]; then
  while IFS='=' read -r key value; do
    case "${key}" in
      Z_IMAGE_NUNCHAKU_PRECISION)
        case "${value}" in
          int4|fp4|auto) preset_precision="${value}" ;;
        esac
        ;;
      Z_IMAGE_NUNCHAKU_RANK)
        case "${value}" in
          32|128|256) preset_rank="${value}" ;;
        esac
        ;;
    esac
  done < "${ZIMAGE_PRESET_FILE}"
fi

export Z_IMAGE_RUNTIME="${Z_IMAGE_RUNTIME:-nunchaku}"
export NYMPHS2D2_RUNTIME="${NYMPHS2D2_RUNTIME:-$Z_IMAGE_RUNTIME}"
export Z_IMAGE_MODEL_ID
export NYMPHS2D2_MODEL_ID="${NYMPHS2D2_MODEL_ID:-$Z_IMAGE_MODEL_ID}"
export Z_IMAGE_PORT
export NYMPHS2D2_PORT="${NYMPHS2D2_PORT:-$Z_IMAGE_PORT}"
export Z_IMAGE_OUTPUT_DIR
export NYMPHS2D2_OUTPUT_DIR="${NYMPHS2D2_OUTPUT_DIR:-$ZIMAGE_OUTPUT_DIR}"
export Z_IMAGE_NUNCHAKU_MODEL_REPO="${Z_IMAGE_NUNCHAKU_MODEL_REPO:-nunchaku-ai/nunchaku-z-image-turbo}"
export Z_IMAGE_NUNCHAKU_RANK="${Z_IMAGE_NUNCHAKU_RANK:-${preset_rank:-32}}"
export Z_IMAGE_NUNCHAKU_PRECISION="${Z_IMAGE_NUNCHAKU_PRECISION:-${preset_precision:-auto}}"
export Z_IMAGE_NUNCHAKU_IMG2IMG="${Z_IMAGE_NUNCHAKU_IMG2IMG:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export NYMPHS3D_HF_CACHE_DIR="${NYMPHS3D_HF_CACHE_DIR:-$NYMPHS_DATA_ROOT/cache/huggingface}"

if [[ -d /usr/local/cuda-13.0 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

zimage_python() {
  printf '%s\n' "${ZIMAGE_VENV_DIR}/bin/python"
}

zimage_ensure_log_dir() {
  mkdir -p "${ZIMAGE_LOG_DIR}"
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

zimage_detect_gpu_vram_mb() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]' || true
    return
  fi

  printf '\n'
}

zimage_recommended_precision() {
  local vram_mb="${1:-}"
  if [[ "${vram_mb}" =~ ^[0-9]+$ ]]; then
    if (( vram_mb <= 16384 )); then
      printf 'int4\n'
      return
    fi
  fi

  printf 'auto\n'
}
