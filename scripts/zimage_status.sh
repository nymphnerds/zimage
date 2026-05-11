#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

installed=false
runtime_present=false
data_present=false
env_ready=false
models_ready=unknown
running=false
version=not-installed
health=unavailable
state=available
marker="${ZIMAGE_INSTALL_ROOT}/.nymph-module-version"
last_log="${ZIMAGE_LOG_DIR}/zimage-server.log"
gpu_vram_mb="$(zimage_detect_gpu_vram_mb)"
recommended_precision="$(zimage_recommended_precision "${gpu_vram_mb}")"
detail="Not installed."

if [[ -f "${marker}" ]]; then
  installed=true
  runtime_present=true
  version="$(head -n 1 "${marker}" 2>/dev/null || true)"
  [[ -n "${version}" ]] || version=unknown
  detail="Source installed."
fi

if [[ -d "${ZIMAGE_OUTPUT_DIR}" && -n "$(find "${ZIMAGE_OUTPUT_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]] ||
   [[ -d "${ZIMAGE_INSTALL_ROOT}/outputs" && -n "$(find "${ZIMAGE_INSTALL_ROOT}/outputs" -mindepth 1 -print -quit 2>/dev/null)" ]] ||
   [[ -d "${ZIMAGE_LOG_DIR}" && -n "$(find "${ZIMAGE_LOG_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]] ||
   [[ -d "${ZIMAGE_INSTALL_ROOT}/logs" && -n "$(find "${ZIMAGE_INSTALL_ROOT}/logs" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  data_present=true
fi

if [[ "${installed}" == "true" && -x "$(zimage_python)" ]]; then
  env_ready=true
  detail="Runtime environment present."
fi

if [[ "${installed}" == "true" ]] && zimage_is_running; then
  running=true
  if zimage_probe_url "${Z_IMAGE_SERVER_URL}/health" >/dev/null 2>&1; then
    health=ok
  else
    health=unreachable
  fi
fi

if [[ "${env_ready}" == "true" ]]; then
  if (
    cd "${ZIMAGE_INSTALL_ROOT}"
    "$(zimage_python)" -m py_compile api_server.py model_manager.py nunchaku_compat.py scripts/prefetch_model.py >/dev/null 2>&1
    "$(zimage_python)" scripts/prefetch_model.py --local-files-only >/dev/null 2>&1
  ); then
    models_ready=true
    if [[ "${running}" != "true" ]]; then
      health=ok
    fi
    detail="Runtime and cached model files are ready."
  else
    models_ready=false
    health=degraded
    detail="Runtime exists, but cached model files are incomplete."
  fi
fi

if [[ "${installed}" == "true" && "${running}" == "true" ]]; then
  state=running
elif [[ "${installed}" == "true" && "${health}" == "degraded" ]]; then
  state=needs_attention
elif [[ "${installed}" == "true" ]]; then
  state=installed
  if [[ "${env_ready}" != "true" ]]; then
    state=needs_attention
    health=degraded
    detail="Z-Image runtime files are installed, but the Python runtime is missing."
  fi
elif [[ "${data_present}" == "true" ]]; then
  detail="Z-Image preserved data remains, but runtime files are not installed."
fi

cat <<EOF
id=zimage
name=Z-Image Turbo
installed=${installed}
runtime_present=${runtime_present}
data_present=${data_present}
version=${version}
env_ready=${env_ready}
models_ready=${models_ready}
running=${running}
state=${state}
health=${health}
url=${Z_IMAGE_SERVER_URL}
health_url=${Z_IMAGE_SERVER_URL}/health
server_info_url=${Z_IMAGE_SERVER_URL}/server_info
model_id=${Z_IMAGE_MODEL_ID}
nunchaku_weight_repo=${Z_IMAGE_NUNCHAKU_MODEL_REPO}
nunchaku_rank=${Z_IMAGE_NUNCHAKU_RANK}
nunchaku_precision=${Z_IMAGE_NUNCHAKU_PRECISION}
recommended_precision=${recommended_precision}
gpu_vram_mb=${gpu_vram_mb:-unknown}
install_root=${ZIMAGE_INSTALL_ROOT}
outputs_dir=${ZIMAGE_OUTPUT_DIR}
hf_cache_dir=${NYMPHS3D_HF_CACHE_DIR}
venv=${ZIMAGE_VENV_DIR}
logs_dir=${ZIMAGE_LOG_DIR}
last_log=${last_log}
marker=${marker}
detail=${detail}
EOF
