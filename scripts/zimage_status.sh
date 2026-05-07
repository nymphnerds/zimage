#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

installed=false
env_ready=false
models_ready=unknown
running=false
detail="Not installed."

if [[ -f "${ZIMAGE_INSTALL_ROOT}/api_server.py" ]]; then
  installed=true
  detail="Source installed."
fi

if [[ -x "$(zimage_python)" ]]; then
  env_ready=true
  detail="Runtime environment present."
fi

if zimage_is_running; then
  running=true
fi

if [[ "${env_ready}" == "true" ]]; then
  if (
    cd "${ZIMAGE_INSTALL_ROOT}"
    "$(zimage_python)" -m py_compile api_server.py model_manager.py nunchaku_compat.py scripts/prefetch_model.py >/dev/null 2>&1
    "$(zimage_python)" scripts/prefetch_model.py --local-files-only >/dev/null 2>&1
  ); then
    models_ready=true
    detail="Runtime and cached model files are ready."
  else
    models_ready=false
    detail="Runtime exists, but cached model files are incomplete."
  fi
fi

cat <<EOF
id=zimage
name=Z-Image Turbo
installed=${installed}
env_ready=${env_ready}
models_ready=${models_ready}
running=${running}
url=${Z_IMAGE_SERVER_URL}
install_root=${ZIMAGE_INSTALL_ROOT}
venv=${ZIMAGE_VENV_DIR}
logs=${ZIMAGE_LOG_DIR}
detail=${detail}
EOF
