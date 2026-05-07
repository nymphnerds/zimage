#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

echo "Z-Image Turbo: installing module into ${ZIMAGE_INSTALL_ROOT}"

mkdir -p "${ZIMAGE_INSTALL_ROOT}" "${ZIMAGE_INSTALL_ROOT}/scripts" "${ZIMAGE_LOG_DIR}"

install -m 644 "${MODULE_ROOT}/.env.example" "${ZIMAGE_INSTALL_ROOT}/.env.example"
install -m 644 "${MODULE_ROOT}/.gitignore" "${ZIMAGE_INSTALL_ROOT}/.gitignore"
install -m 644 "${MODULE_ROOT}/README.md" "${ZIMAGE_INSTALL_ROOT}/README.md"
install -m 644 "${MODULE_ROOT}/nymph.json" "${ZIMAGE_INSTALL_ROOT}/nymph.json"
install -m 644 "${MODULE_ROOT}/api_server.py" "${ZIMAGE_INSTALL_ROOT}/api_server.py"
install -m 644 "${MODULE_ROOT}/config.py" "${ZIMAGE_INSTALL_ROOT}/config.py"
install -m 644 "${MODULE_ROOT}/image_store.py" "${ZIMAGE_INSTALL_ROOT}/image_store.py"
install -m 644 "${MODULE_ROOT}/model_manager.py" "${ZIMAGE_INSTALL_ROOT}/model_manager.py"
install -m 644 "${MODULE_ROOT}/nunchaku_compat.py" "${ZIMAGE_INSTALL_ROOT}/nunchaku_compat.py"
install -m 644 "${MODULE_ROOT}/progress_state.py" "${ZIMAGE_INSTALL_ROOT}/progress_state.py"
install -m 644 "${MODULE_ROOT}/requirements.lock.txt" "${ZIMAGE_INSTALL_ROOT}/requirements.lock.txt"
install -m 644 "${MODULE_ROOT}/schemas.py" "${ZIMAGE_INSTALL_ROOT}/schemas.py"
install -m 755 "${MODULE_ROOT}/scripts/prefetch_model.py" "${ZIMAGE_INSTALL_ROOT}/scripts/prefetch_model.py"
install -m 755 "${MODULE_ROOT}/scripts/run_nunchaku_zimage_test.py" "${ZIMAGE_INSTALL_ROOT}/scripts/run_nunchaku_zimage_test.py"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 is required for Z-Image Turbo." >&2
  exit 1
fi

if [[ ! -x "${ZIMAGE_VENV_DIR}/bin/python" ]]; then
  echo "Creating runtime venv: ${ZIMAGE_VENV_DIR}"
  python3.11 -m venv "${ZIMAGE_VENV_DIR}"
fi

VENV_PYTHON="$(zimage_python)"

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install torch==2.11.0 torchvision torchaudio --index-url "${ZIMAGE_TORCH_INDEX_URL}"
"${VENV_PYTHON}" -m pip install -r "${ZIMAGE_INSTALL_ROOT}/requirements.lock.txt"
"${VENV_PYTHON}" -m pip install "${ZIMAGE_NUNCHAKU_SPEC}"
"${VENV_PYTHON}" -m pip install --force-reinstall "${ZIMAGE_DIFFUSERS_SPEC}"

(
  cd "${ZIMAGE_INSTALL_ROOT}"
  "${VENV_PYTHON}" -m py_compile api_server.py config.py image_store.py model_manager.py nunchaku_compat.py progress_state.py schemas.py scripts/prefetch_model.py scripts/run_nunchaku_zimage_test.py
  "${VENV_PYTHON}" - <<'PY'
from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku_compat import patch_zimage_transformer_forward

patched = patch_zimage_transformer_forward(NunchakuZImageTransformer2DModel)
if not patched:
    raise SystemExit("nunchaku forward patch validation failed")
print("Z-Image Turbo runtime imports validated.")
PY
)

cat > "${ZIMAGE_VENV_DIR}/.nymphs_nunchaku_runtime.json" <<EOF
{
  "nunchaku_repo": "${ZIMAGE_NUNCHAKU_REPO}",
  "nunchaku_commit": "${ZIMAGE_NUNCHAKU_COMMIT}",
  "diffusers": "${ZIMAGE_DIFFUSERS_SPEC}",
  "torch_index_url": "${ZIMAGE_TORCH_INDEX_URL}"
}
EOF

echo "Z-Image Turbo install complete."
