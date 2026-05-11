#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

echo "Z-Image Turbo: installing module into ${ZIMAGE_INSTALL_ROOT}"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 is required for Z-Image Turbo." >&2
  exit 1
fi

module_version="$(python3 - "${MODULE_ROOT}/nymph.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

print(str(manifest.get("version", "unknown")).strip() or "unknown")
PY
)"

install_parent="$(dirname "${ZIMAGE_INSTALL_ROOT}")"
mkdir -p "${install_parent}"

STAGING_ROOT="$(mktemp -d "${install_parent}/.zimage-install-staging.XXXXXX")"
OLD_ROOT=""

cleanup_on_exit() {
  local exit_code=$?
  if [[ "${exit_code}" -ne 0 ]]; then
    rm -rf "${STAGING_ROOT}" || true
    if [[ -n "${OLD_ROOT}" && -d "${OLD_ROOT}" && ! -e "${ZIMAGE_INSTALL_ROOT}" ]]; then
      mv "${OLD_ROOT}" "${ZIMAGE_INSTALL_ROOT}" || true
    fi
  fi
}
trap cleanup_on_exit EXIT

mkdir -p "${STAGING_ROOT}/scripts"

install -m 644 "${MODULE_ROOT}/.env.example" "${STAGING_ROOT}/.env.example"
install -m 644 "${MODULE_ROOT}/.gitignore" "${STAGING_ROOT}/.gitignore"
install -m 644 "${MODULE_ROOT}/README.md" "${STAGING_ROOT}/README.md"
install -m 644 "${MODULE_ROOT}/nymph.json" "${STAGING_ROOT}/nymph.json"
install -m 644 "${MODULE_ROOT}/api_server.py" "${STAGING_ROOT}/api_server.py"
install -m 644 "${MODULE_ROOT}/config.py" "${STAGING_ROOT}/config.py"
install -m 644 "${MODULE_ROOT}/image_store.py" "${STAGING_ROOT}/image_store.py"
install -m 644 "${MODULE_ROOT}/model_manager.py" "${STAGING_ROOT}/model_manager.py"
install -m 644 "${MODULE_ROOT}/nunchaku_compat.py" "${STAGING_ROOT}/nunchaku_compat.py"
install -m 644 "${MODULE_ROOT}/progress_state.py" "${STAGING_ROOT}/progress_state.py"
install -m 644 "${MODULE_ROOT}/requirements.lock.txt" "${STAGING_ROOT}/requirements.lock.txt"
install -m 644 "${MODULE_ROOT}/schemas.py" "${STAGING_ROOT}/schemas.py"

for script in \
  _zimage_common.sh \
  prefetch_model.py \
  run_nunchaku_zimage_test.py \
  zimage_fetch_models.sh \
  zimage_logs.sh \
  zimage_open.sh \
  zimage_smoke_test.sh \
  zimage_start.sh \
  zimage_status.sh \
  zimage_stop.sh \
  zimage_uninstall.sh; do
  install -m 755 "${MODULE_ROOT}/scripts/${script}" "${STAGING_ROOT}/scripts/${script}"
done

if [[ -d "${MODULE_ROOT}/ui" ]]; then
  mkdir -p "${STAGING_ROOT}/ui"
  find "${MODULE_ROOT}/ui" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' ui_file; do
    install -m 644 "${ui_file}" "${STAGING_ROOT}/ui/$(basename "${ui_file}")"
  done
fi

STAGING_VENV_DIR="${STAGING_ROOT}/.venv-nunchaku"
echo "Creating staged runtime venv: ${STAGING_VENV_DIR}"
python3.11 -m venv "${STAGING_VENV_DIR}"

VENV_PYTHON="${STAGING_VENV_DIR}/bin/python"

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install torch==2.11.0 torchvision torchaudio --index-url "${ZIMAGE_TORCH_INDEX_URL}"
"${VENV_PYTHON}" -m pip install -r "${STAGING_ROOT}/requirements.lock.txt"
"${VENV_PYTHON}" -m pip install "${ZIMAGE_NUNCHAKU_SPEC}"
"${VENV_PYTHON}" -m pip install --force-reinstall "${ZIMAGE_DIFFUSERS_SPEC}"

(
  cd "${STAGING_ROOT}"
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

cat > "${STAGING_VENV_DIR}/.nymphs_nunchaku_runtime.json" <<EOF
{
  "nunchaku_repo": "${ZIMAGE_NUNCHAKU_REPO}",
  "nunchaku_commit": "${ZIMAGE_NUNCHAKU_COMMIT}",
  "diffusers": "${ZIMAGE_DIFFUSERS_SPEC}",
  "torch_index_url": "${ZIMAGE_TORCH_INDEX_URL}"
}
EOF

printf '%s\n' "${module_version}" > "${STAGING_ROOT}/.nymph-module-version"

if [[ -e "${ZIMAGE_INSTALL_ROOT}" && ! -d "${ZIMAGE_INSTALL_ROOT}" ]]; then
  echo "Install root exists but is not a directory: ${ZIMAGE_INSTALL_ROOT}" >&2
  exit 1
fi

"${SCRIPT_DIR}/zimage_stop.sh" || true

for preserve_name in outputs logs; do
  if [[ -e "${ZIMAGE_INSTALL_ROOT}/${preserve_name}" ]]; then
    rm -rf "${STAGING_ROOT:?}/${preserve_name}"
    mv "${ZIMAGE_INSTALL_ROOT}/${preserve_name}" "${STAGING_ROOT}/${preserve_name}"
  fi
done

if [[ -d "${ZIMAGE_INSTALL_ROOT}" ]]; then
  OLD_ROOT="${ZIMAGE_INSTALL_ROOT}.previous.$$"
  rm -rf "${OLD_ROOT}"
  mv "${ZIMAGE_INSTALL_ROOT}" "${OLD_ROOT}"
fi

mv "${STAGING_ROOT}" "${ZIMAGE_INSTALL_ROOT}"

if [[ -n "${OLD_ROOT}" ]]; then
  rm -rf "${OLD_ROOT}"
fi

trap - EXIT
echo "installed_module_version=${module_version}"
echo "Z-Image Turbo install complete."
