#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ZIMAGE_INSTALL_ROOT="${ZIMAGE_INSTALL_ROOT:-$HOME/Z-Image}"
if [[ ! -f "${ZIMAGE_INSTALL_ROOT}/.nymph-module-version" ]]; then
  echo "Z-Image Turbo is not installed yet. Use Install first." >&2
  exit 2
fi

mkdir -p "${ZIMAGE_INSTALL_ROOT}/scripts"

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

find "${MODULE_ROOT}/scripts" -maxdepth 1 -type f \( -name '*.sh' -o -name '*.py' \) -print0 |
  while IFS= read -r -d '' script_file; do
    install -m 755 "${script_file}" "${ZIMAGE_INSTALL_ROOT}/scripts/$(basename "${script_file}")"
  done

if [[ -d "${MODULE_ROOT}/ui" ]]; then
  mkdir -p "${ZIMAGE_INSTALL_ROOT}/ui"
  find "${MODULE_ROOT}/ui" -maxdepth 1 -type f -print0 |
    while IFS= read -r -d '' ui_file; do
      install -m 644 "${ui_file}" "${ZIMAGE_INSTALL_ROOT}/ui/$(basename "${ui_file}")"
    done
fi

module_version="$(python3 - "${MODULE_ROOT}/nymph.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

print(str(manifest.get("version", "unknown")).strip() or "unknown")
PY
)"
printf '%s\n' "${module_version}" > "${ZIMAGE_INSTALL_ROOT}/.nymph-module-version"

echo "Z-Image Turbo module wrappers updated."
echo "installed_version=${module_version}"
