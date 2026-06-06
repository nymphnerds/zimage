#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

if [[ ! -f "${ZIMAGE_INSTALL_ROOT}/.nymph-module-version" ]]; then
  echo "Z-Image Turbo is not installed yet. Use Install first." >&2
  exit 2
fi

runtime_marker="${ZIMAGE_VENV_DIR}/.nymphs_nunchaku_runtime.json"
if [[ ! -x "$(zimage_python)" ]] || [[ ! -f "${runtime_marker}" ]] || ! python3 - "${runtime_marker}" "${ZIMAGE_NUNCHAKU_COMMIT}" "${ZIMAGE_DIFFUSERS_SPEC}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

marker = Path(sys.argv[1])
expected_commit = sys.argv[2].strip()
expected_diffusers = sys.argv[3].strip()

try:
    data = json.loads(marker.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

if str(data.get("nunchaku_commit") or "").strip() != expected_commit:
    raise SystemExit(1)
if str(data.get("diffusers") or "").strip() != expected_diffusers:
    raise SystemExit(1)
PY
then
  echo "Z-Image runtime venv is missing or pinned to an older Nunchaku/diffusers build."
  echo "Rebuilding the runtime through the normal installer path..."
  exec "${SCRIPT_DIR}/install_zimage.sh"
fi

if ! "$(zimage_python)" - <<'PY'
from diffusers.pipelines.z_image.pipeline_z_image_controlnet import ZImageControlNetPipeline
from diffusers.models.controlnets.controlnet_z_image import ZImageControlNetModel

print(f"zimage_controlnet_pipeline={ZImageControlNetPipeline.__name__}")
print(f"zimage_controlnet_model={ZImageControlNetModel.__name__}")
PY
then
  echo "Z-Image runtime venv is missing Z-Image ControlNet support."
  echo "Rebuilding the runtime through the normal installer path..."
  exec "${SCRIPT_DIR}/install_zimage.sh"
fi

mkdir -p "${ZIMAGE_INSTALL_ROOT}/scripts"

install -m 644 "${MODULE_ROOT}/.env.example" "${ZIMAGE_INSTALL_ROOT}/.env.example"
install -m 644 "${MODULE_ROOT}/.gitignore" "${ZIMAGE_INSTALL_ROOT}/.gitignore"
install -m 644 "${MODULE_ROOT}/README.md" "${ZIMAGE_INSTALL_ROOT}/README.md"
install -m 644 "${MODULE_ROOT}/nymph.json" "${ZIMAGE_INSTALL_ROOT}/nymph.json"
install -m 644 "${MODULE_ROOT}/api_server.py" "${ZIMAGE_INSTALL_ROOT}/api_server.py"
install -m 644 "${MODULE_ROOT}/nymph_image.html" "${ZIMAGE_INSTALL_ROOT}/nymph_image.html"
install -m 644 "${MODULE_ROOT}/shared_image_parts.py" "${ZIMAGE_INSTALL_ROOT}/shared_image_parts.py"
install -m 644 "${MODULE_ROOT}/brain_client.py" "${ZIMAGE_INSTALL_ROOT}/brain_client.py"
install -m 644 "${MODULE_ROOT}/config.py" "${ZIMAGE_INSTALL_ROOT}/config.py"
install -m 644 "${MODULE_ROOT}/image_providers.py" "${ZIMAGE_INSTALL_ROOT}/image_providers.py"
install -m 644 "${MODULE_ROOT}/image_store.py" "${ZIMAGE_INSTALL_ROOT}/image_store.py"
install -m 644 "${MODULE_ROOT}/model_manager.py" "${ZIMAGE_INSTALL_ROOT}/model_manager.py"
install -m 644 "${MODULE_ROOT}/nunchaku_compat.py" "${ZIMAGE_INSTALL_ROOT}/nunchaku_compat.py"
install -m 644 "${MODULE_ROOT}/progress_state.py" "${ZIMAGE_INSTALL_ROOT}/progress_state.py"
install -m 644 "${MODULE_ROOT}/requirements.lock.txt" "${ZIMAGE_INSTALL_ROOT}/requirements.lock.txt"
install -m 644 "${MODULE_ROOT}/schemas.py" "${ZIMAGE_INSTALL_ROOT}/schemas.py"

if [[ -d "${MODULE_ROOT}/prompt_presets" ]]; then
  mkdir -p "${ZIMAGE_INSTALL_ROOT}/prompt_presets"
  find "${MODULE_ROOT}/prompt_presets" -maxdepth 1 -type f -name '*.json' -print0 |
    while IFS= read -r -d '' preset_file; do
      install -m 644 "${preset_file}" "${ZIMAGE_INSTALL_ROOT}/prompt_presets/$(basename "${preset_file}")"
    done
fi

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

if [[ -x "${ZIMAGE_INSTALL_ROOT}/scripts/zimage_seed_image_presets.sh" ]]; then
  ZIMAGE_SEED_IMAGE_PRESETS_IF_EMPTY=1 "${ZIMAGE_INSTALL_ROOT}/scripts/zimage_seed_image_presets.sh" >/dev/null || true
fi

echo "Z-Image Turbo module wrappers updated."
echo "installed_version=${module_version}"
