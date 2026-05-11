#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

PURGE=0
DRY_RUN=0
YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --purge) PURGE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes) YES=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: zimage_uninstall.sh [--dry-run] [--yes] [--purge]

Default uninstall removes the Z-Image runtime install but preserves outputs and logs.
--purge removes the whole install root, including outputs and logs.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

echo "Z-Image Turbo uninstall plan"
echo "install_root=${ZIMAGE_INSTALL_ROOT}"
if [[ "${PURGE}" -eq 1 ]]; then
  echo "mode=purge"
  echo "delete=${ZIMAGE_INSTALL_ROOT}"
else
  echo "mode=uninstall"
  echo "delete=runtime files, source files, venvs inside ${ZIMAGE_INSTALL_ROOT}"
  echo "preserve=${ZIMAGE_OUTPUT_DIR}"
  echo "preserve=${ZIMAGE_LOG_DIR}"
  echo "preserve_legacy=${ZIMAGE_INSTALL_ROOT}/outputs"
  echo "preserve_legacy=${ZIMAGE_INSTALL_ROOT}/logs"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

if [[ "${YES}" -ne 1 ]]; then
  echo "Refusing to delete without --yes. Run with --dry-run first to preview." >&2
  exit 2
fi

"${SCRIPT_DIR}/zimage_stop.sh" || true

if [[ ! -d "${ZIMAGE_INSTALL_ROOT}" ]]; then
  echo "Z-Image Turbo is already uninstalled."
  exit 0
fi

if [[ "${PURGE}" -eq 1 ]]; then
  rm -rf "${ZIMAGE_INSTALL_ROOT}"
else
  rm -f "${ZIMAGE_INSTALL_ROOT}/.nymph-module-version"
  find "${ZIMAGE_INSTALL_ROOT}" -mindepth 1 \
    \( -name outputs -o -name logs \) -prune -o \
    -exec rm -rf {} +
fi

echo "Z-Image Turbo uninstalled."
