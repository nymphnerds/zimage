#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

if zimage_is_running; then
  echo "Z-Image Turbo is already running at ${Z_IMAGE_SERVER_URL}"
  exit 0
fi

if [[ ! -x "$(zimage_python)" ]]; then
  echo "Z-Image Turbo runtime is missing. Run scripts/install_zimage.sh first." >&2
  exit 1
fi

if [[ ! -f "${ZIMAGE_INSTALL_ROOT}/api_server.py" ]]; then
  echo "Z-Image Turbo source is missing from ${ZIMAGE_INSTALL_ROOT}. Run scripts/install_zimage.sh first." >&2
  exit 1
fi

log_file="${ZIMAGE_LOG_DIR}/zimage-server.log"
echo "Starting Z-Image Turbo at ${Z_IMAGE_SERVER_URL}"
(
  cd "${ZIMAGE_INSTALL_ROOT}"
  nohup "$(zimage_python)" -u api_server.py --host "${Z_IMAGE_HOST}" --port "${Z_IMAGE_PORT}" >"${log_file}" 2>&1 &
  echo $! > "${ZIMAGE_PID_FILE}"
)

for _ in $(seq 1 40); do
  if zimage_probe_url "${Z_IMAGE_SERVER_URL}/server_info" >/dev/null 2>&1; then
    echo "Z-Image Turbo started."
    exit 0
  fi
  sleep 1
done

echo "Z-Image Turbo did not answer before timeout. Check ${log_file}" >&2
exit 1
