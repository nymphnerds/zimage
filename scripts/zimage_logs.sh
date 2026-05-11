#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

echo "logs_dir=${ZIMAGE_LOG_DIR}"
echo "last_log=${ZIMAGE_LOG_DIR}/zimage-server.log"
if [[ -f "${ZIMAGE_LOG_DIR}/zimage-server.log" ]]; then
  tail -n "${ZIMAGE_LOG_LINES:-160}" "${ZIMAGE_LOG_DIR}/zimage-server.log"
else
  echo "No Z-Image Turbo server log yet."
fi
