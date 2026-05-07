#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

was_running=false
if zimage_is_running; then
  was_running=true
else
  "${SCRIPT_DIR}/zimage_start.sh"
fi

zimage_probe_url "${Z_IMAGE_SERVER_URL}/server_info"

if [[ "${was_running}" != "true" && "${ZIMAGE_SMOKE_KEEP_RUNNING:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/zimage_stop.sh"
fi
