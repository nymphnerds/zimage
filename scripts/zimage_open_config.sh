#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

NYMPHS_CONFIG_ROOT="${NYMPHS_CONFIG_ROOT:-$NYMPHS_DATA_ROOT/config}"
mkdir -p "${NYMPHS_CONFIG_ROOT}/image_presets" "${NYMPHS_CONFIG_ROOT}/image_settings_presets" "${ZIMAGE_CONFIG_DIR}"
echo "directory=${NYMPHS_CONFIG_ROOT}"
