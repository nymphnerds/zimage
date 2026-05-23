#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

NYMPHS_CONFIG_ROOT="${NYMPHS_CONFIG_ROOT:-$NYMPHS_DATA_ROOT/config}"
TARGET_DIR="${NYMPHS_IMAGE_SETTINGS_PRESET_DIR:-${NYMPHS_IMAGE_SETTINGS_PRESETS_DIR:-$NYMPHS_CONFIG_ROOT/image_settings_presets}}"
mkdir -p "${TARGET_DIR}"
echo "directory=${TARGET_DIR}"
