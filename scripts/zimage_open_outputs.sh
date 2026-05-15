#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

mkdir -p "${ZIMAGE_OUTPUT_DIR}"
echo "directory=${ZIMAGE_OUTPUT_DIR}"
