#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

mkdir -p "${NYMPHS3D_HF_CACHE_DIR}"
echo "directory=${NYMPHS3D_HF_CACHE_DIR}"
