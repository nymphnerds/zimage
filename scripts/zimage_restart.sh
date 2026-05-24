#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/zimage_stop.sh" || true
"${SCRIPT_DIR}/zimage_start.sh"
