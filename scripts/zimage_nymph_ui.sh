#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  mkdir -p "$(dirname "${ZIMAGE_OPENROUTER_ENV_FILE}")"
  {
    printf '%s\n' "# Z-Image OpenRouter configuration"
    printf 'OPENROUTER_API_KEY=%s\n' "${OPENROUTER_API_KEY}"
  } > "${ZIMAGE_OPENROUTER_ENV_FILE}"
  chmod 600 "${ZIMAGE_OPENROUTER_ENV_FILE}"
  echo "OpenRouter key saved for Nymphs Image."
fi

"${SCRIPT_DIR}/zimage_start.sh"

echo "url=${Z_IMAGE_SERVER_URL}/nymph"
echo "module_ui_url=${Z_IMAGE_SERVER_URL}/nymph"
