#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

scope="all"
confirmed=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      scope="${2:-}"
      shift 2
      ;;
    --scope=*)
      scope="${1#*=}"
      shift
      ;;
    --yes)
      confirmed=true
      if [[ "${2:-}" =~ ^(1|true|yes)$ ]]; then
        shift 2
      else
        shift
      fi
      ;;
    --yes=*)
      case "${1#*=}" in
        1|true|yes) confirmed=true ;;
        *) confirmed=false ;;
      esac
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: zimage_delete_models.sh --scope base|weights|all --yes

Deletes local Hugging Face cache folders used by Z-Image Turbo.

Scopes:
  base     Tongyi-MAI/Z-Image-Turbo cache
  weights  nunchaku-ai/nunchaku-z-image-turbo cache
  all      both base and Nunchaku-compatible weight caches

This does not delete the runtime install, generated outputs, logs, LoRAs, or
other modules' model caches.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

case "${scope}" in
  base|weights|all) ;;
  *)
    echo "Unsupported delete scope: ${scope}. Expected base, weights, or all." >&2
    exit 2
    ;;
esac

if [[ "${confirmed}" != "true" ]]; then
  echo "Refusing to delete model cache without --yes." >&2
  exit 2
fi

repo_cache_dir() {
  local repo_id="$1"
  local repo_path="${repo_id//\//--}"
  printf '%s/models--%s\n' "${NYMPHS3D_HF_CACHE_DIR}" "${repo_path}"
}

delete_repo_cache() {
  local label="$1"
  local repo_id="$2"
  local path
  path="$(repo_cache_dir "${repo_id}")"

  case "${path}" in
    "${NYMPHS3D_HF_CACHE_DIR}"/models--*) ;;
    *)
      echo "Refusing unsafe cache path for ${label}: ${path}" >&2
      exit 3
      ;;
  esac

  if [[ -e "${path}" ]]; then
    echo "Deleting ${label} cache: ${path}"
    rm -rf -- "${path}"
  else
    echo "${label} cache already absent: ${path}"
  fi
}

echo "hf_cache_dir=${NYMPHS3D_HF_CACHE_DIR}"
mkdir -p "${NYMPHS3D_HF_CACHE_DIR}"

if [[ "${scope}" == "base" || "${scope}" == "all" ]]; then
  delete_repo_cache "Z-Image Turbo base model" "${Z_IMAGE_MODEL_ID}"
fi

if [[ "${scope}" == "weights" || "${scope}" == "all" ]]; then
  delete_repo_cache "Z-Image Turbo Nunchaku-compatible weights" "${Z_IMAGE_NUNCHAKU_MODEL_REPO}"
fi

echo "Z-Image Turbo model cache delete complete."
