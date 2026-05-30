#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

scope=""
profile=""
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
    --profile)
      profile="${2:-}"
      shift 2
      ;;
    --profile=*)
      profile="${1#*=}"
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
Usage:
  zimage_delete_models.sh --scope base|weights|qwen|all --yes
  zimage_delete_models.sh --profile PROFILE_ID --yes

Deletes local Hugging Face cache folders used by Z-Image Turbo.

Scopes:
  base     Tongyi-MAI/Z-Image-Turbo cache
  weights  nunchaku-ai/nunchaku-z-image-turbo cache
  qwen     Qwen/Qwen-Image-Edit-2511 and QuantFunc Qwen Nunchaku caches
  all      base, Z-Image Nunchaku-compatible weights, and Qwen Edit caches

Profiles:
  zimage_int4_r32
  zimage_int4_r128
  zimage_int4_r256
  zimage_fp4_r32
  zimage_fp4_r128
  qwen_edit_2511_ultimate_speed_int4
  qwen_edit_2511_balance_int4
  qwen_edit_2511_best_quality_int4
  qwen_edit_2511_ultimate_speed_fp4
  qwen_edit_2511_balance_fp4
  qwen_edit_2511_best_quality_fp4

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

if [[ -n "${scope}" && -n "${profile}" ]]; then
  echo "Choose only one of --scope or --profile." >&2
  exit 2
fi

if [[ -z "${scope}" && -z "${profile}" ]]; then
  scope="all"
fi

if [[ -n "${scope}" ]]; then
  case "${scope}" in
    base|weights|qwen|all) ;;
    *)
      echo "Unsupported delete scope: ${scope}. Expected base, weights, qwen, or all." >&2
      exit 2
      ;;
  esac
fi

if [[ -n "${profile}" ]]; then
  case "${profile}" in
    zimage_int4_r32|zimage_int4_r128|zimage_int4_r256|zimage_fp4_r32|zimage_fp4_r128|\
qwen_edit_2511_ultimate_speed_int4|qwen_edit_2511_balance_int4|qwen_edit_2511_best_quality_int4|\
qwen_edit_2511_ultimate_speed_fp4|qwen_edit_2511_balance_fp4|qwen_edit_2511_best_quality_fp4) ;;
    *)
      echo "Unsupported delete profile: ${profile}." >&2
      exit 2
      ;;
  esac
fi

if [[ "${confirmed}" != "true" ]]; then
  echo "Refusing to delete model cache without --yes." >&2
  exit 2
fi

repo_cache_dir() {
  local repo_id="$1"
  local repo_path="${repo_id//\//--}"
  printf '%s/models--%s\n' "${NYMPHS3D_HF_CACHE_DIR}" "${repo_path}"
}

assert_repo_cache_path() {
  local path="$1"

  case "${path}" in
    "${NYMPHS3D_HF_CACHE_DIR}"/models--*) ;;
    *)
      echo "Refusing unsafe cache path: ${path}" >&2
      exit 3
      ;;
  esac
}

delete_cached_file() {
  local label="$1"
  local repo_id="$2"
  local filename="$3"
  local path
  path="$(repo_cache_dir "${repo_id}")"
  assert_repo_cache_path "${path}"

  if [[ ! -d "${path}" ]]; then
    echo "${label} cache already absent: ${path}"
    return
  fi

  local deleted=0
  while IFS= read -r -d '' file_path; do
    local blob_path=""
    if [[ -L "${file_path}" ]]; then
      blob_path="$(readlink -f "${file_path}" 2>/dev/null || true)"
    fi

    case "${file_path}" in
      "${path}"/snapshots/*/"${filename}") ;;
      *)
        echo "Refusing unsafe cached file path for ${label}: ${file_path}" >&2
        exit 3
        ;;
    esac

    echo "Deleting ${label}: ${file_path}"
    rm -f -- "${file_path}"
    deleted=1

    if [[ -n "${blob_path}" && "${blob_path}" == "${path}/blobs/"* && -f "${blob_path}" ]]; then
      local still_referenced=false
      while IFS= read -r -d '' sibling; do
        if [[ "$(readlink -f "${sibling}" 2>/dev/null || true)" == "${blob_path}" ]]; then
          still_referenced=true
          break
        fi
      done < <(find "${path}/snapshots" -type l -print0 2>/dev/null || true)

      if [[ "${still_referenced}" != "true" ]]; then
        echo "Deleting ${label} blob: ${blob_path}"
        rm -f -- "${blob_path}"
      fi
    fi
  done < <(find -L "${path}/snapshots" -mindepth 2 -maxdepth 2 -type f -name "${filename}" -print0 2>/dev/null || true)

  if [[ "${deleted}" -eq 0 ]]; then
    echo "${label} already absent: ${filename}"
  fi
}

delete_profile_cache() {
  local profile_id="$1"
  case "${profile_id}" in
    zimage_int4_r32)
      delete_cached_file "Z-Image Turbo INT4 r32 weight" "${Z_IMAGE_NUNCHAKU_MODEL_REPO}" "svdq-int4_r32-z-image-turbo.safetensors"
      ;;
    zimage_int4_r128)
      delete_cached_file "Z-Image Turbo INT4 r128 weight" "${Z_IMAGE_NUNCHAKU_MODEL_REPO}" "svdq-int4_r128-z-image-turbo.safetensors"
      ;;
    zimage_int4_r256)
      delete_cached_file "Z-Image Turbo INT4 r256 weight" "${Z_IMAGE_NUNCHAKU_MODEL_REPO}" "svdq-int4_r256-z-image-turbo.safetensors"
      ;;
    zimage_fp4_r32)
      delete_cached_file "Z-Image Turbo FP4 r32 weight" "${Z_IMAGE_NUNCHAKU_MODEL_REPO}" "svdq-fp4_r32-z-image-turbo.safetensors"
      ;;
    zimage_fp4_r128)
      delete_cached_file "Z-Image Turbo FP4 r128 weight" "${Z_IMAGE_NUNCHAKU_MODEL_REPO}" "svdq-fp4_r128-z-image-turbo.safetensors"
      ;;
    qwen_edit_2511_ultimate_speed_int4)
      delete_cached_file "Qwen Image Edit 2511 ultimate speed INT4 weight" "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_ultimate_speed_int4.safetensors"
      ;;
    qwen_edit_2511_balance_int4)
      delete_cached_file "Qwen Image Edit 2511 balance INT4 weight" "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_balance_int4.safetensors"
      ;;
    qwen_edit_2511_best_quality_int4)
      delete_cached_file "Qwen Image Edit 2511 best quality INT4 weight" "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_best_quality_int4.safetensors"
      ;;
    qwen_edit_2511_ultimate_speed_fp4)
      delete_cached_file "Qwen Image Edit 2511 ultimate speed FP4 weight" "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_ultimate_speed_fp4.safetensors"
      ;;
    qwen_edit_2511_balance_fp4)
      delete_cached_file "Qwen Image Edit 2511 balance FP4 weight" "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_balance_fp4.safetensors"
      ;;
    qwen_edit_2511_best_quality_fp4)
      delete_cached_file "Qwen Image Edit 2511 best quality FP4 weight" "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_best_quality_fp4.safetensors"
      ;;
    *)
      echo "Unsupported delete profile: ${profile_id}." >&2
      exit 2
      ;;
  esac
}

delete_repo_cache() {
  local label="$1"
  local repo_id="$2"
  local path
  path="$(repo_cache_dir "${repo_id}")"
  assert_repo_cache_path "${path}"

  if [[ -e "${path}" ]]; then
    echo "Deleting ${label} cache: ${path}"
    rm -rf -- "${path}"
  else
    echo "${label} cache already absent: ${path}"
  fi
}

echo "hf_cache_dir=${NYMPHS3D_HF_CACHE_DIR}"
mkdir -p "${NYMPHS3D_HF_CACHE_DIR}"

if [[ -n "${profile}" ]]; then
  echo "delete_profile=${profile}"
  delete_profile_cache "${profile}"
  echo "Z-Image Turbo model cache delete complete."
  exit 0
fi

if [[ "${scope}" == "base" || "${scope}" == "all" ]]; then
  delete_repo_cache "Z-Image Turbo base model" "${Z_IMAGE_MODEL_ID}"
fi

if [[ "${scope}" == "weights" || "${scope}" == "all" ]]; then
  delete_repo_cache "Z-Image Turbo Nunchaku-compatible weights" "${Z_IMAGE_NUNCHAKU_MODEL_REPO}"
fi

if [[ "${scope}" == "qwen" || "${scope}" == "all" ]]; then
  delete_repo_cache "Qwen Image Edit 2511 base model" "Qwen/Qwen-Image-Edit-2511"
  delete_repo_cache "Qwen Image Edit 2511 Nunchaku-compatible weights" "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511"
fi

echo "Z-Image Turbo model cache delete complete."
