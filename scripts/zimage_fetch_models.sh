#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

requested_gpu_family=""
requested_preset=""
selected_fetch_label=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-family)
      if [[ $# -lt 2 ]]; then
        echo "--gpu-family requires a value." >&2
        exit 2
      fi
      requested_gpu_family="${2:-}"
      shift 2
      ;;
    --gpu-family=*)
      requested_gpu_family="${1#*=}"
      shift
      ;;
    --preset)
      if [[ $# -lt 2 ]]; then
        echo "--preset requires a value." >&2
        exit 2
      fi
      requested_preset="${2:-}"
      shift 2
      ;;
    --preset=*)
      requested_preset="${1#*=}"
      shift
      ;;
    --precision)
      if [[ $# -lt 2 ]]; then
        echo "--precision requires a value." >&2
        exit 2
      fi
      Z_IMAGE_NUNCHAKU_PRECISION="${2:-}"
      export Z_IMAGE_NUNCHAKU_PRECISION
      shift 2
      ;;
    --precision=*)
      Z_IMAGE_NUNCHAKU_PRECISION="${1#*=}"
      export Z_IMAGE_NUNCHAKU_PRECISION
      shift
      ;;
    --rank)
      if [[ $# -lt 2 ]]; then
        echo "--rank requires a value." >&2
        exit 2
      fi
      Z_IMAGE_NUNCHAKU_RANK="${2:-}"
      export Z_IMAGE_NUNCHAKU_RANK
      shift 2
      ;;
    --rank=*)
      Z_IMAGE_NUNCHAKU_RANK="${1#*=}"
      export Z_IMAGE_NUNCHAKU_RANK
      shift
      ;;
    --hf_token|--hf-token)
      if [[ $# -lt 2 || "${2:-}" == --* ]]; then
        export NYMPHS3D_HF_TOKEN=""
        shift 1
      else
        export NYMPHS3D_HF_TOKEN="${2:-}"
        shift 2
      fi
      ;;
    --hf_token=*|--hf-token=*)
      export NYMPHS3D_HF_TOKEN="${1#*=}"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  zimage_fetch_models.sh --gpu-family rtx_20_30_40|rtx_50 --preset fast|balanced|highest
  zimage_fetch_models.sh [--precision auto|int4|fp4] [--rank 32|128|256] [--hf_token TOKEN]

Downloads the base Z-Image Turbo model files and the selected Nunchaku
quantized weight.

Friendly presets:
  RTX 20/30/40 + fast     -> int4 r32
  RTX 20/30/40 + balanced -> int4 r128
  RTX 20/30/40 + highest  -> int4 r256
  RTX 50       + fast     -> fp4 r32
  RTX 50       + balanced -> fp4 r128

Published Nunchaku Z-Image Turbo weights:
  int4 r32
  int4 r128
  int4 r256
  fp4  r32
  fp4  r128

Use --precision auto only with ranks that exist for both INT4 and FP4.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${requested_gpu_family}" || -n "${requested_preset}" ]]; then
  if [[ -z "${requested_gpu_family}" || -z "${requested_preset}" ]]; then
    echo "--gpu-family and --preset must be used together." >&2
    exit 2
  fi

  case "${requested_gpu_family}:${requested_preset}" in
    rtx_20_30_40:fast)
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="RTX 20/30/40 Fast"
      ;;
    rtx_20_30_40:balanced)
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="128"
      selected_fetch_label="RTX 20/30/40 Balanced"
      ;;
    rtx_20_30_40:highest)
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="256"
      selected_fetch_label="RTX 20/30/40 Highest"
      ;;
    rtx_50:fast)
      Z_IMAGE_NUNCHAKU_PRECISION="fp4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="RTX 50 Fast"
      ;;
    rtx_50:balanced)
      Z_IMAGE_NUNCHAKU_PRECISION="fp4"
      Z_IMAGE_NUNCHAKU_RANK="128"
      selected_fetch_label="RTX 50 Balanced"
      ;;
    rtx_50:highest)
      echo "Unsupported preset: RTX 50 Highest would require fp4 r256, but the published r256 Z-Image Turbo Nunchaku weight is INT4 only." >&2
      exit 2
      ;;
    *)
      echo "Unsupported Z-Image fetch preset: gpu=${requested_gpu_family} preset=${requested_preset}." >&2
      echo "Expected GPU rtx_20_30_40 or rtx_50, and preset fast, balanced, or highest." >&2
      exit 2
      ;;
  esac

  export Z_IMAGE_NUNCHAKU_PRECISION
  export Z_IMAGE_NUNCHAKU_RANK
fi

case "${Z_IMAGE_NUNCHAKU_PRECISION}" in
  auto|int4|fp4) ;;
  *)
    echo "Unsupported precision: ${Z_IMAGE_NUNCHAKU_PRECISION}. Expected auto, int4, or fp4." >&2
    exit 2
    ;;
esac

if [[ ! "${Z_IMAGE_NUNCHAKU_RANK}" =~ ^[0-9]+$ ]]; then
  echo "Unsupported rank: ${Z_IMAGE_NUNCHAKU_RANK}. Expected a numeric rank such as 32." >&2
  exit 2
fi

case "${Z_IMAGE_NUNCHAKU_PRECISION}:${Z_IMAGE_NUNCHAKU_RANK}" in
  auto:32|auto:128|int4:32|int4:128|int4:256|fp4:32|fp4:128) ;;
  auto:256)
    echo "Unsupported weight: auto r256. The published r256 Nunchaku Z-Image Turbo weight is INT4 only." >&2
    exit 2
    ;;
  fp4:256)
    echo "Unsupported weight: fp4 r256. The published r256 Nunchaku Z-Image Turbo weight is INT4 only." >&2
    exit 2
    ;;
  *)
    echo "Unsupported weight: ${Z_IMAGE_NUNCHAKU_PRECISION} r${Z_IMAGE_NUNCHAKU_RANK}." >&2
    echo "Expected one of: int4 r32, int4 r128, int4 r256, fp4 r32, fp4 r128, auto r32, auto r128." >&2
    exit 2
    ;;
esac

if [[ ! -x "$(zimage_python)" ]]; then
  echo "Z-Image Turbo runtime is missing. Run scripts/install_zimage.sh first." >&2
  exit 1
fi

cache_size_bytes() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo 0
    return
  fi
  find "${path}" -type f -printf '%s\n' 2>/dev/null | awk '{ total += $1 } END { printf "%.0f\n", total }'
}

format_bytes() {
  local size_bytes="${1:-0}"
  awk -v bytes="${size_bytes}" 'BEGIN {
    split("B KiB MiB GiB TiB", units, " ");
    value = bytes + 0;
    unit_index = 1;
    while (value >= 1024 && unit_index < 5) {
      value = value / 1024;
      unit_index++;
    }
    if (unit_index == 1) {
      printf "%d %s", value, units[unit_index];
    } else {
      printf "%.2f %s", value, units[unit_index];
    }
  }'
}

hf_repo_cache_dir() {
  local repo_id="$1"
  local repo_path="${repo_id//\//--}"
  echo "${NYMPHS3D_HF_CACHE_DIR}/models--${repo_path}"
}

hf_repo_blob_bytes() {
  local repo_id="$1"
  local repo_dir
  repo_dir="$(hf_repo_cache_dir "${repo_id}")"
  cache_size_bytes "${repo_dir}/blobs"
}

hf_repo_incomplete_count() {
  local repo_id="$1"
  local repo_dir
  repo_dir="$(hf_repo_cache_dir "${repo_id}")"
  if [[ ! -d "${repo_dir}/blobs" ]]; then
    echo 0
    return
  fi
  find "${repo_dir}/blobs" -type f -name '*.incomplete' 2>/dev/null | wc -l | tr -d ' '
}

print_hf_download_progress() {
  local label="$1"
  local repo_id="$2"
  local start_cache_bytes="$3"
  local current_cache_bytes
  local repo_bytes
  local incomplete_count
  local cache_delta

  current_cache_bytes="$(cache_size_bytes "${NYMPHS3D_HF_CACHE_DIR}")"
  repo_bytes="$(hf_repo_blob_bytes "${repo_id}")"
  incomplete_count="$(hf_repo_incomplete_count "${repo_id}")"
  cache_delta=$(( current_cache_bytes - start_cache_bytes ))
  if [[ "${cache_delta}" -lt 0 ]]; then
    cache_delta=0
  fi

  echo "MODEL DOWNLOAD STATUS: phase=${label} repo=${repo_id} status=downloading"
  if [[ -n "${NYMPHS3D_PREFETCH_COMPONENT_HINT:-}" ]]; then
    echo "MODEL DOWNLOAD STATUS: waiting_on=${NYMPHS3D_PREFETCH_COMPONENT_HINT}"
  fi
  echo "MODEL DOWNLOAD STATUS: cache_dir=${NYMPHS3D_HF_CACHE_DIR} shared_cache=$(format_bytes "${current_cache_bytes}") downloaded_this_step=$(format_bytes "${cache_delta}")"
  echo "MODEL DOWNLOAD STATUS: repo_cache_blobs=$(format_bytes "${repo_bytes}") active_partial_files=${incomplete_count}"
}

run_with_hf_download_progress() {
  local label="$1"
  local repo_id="$2"
  shift 2

  local interval="${NYMPHS3D_PREFETCH_PROGRESS_INTERVAL:-5}"
  local start_cache_bytes
  local marker
  local pid
  local status

  if [[ ! "${interval}" =~ ^[0-9]+$ || "${interval}" -lt 1 ]]; then
    interval=5
  fi

  start_cache_bytes="$(cache_size_bytes "${NYMPHS3D_HF_CACHE_DIR}")"
  marker="$(mktemp "${TMPDIR:-/tmp}/nymphs-zimage-prefetch.XXXXXX.status")"
  rm -f "${marker}"

  echo "MODEL DOWNLOAD STARTED: phase=${label} repo=${repo_id} cache_dir=${NYMPHS3D_HF_CACHE_DIR} progress_interval=${interval}s"
  (
    set +e
    "$@"
    status=$?
    printf '%s\n' "${status}" > "${marker}"
    exit "${status}"
  ) &
  pid=$!

  while [[ ! -f "${marker}" ]]; do
    sleep "${interval}"
    if [[ -f "${marker}" ]]; then
      break
    fi
    print_hf_download_progress "${label}" "${repo_id}" "${start_cache_bytes}"
  done

  wait "${pid}" || true
  status="$(cat "${marker}" 2>/dev/null || echo 1)"
  rm -f "${marker}"

  if [[ "${status}" -eq 0 ]]; then
    print_hf_download_progress "${label}" "${repo_id}" "${start_cache_bytes}"
    echo "MODEL DOWNLOAD COMPLETE: phase=${label} repo=${repo_id}"
  else
    echo "MODEL DOWNLOAD FAILED: phase=${label} repo=${repo_id} exit_status=${status}"
  fi

  return "${status}"
}

prefetch_zimage_base_model() {
  (
    cd "${ZIMAGE_INSTALL_ROOT}"
    export Z_IMAGE_RUNTIME=standard
    export NYMPHS2D2_RUNTIME=standard
    "$(zimage_python)" scripts/prefetch_model.py
  )
}

prefetch_zimage_nunchaku_weight() {
  (
    cd "${ZIMAGE_INSTALL_ROOT}"
    "$(zimage_python)" - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = os.getenv("Z_IMAGE_NUNCHAKU_MODEL_REPO") or "nunchaku-ai/nunchaku-z-image-turbo"
rank = os.getenv("Z_IMAGE_NUNCHAKU_RANK") or "32"
precision = (os.getenv("Z_IMAGE_NUNCHAKU_PRECISION") or "auto").strip().lower()
cache_dir = os.getenv("NYMPHS3D_HF_CACHE_DIR") or None
token = os.getenv("NYMPHS3D_HF_TOKEN") or None
if precision == "auto":
    try:
        from nunchaku.utils import get_precision
        device = os.getenv("Z_IMAGE_DEVICE") or "cuda"
        precision = get_precision(precision="auto", device=device)
    except Exception:
        precision = "int4"

filename = f"svdq-{precision}_r{rank}-z-image-turbo.safetensors"
print(f"Z-Image Turbo Nunchaku weight prefetch: {repo_id}/{filename}", flush=True)
path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, token=token)
print(f"Z-Image Turbo Nunchaku weight ready: {path}", flush=True)
PY
  )
}

save_zimage_generation_preset() {
  mkdir -p "${ZIMAGE_CONFIG_DIR}"
  {
    printf 'Z_IMAGE_NUNCHAKU_PRECISION=%s\n' "${Z_IMAGE_NUNCHAKU_PRECISION}"
    printf 'Z_IMAGE_NUNCHAKU_RANK=%s\n' "${Z_IMAGE_NUNCHAKU_RANK}"
    if [[ -n "${selected_fetch_label}" ]]; then
      printf 'ZIMAGE_FETCH_LABEL=%s\n' "${selected_fetch_label}"
    fi
  } > "${ZIMAGE_PRESET_FILE}"
}

export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
mkdir -p "${NYMPHS3D_HF_CACHE_DIR}"

echo "zimage_model=${Z_IMAGE_MODEL_ID}"
echo "nunchaku_weight_repo=${Z_IMAGE_NUNCHAKU_MODEL_REPO}"
echo "nunchaku_precision=${Z_IMAGE_NUNCHAKU_PRECISION}"
echo "nunchaku_rank=${Z_IMAGE_NUNCHAKU_RANK}"
echo "hf_cache_dir=${NYMPHS3D_HF_CACHE_DIR}"

export NYMPHS3D_PREFETCH_COMPONENT_HINT="base Z-Image files: scheduler, text encoder, tokenizer, transformer, and VAE"
run_with_hf_download_progress \
  "Z-Image Turbo model prefetch" \
  "${Z_IMAGE_MODEL_ID}" \
  prefetch_zimage_base_model
unset NYMPHS3D_PREFETCH_COMPONENT_HINT

export NYMPHS3D_PREFETCH_COMPONENT_HINT="Nunchaku rank weight for the selected Z-Image preset"
run_with_hf_download_progress \
  "Z-Image Turbo Nunchaku weight prefetch" \
  "${Z_IMAGE_NUNCHAKU_MODEL_REPO}" \
  prefetch_zimage_nunchaku_weight
unset NYMPHS3D_PREFETCH_COMPONENT_HINT

save_zimage_generation_preset
echo "Z-Image generation preset saved: precision=${Z_IMAGE_NUNCHAKU_PRECISION} rank=${Z_IMAGE_NUNCHAKU_RANK} file=${ZIMAGE_PRESET_FILE}"
