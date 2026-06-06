#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

requested_gpu_family=""
requested_preset=""
requested_weight=""
download_all_weights=false
download_int4_package=false
download_fp4_package=false
selected_fetch_label=""
fetch_zimage=true
fetch_qwen_edit=false
QWEN_EDIT_MODEL_ID="Qwen/Qwen-Image-Edit-2511"
QWEN_EDIT_WEIGHT_REPO="QuantFunc/Nunchaku-Qwen-Image-EDIT-2511"
QWEN_EDIT_WEIGHT_FILE=""
QWEN_EDIT_WEIGHT_FILES=()
license_ack=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--weight)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a value." >&2
        exit 2
      fi
      requested_weight="${2:-}"
      shift 2
      ;;
    --model=*|--weight=*)
      requested_weight="${1#*=}"
      shift
      ;;
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
    --license-ack|--license_ack)
      if [[ $# -ge 2 && "${2:-}" != --* ]]; then
        case "${2:-}" in
          1|true|yes|y|on|acknowledged) license_ack=true ;;
          *) license_ack=false ;;
        esac
        shift 2
      else
        license_ack=true
        shift 1
      fi
      ;;
    --license-ack=*|--license_ack=*)
      case "${1#*=}" in
        1|true|yes|y|on|acknowledged) license_ack=true ;;
        *) license_ack=false ;;
      esac
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  zimage_fetch_models.sh --model svdq-int4_r128-z-image-turbo.safetensors
  zimage_fetch_models.sh --model complete_int4_package
  zimage_fetch_models.sh --model complete_fp4_package
  zimage_fetch_models.sh --model local_image_stack_qwen_2511_int4
  zimage_fetch_models.sh --model qwen_edit_2511_balance_int4
  zimage_fetch_models.sh --gpu-family rtx_20_30_40|rtx_50 --preset fast|balanced|highest
  zimage_fetch_models.sh [--precision auto|int4|fp4] [--rank 32|128|256] [--hf_token TOKEN]

Downloads local Nymphs Image model files. Z-Image Turbo is the fast default.
Qwen Image Edit 2511 is local image edit, reference edit, and parts extraction.
Brain owns Qwen vision/planner model downloads.

Friendly presets:
  RTX 20/30/40 + fast     -> int4 r32
  RTX 20/30/40 + balanced -> int4 r128
  RTX 20/30/40 + highest  -> int4 r256
  RTX 50       + fast     -> fp4 r32
  RTX 50       + balanced -> fp4 r128

Published Z-Image Turbo quantized weights for the Nunchaku runtime:
  svdq-int4_r32-z-image-turbo.safetensors
  svdq-int4_r128-z-image-turbo.safetensors
  svdq-int4_r256-z-image-turbo.safetensors
  svdq-fp4_r32-z-image-turbo.safetensors
  svdq-fp4_r128-z-image-turbo.safetensors

Published Qwen Image Edit 2511 Nunchaku weights:
  ultimate_speed INT4/FP4
  balanced INT4/FP4
  best_quality INT4/FP4

Use --precision auto only with ranks that exist for both INT4 and FP4.

Qwen vision/VLM models are fetched and configured in Brain, not Nymphs Image.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${requested_weight}" && ( -n "${requested_gpu_family}" || -n "${requested_preset}" ) ]]; then
  echo "--model/--weight cannot be combined with --gpu-family/--preset." >&2
  exit 2
fi

if [[ -n "${requested_weight}" ]]; then
  case "${requested_weight}" in
    all|all_weights|all-weights)
      download_all_weights=true
      selected_fetch_label="all published Z-Image Turbo Nunchaku-compatible weights"
      ;;
    complete_int4_package|complete-int4-package|int4_package|int4-package)
      download_int4_package=true
      fetch_zimage=true
      fetch_qwen_edit=true
      QWEN_EDIT_WEIGHT_FILES=(
        "nunchaku_qwen_image_edit_2511_ultimate_speed_int4.safetensors"
        "nunchaku_qwen_image_edit_2511_balance_int4.safetensors"
        "nunchaku_qwen_image_edit_2511_best_quality_int4.safetensors"
      )
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="Complete INT4 package: every INT4 Z-Image Turbo and Qwen Image Edit 2511 weight"
      ;;
    complete_fp4_package|complete-fp4-package|fp4_package|fp4-package)
      download_fp4_package=true
      fetch_zimage=true
      fetch_qwen_edit=true
      QWEN_EDIT_WEIGHT_FILES=(
        "nunchaku_qwen_image_edit_2511_ultimate_speed_fp4.safetensors"
        "nunchaku_qwen_image_edit_2511_balance_fp4.safetensors"
        "nunchaku_qwen_image_edit_2511_best_quality_fp4.safetensors"
      )
      Z_IMAGE_NUNCHAKU_PRECISION="fp4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="Complete FP4 package: every FP4 Z-Image Turbo and Qwen Image Edit 2511 weight"
      ;;
    local_image_stack_qwen_2511_int4|local-image-stack-qwen-2511-int4|all_models|all-models|all_image_models|all-image-models)
      fetch_zimage=true
      fetch_qwen_edit=true
      QWEN_EDIT_WEIGHT_FILE="nunchaku_qwen_image_edit_2511_balance_int4.safetensors"
      QWEN_EDIT_WEIGHT_FILES=("${QWEN_EDIT_WEIGHT_FILE}")
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="Local Image Stack: Z-Image Turbo INT4 r32 plus Qwen Image Edit 2511 balanced INT4"
      ;;
    svdq-int4_r32-z-image-turbo.safetensors|int4_r32|int4:32)
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="svdq-int4_r32-z-image-turbo.safetensors"
      ;;
    svdq-int4_r128-z-image-turbo.safetensors|int4_r128|int4:128)
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="128"
      selected_fetch_label="svdq-int4_r128-z-image-turbo.safetensors"
      ;;
    svdq-int4_r256-z-image-turbo.safetensors|int4_r256|int4:256)
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="256"
      selected_fetch_label="svdq-int4_r256-z-image-turbo.safetensors"
      ;;
    svdq-fp4_r32-z-image-turbo.safetensors|fp4_r32|fp4:32)
      Z_IMAGE_NUNCHAKU_PRECISION="fp4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="svdq-fp4_r32-z-image-turbo.safetensors"
      ;;
    svdq-fp4_r128-z-image-turbo.safetensors|fp4_r128|fp4:128)
      Z_IMAGE_NUNCHAKU_PRECISION="fp4"
      Z_IMAGE_NUNCHAKU_RANK="128"
      selected_fetch_label="svdq-fp4_r128-z-image-turbo.safetensors"
      ;;
    qwen_edit_2511_ultimate_speed_int4|qwen-edit-2511-ultimate-speed-int4)
      fetch_zimage=false
      fetch_qwen_edit=true
      QWEN_EDIT_WEIGHT_FILE="nunchaku_qwen_image_edit_2511_ultimate_speed_int4.safetensors"
      QWEN_EDIT_WEIGHT_FILES=("${QWEN_EDIT_WEIGHT_FILE}")
      selected_fetch_label="Qwen Image Edit 2511 ultimate speed INT4"
      ;;
    qwen_edit_2511_balance_int4|qwen-edit-2511-balance-int4)
      fetch_zimage=false
      fetch_qwen_edit=true
      QWEN_EDIT_WEIGHT_FILE="nunchaku_qwen_image_edit_2511_balance_int4.safetensors"
      QWEN_EDIT_WEIGHT_FILES=("${QWEN_EDIT_WEIGHT_FILE}")
      selected_fetch_label="Qwen Image Edit 2511 balanced INT4"
      ;;
    qwen_edit_2511_best_quality_int4|qwen-edit-2511-best-quality-int4)
      fetch_zimage=false
      fetch_qwen_edit=true
      QWEN_EDIT_WEIGHT_FILE="nunchaku_qwen_image_edit_2511_best_quality_int4.safetensors"
      QWEN_EDIT_WEIGHT_FILES=("${QWEN_EDIT_WEIGHT_FILE}")
      selected_fetch_label="Qwen Image Edit 2511 best quality INT4"
      ;;
    qwen_edit_2511_ultimate_speed_fp4|qwen-edit-2511-ultimate-speed-fp4)
      fetch_zimage=false
      fetch_qwen_edit=true
      QWEN_EDIT_WEIGHT_FILE="nunchaku_qwen_image_edit_2511_ultimate_speed_fp4.safetensors"
      QWEN_EDIT_WEIGHT_FILES=("${QWEN_EDIT_WEIGHT_FILE}")
      selected_fetch_label="Qwen Image Edit 2511 ultimate speed FP4"
      ;;
    qwen_edit_2511_balance_fp4|qwen-edit-2511-balance-fp4)
      fetch_zimage=false
      fetch_qwen_edit=true
      QWEN_EDIT_WEIGHT_FILE="nunchaku_qwen_image_edit_2511_balance_fp4.safetensors"
      QWEN_EDIT_WEIGHT_FILES=("${QWEN_EDIT_WEIGHT_FILE}")
      selected_fetch_label="Qwen Image Edit 2511 balanced FP4"
      ;;
    qwen_edit_2511_best_quality_fp4|qwen-edit-2511-best-quality-fp4)
      fetch_zimage=false
      fetch_qwen_edit=true
      QWEN_EDIT_WEIGHT_FILE="nunchaku_qwen_image_edit_2511_best_quality_fp4.safetensors"
      QWEN_EDIT_WEIGHT_FILES=("${QWEN_EDIT_WEIGHT_FILE}")
      selected_fetch_label="Qwen Image Edit 2511 best quality FP4"
      ;;
    qwen3_vl_8b_q4_vision|qwen3-vl-8b-q4-vision)
      echo "Brain owns local vision model downloads. Use Brain to fetch/configure Qwen or another vision-capable model." >&2
      exit 2
      ;;
    *)
      echo "Unsupported Nymphs Image model selection: ${requested_weight}." >&2
      echo "Run --help for supported Z-Image and Qwen options." >&2
      exit 2
      ;;
  esac

  if [[ "${download_all_weights}" != "true" ]]; then
    export Z_IMAGE_NUNCHAKU_PRECISION
    export Z_IMAGE_NUNCHAKU_RANK
  fi
fi

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
      echo "Unsupported preset: RTX 50 Highest would require fp4 r256, but the published r256 Z-Image Turbo quantized weight is INT4 only." >&2
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

if [[ "${download_all_weights}" != "true" && "${download_int4_package}" != "true" && "${download_fp4_package}" != "true" ]]; then
  case "${Z_IMAGE_NUNCHAKU_PRECISION}:${Z_IMAGE_NUNCHAKU_RANK}" in
    auto:32|auto:128|int4:32|int4:128|int4:256|fp4:32|fp4:128) ;;
    auto:256)
      echo "Unsupported weight: auto r256. The published r256 Z-Image Turbo quantized weight is INT4 only." >&2
      exit 2
      ;;
    fp4:256)
      echo "Unsupported weight: fp4 r256. The published r256 Z-Image Turbo quantized weight is INT4 only." >&2
      exit 2
      ;;
    *)
      echo "Unsupported weight: ${Z_IMAGE_NUNCHAKU_PRECISION} r${Z_IMAGE_NUNCHAKU_RANK}." >&2
      echo "Expected one of: int4 r32, int4 r128, int4 r256, fp4 r32, fp4 r128, auto r32, auto r128." >&2
      exit 2
      ;;
  esac
fi

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
  echo "MODEL FETCH STATUS: step=${label} repo=${repo_id} status=downloading"
  if [[ -n "${NYMPHS3D_PREFETCH_COMPONENT_HINT:-}" ]]; then
    echo "MODEL FETCH STATUS: downloading=${NYMPHS3D_PREFETCH_COMPONENT_HINT}"
  fi
  echo "MODEL FETCH STATUS: huggingface_cache_total=$(format_bytes "${current_cache_bytes}") downloaded_this_step=$(format_bytes "${cache_delta}") cache_dir=${NYMPHS3D_HF_CACHE_DIR}"
  echo "MODEL FETCH STATUS: this_repo_cache=$(format_bytes "${repo_bytes}") active_download_files=${incomplete_count}"
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

  echo "MODEL FETCH STARTED: step=${label} repo=${repo_id} cache_dir=${NYMPHS3D_HF_CACHE_DIR} progress_interval=${interval}s"
  (
    set +e
    local_attempt=1
    max_attempts=3
    while [[ "${local_attempt}" -le "${max_attempts}" ]]; do
      "$@"
      status=$?
      if [[ "${status}" -eq 0 || "${local_attempt}" -ge "${max_attempts}" ]]; then
        break
      fi
      next_attempt=$(( local_attempt + 1 ))
      echo "MODEL FETCH STATUS: step ${label} ${repo_id} download was interrupted. Retrying attempt ${next_attempt}/${max_attempts} using the existing cache."
      sleep $(( local_attempt * 5 ))
      local_attempt="${next_attempt}"
    done
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
    echo "MODEL FETCH COMPLETE: step=${label} repo=${repo_id}"
  else
    echo "MODEL FETCH FAILED: step=${label} repo=${repo_id} exit_status=${status}"
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
download_all = (os.getenv("ZIMAGE_FETCH_ALL_WEIGHTS") or "").strip().lower() in {"1", "true", "yes", "on"}
download_int4_package = (os.getenv("ZIMAGE_FETCH_INT4_PACKAGE") or "").strip().lower() in {"1", "true", "yes", "on"}
download_fp4_package = (os.getenv("ZIMAGE_FETCH_FP4_PACKAGE") or "").strip().lower() in {"1", "true", "yes", "on"}
cache_dir = os.getenv("NYMPHS3D_HF_CACHE_DIR") or None
token = os.getenv("NYMPHS3D_HF_TOKEN") or None
if download_all:
    filenames = [
        "svdq-int4_r32-z-image-turbo.safetensors",
        "svdq-int4_r128-z-image-turbo.safetensors",
        "svdq-int4_r256-z-image-turbo.safetensors",
        "svdq-fp4_r32-z-image-turbo.safetensors",
        "svdq-fp4_r128-z-image-turbo.safetensors",
    ]
elif download_int4_package:
    filenames = [
        "svdq-int4_r32-z-image-turbo.safetensors",
        "svdq-int4_r128-z-image-turbo.safetensors",
        "svdq-int4_r256-z-image-turbo.safetensors",
    ]
elif download_fp4_package:
    filenames = [
        "svdq-fp4_r32-z-image-turbo.safetensors",
        "svdq-fp4_r128-z-image-turbo.safetensors",
    ]
elif precision == "auto":
    try:
        from nunchaku.utils import get_precision
        device = os.getenv("Z_IMAGE_DEVICE") or "cuda"
        precision = get_precision(precision="auto", device=device)
    except Exception:
        precision = "int4"
    filenames = [f"svdq-{precision}_r{rank}-z-image-turbo.safetensors"]
else:
    filenames = [f"svdq-{precision}_r{rank}-z-image-turbo.safetensors"]

for filename in filenames:
    print(f"Z-Image Turbo quantized weight prefetch: {repo_id}/{filename}", flush=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, token=token)
    print(f"Z-Image Turbo quantized weight ready: {path}", flush=True)
PY
  )
}

prefetch_zimage_controlnet_weight() {
  (
    cd "${ZIMAGE_INSTALL_ROOT}"
    "$(zimage_python)" - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = os.getenv("Z_IMAGE_CONTROLNET_REPO") or "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"
filename = os.getenv("Z_IMAGE_CONTROLNET_FILE") or "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"
cache_dir = os.getenv("NYMPHS3D_HF_CACHE_DIR") or None
token = os.getenv("NYMPHS3D_HF_TOKEN") or None
print(f"Z-Image ControlNet prefetch: {repo_id}/{filename}", flush=True)
path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, token=token)
print(f"Z-Image ControlNet ready: {path}", flush=True)
PY
  )
}

prefetch_hf_snapshot_model() {
  local model_id="$1"
  local profile="${2:-full}"
  (
    cd "${ZIMAGE_INSTALL_ROOT}"
    "$(zimage_python)" scripts/prefetch_model.py --model-id "${model_id}" --profile "${profile}"
  )
}

prefetch_hf_file() {
  local repo_id="$1"
  local filename="$2"
  (
    cd "${ZIMAGE_INSTALL_ROOT}"
    HF_REPO_ID="${repo_id}" HF_FILENAME="${filename}" "$(zimage_python)" - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = os.environ["HF_REPO_ID"]
filename = os.environ["HF_FILENAME"]
cache_dir = os.getenv("NYMPHS3D_HF_CACHE_DIR") or None
token = os.getenv("NYMPHS3D_HF_TOKEN") or None
print(f"HF file prefetch: {repo_id}/{filename}", flush=True)
path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, token=token)
print(f"HF file ready: {path}", flush=True)
PY
  )
}

prefetch_qwen_edit_base() {
  prefetch_hf_snapshot_model "${QWEN_EDIT_MODEL_ID}" "qwen-image-edit-nunchaku-base"
}

prefetch_qwen_edit_weight() {
  prefetch_hf_file "${QWEN_EDIT_WEIGHT_REPO}" "${QWEN_EDIT_WEIGHT_FILE}"
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
echo "controlnet_weight=${Z_IMAGE_CONTROLNET_REPO}/${Z_IMAGE_CONTROLNET_FILE}"
echo "download_all_weights=${download_all_weights}"
echo "download_int4_package=${download_int4_package}"
echo "download_fp4_package=${download_fp4_package}"
echo "fetch_zimage=${fetch_zimage}"
echo "fetch_qwen_edit=${fetch_qwen_edit}"
if [[ "${fetch_qwen_edit}" == "true" ]]; then
  echo "qwen_edit_model=${QWEN_EDIT_MODEL_ID}"
  echo "qwen_edit_weight_repo=${QWEN_EDIT_WEIGHT_REPO}"
  echo "qwen_edit_weight_files=$(IFS=,; printf '%s' "${QWEN_EDIT_WEIGHT_FILES[*]}")"
fi
echo "hf_cache_dir=${NYMPHS3D_HF_CACHE_DIR}"

echo "model_fetch_plan=${selected_fetch_label:-Z-Image selected model files}"

if [[ "${fetch_zimage}" == "true" ]]; then
  export NYMPHS3D_PREFETCH_COMPONENT_HINT="required large base Z-Image Turbo model files"
  run_with_hf_download_progress \
    "Z-Image required base model" \
    "${Z_IMAGE_MODEL_ID}" \
    prefetch_zimage_base_model
  unset NYMPHS3D_PREFETCH_COMPONENT_HINT

  if [[ "${download_all_weights}" == "true" ]]; then
    export ZIMAGE_FETCH_ALL_WEIGHTS=1
    export NYMPHS3D_PREFETCH_COMPONENT_HINT="all published Nunchaku-compatible runtime weights"
  elif [[ "${download_int4_package}" == "true" ]]; then
    export ZIMAGE_FETCH_INT4_PACKAGE=1
    export NYMPHS3D_PREFETCH_COMPONENT_HINT="all published INT4 Nunchaku-compatible runtime weights"
  elif [[ "${download_fp4_package}" == "true" ]]; then
    export ZIMAGE_FETCH_FP4_PACKAGE=1
    export NYMPHS3D_PREFETCH_COMPONENT_HINT="all published FP4 Nunchaku-compatible runtime weights"
  else
    unset ZIMAGE_FETCH_ALL_WEIGHTS
    unset ZIMAGE_FETCH_INT4_PACKAGE
    unset ZIMAGE_FETCH_FP4_PACKAGE
    export NYMPHS3D_PREFETCH_COMPONENT_HINT="selected Nunchaku-compatible runtime weight: ${Z_IMAGE_NUNCHAKU_PRECISION} r${Z_IMAGE_NUNCHAKU_RANK}"
  fi
  run_with_hf_download_progress \
    "Z-Image selected Nunchaku weight" \
    "${Z_IMAGE_NUNCHAKU_MODEL_REPO}" \
    prefetch_zimage_nunchaku_weight
  unset NYMPHS3D_PREFETCH_COMPONENT_HINT

  export NYMPHS3D_PREFETCH_COMPONENT_HINT="Z-Image Turbo ControlNet union weight for Sprite Foundry direction control"
  run_with_hf_download_progress \
    "Z-Image ControlNet weight" \
    "${Z_IMAGE_CONTROLNET_REPO}" \
    prefetch_zimage_controlnet_weight
  unset NYMPHS3D_PREFETCH_COMPONENT_HINT

  unset ZIMAGE_FETCH_ALL_WEIGHTS
  unset ZIMAGE_FETCH_INT4_PACKAGE
  unset ZIMAGE_FETCH_FP4_PACKAGE
fi

if [[ "${fetch_qwen_edit}" == "true" ]]; then
  if [[ ${#QWEN_EDIT_WEIGHT_FILES[@]} -eq 0 ]]; then
    echo "Qwen Image Edit weight files were not selected." >&2
    exit 2
  fi
  export NYMPHS3D_PREFETCH_COMPONENT_HINT="Qwen Image Edit 2511 base model files"
  run_with_hf_download_progress \
    "Qwen Image Edit 2511 base" \
    "${QWEN_EDIT_MODEL_ID}" \
    prefetch_qwen_edit_base
  unset NYMPHS3D_PREFETCH_COMPONENT_HINT

  for QWEN_EDIT_WEIGHT_FILE in "${QWEN_EDIT_WEIGHT_FILES[@]}"; do
    export QWEN_EDIT_WEIGHT_FILE
    export NYMPHS3D_PREFETCH_COMPONENT_HINT="selected Qwen Image Edit 2511 Nunchaku transformer: ${QWEN_EDIT_WEIGHT_FILE}"
    run_with_hf_download_progress \
      "Qwen Image Edit 2511 weight" \
      "${QWEN_EDIT_WEIGHT_REPO}" \
      prefetch_qwen_edit_weight
    unset NYMPHS3D_PREFETCH_COMPONENT_HINT
  done
fi

save_zimage_generation_preset
echo "Z-Image generation preset saved: precision=${Z_IMAGE_NUNCHAKU_PRECISION} rank=${Z_IMAGE_NUNCHAKU_RANK} file=${ZIMAGE_PRESET_FILE}"
