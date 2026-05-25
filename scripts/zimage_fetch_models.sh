#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

requested_gpu_family=""
requested_preset=""
requested_weight=""
download_all_weights=false
selected_fetch_label=""
fetch_zimage=true
fetch_flux_dev=false
fetch_flux_kontext=false
fetch_flux_dev_all_precisions=false
fetch_flux_kontext_all_precisions=false
FLUX_DEV_PRECISION="int4"
FLUX_KONTEXT_PRECISION="int4"
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
  zimage_fetch_models.sh --model all_models --license-ack yes
  zimage_fetch_models.sh --model flux_dev_int4_r32 --license-ack yes
  zimage_fetch_models.sh --model flux_kontext_int4_r32 --license-ack yes
  zimage_fetch_models.sh --gpu-family rtx_20_30_40|rtx_50 --preset fast|balanced|highest
  zimage_fetch_models.sh [--precision auto|int4|fp4] [--rank 32|128|256] [--hf_token TOKEN]

Downloads local Nymphs Image model files. Z-Image Turbo is the fast default.
FLUX.1-dev is text-to-image. FLUX.1-Kontext-dev is image edit and parts extraction.
Brain owns local vision/planner model downloads.

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

Use --precision auto only with ranks that exist for both INT4 and FP4.

BFL FLUX.1-dev and FLUX.1-Kontext-dev are gated/non-commercial model families.
Use --license-ack yes only after accepting and complying with the upstream
model terms for your use.
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
    all_models|all-models|all_image_models|all-image-models)
      download_all_weights=true
      fetch_zimage=true
      fetch_flux_dev=true
      fetch_flux_kontext=true
      fetch_flux_dev_all_precisions=true
      fetch_flux_kontext_all_precisions=true
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="All Models (very large): Z-Image Turbo weights plus FLUX.1-dev and FLUX.1-Kontext-dev INT4/FP4 r32"
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
    flux_dev_int4_r32|flux-dev-int4-r32)
      fetch_zimage=false
      fetch_flux_dev=true
      FLUX_DEV_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="FLUX.1-dev INT4 r32"
      ;;
    flux_dev_fp4_r32|flux-dev-fp4-r32)
      fetch_zimage=false
      fetch_flux_dev=true
      FLUX_DEV_PRECISION="fp4"
      Z_IMAGE_NUNCHAKU_PRECISION="fp4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="FLUX.1-dev FP4 r32"
      ;;
    flux_kontext_int4_r32|flux-kontext-int4-r32)
      fetch_zimage=false
      fetch_flux_kontext=true
      FLUX_KONTEXT_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="FLUX.1-Kontext-dev INT4 r32"
      ;;
    flux_kontext_fp4_r32|flux-kontext-fp4-r32)
      fetch_zimage=false
      fetch_flux_kontext=true
      FLUX_KONTEXT_PRECISION="fp4"
      Z_IMAGE_NUNCHAKU_PRECISION="fp4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="FLUX.1-Kontext-dev FP4 r32"
      ;;
    qwen3_vl_8b_q4_vision|qwen3-vl-8b-q4-vision)
      echo "Brain owns local vision model downloads. Use Brain to fetch/configure Qwen or another vision-capable model." >&2
      exit 2
      ;;
    local_parts_flux_16gb|local-parts-flux-16gb|local_parts_stack_16gb|local-parts-stack-16gb)
      fetch_zimage=false
      fetch_flux_kontext=true
      FLUX_KONTEXT_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_PRECISION="int4"
      Z_IMAGE_NUNCHAKU_RANK="32"
      selected_fetch_label="FLUX.1-Kontext-dev INT4 r32"
      ;;
    *)
      echo "Unsupported Nymphs Image model selection: ${requested_weight}." >&2
      echo "Run --help for supported Z-Image and FLUX options." >&2
      exit 2
      ;;
  esac

  if [[ "${download_all_weights}" != "true" ]]; then
    export Z_IMAGE_NUNCHAKU_PRECISION
    export Z_IMAGE_NUNCHAKU_RANK
  fi
fi

if [[ "${fetch_flux_dev}" == "true" || "${fetch_flux_kontext}" == "true" ]] && [[ "${license_ack}" != "true" ]]; then
  cat >&2 <<'EOF'
LICENSE ACK REQUIRED:
FLUX.1-dev and FLUX.1-Kontext-dev are gated/non-commercial BFL models.
Accept the Hugging Face access pages with the same account used for your token:
https://huggingface.co/black-forest-labs/FLUX.1-dev
https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
Rerun Fetch Models after selecting "Yes" in the module action popup.
EOF
  exit 2
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

if [[ "${download_all_weights}" != "true" ]]; then
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
    case "${repo_id}" in
      black-forest-labs/FLUX.1-dev)
        echo "MODEL FETCH FAILED: step=${label} repo=${repo_id} status=failed error=flux_access_needed next_step=Accept FLUX.1-dev access, then run Fetch Models again. link=https://huggingface.co/black-forest-labs/FLUX.1-dev"
        ;;
      black-forest-labs/FLUX.1-Kontext-dev)
        echo "MODEL FETCH FAILED: step=${label} repo=${repo_id} status=failed error=flux_access_needed next_step=Accept FLUX.1-Kontext-dev access, then run Fetch Models again. link=https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev"
        ;;
      *)
        echo "MODEL FETCH FAILED: step=${label} repo=${repo_id} exit_status=${status}"
        ;;
    esac
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

prefetch_flux_dev() {
  prefetch_hf_snapshot_model "black-forest-labs/FLUX.1-dev" "full"
  if [[ "${fetch_flux_dev_all_precisions}" == "true" ]]; then
    prefetch_hf_file "nunchaku-tech/nunchaku-flux.1-dev" "svdq-int4_r32-flux.1-dev.safetensors"
    prefetch_hf_file "nunchaku-tech/nunchaku-flux.1-dev" "svdq-fp4_r32-flux.1-dev.safetensors"
  else
    prefetch_hf_file "nunchaku-tech/nunchaku-flux.1-dev" "svdq-${FLUX_DEV_PRECISION}_r32-flux.1-dev.safetensors"
  fi
}

prefetch_flux_kontext() {
  prefetch_hf_snapshot_model "black-forest-labs/FLUX.1-Kontext-dev" "full"
  if [[ "${fetch_flux_kontext_all_precisions}" == "true" ]]; then
    prefetch_hf_file "nunchaku-tech/nunchaku-flux.1-kontext-dev" "svdq-int4_r32-flux.1-kontext-dev.safetensors"
    prefetch_hf_file "nunchaku-tech/nunchaku-flux.1-kontext-dev" "svdq-fp4_r32-flux.1-kontext-dev.safetensors"
  else
    prefetch_hf_file "nunchaku-tech/nunchaku-flux.1-kontext-dev" "svdq-${FLUX_KONTEXT_PRECISION}_r32-flux.1-kontext-dev.safetensors"
  fi
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
echo "download_all_weights=${download_all_weights}"
echo "fetch_zimage=${fetch_zimage}"
echo "fetch_flux_dev=${fetch_flux_dev}"
echo "fetch_flux_kontext=${fetch_flux_kontext}"
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
    export NYMPHS3D_PREFETCH_COMPONENT_HINT="all published Nunchaku-compatible Blender weights"
  else
    unset ZIMAGE_FETCH_ALL_WEIGHTS
    export NYMPHS3D_PREFETCH_COMPONENT_HINT="selected Nunchaku-compatible Blender weight: ${Z_IMAGE_NUNCHAKU_PRECISION} r${Z_IMAGE_NUNCHAKU_RANK}"
  fi
  run_with_hf_download_progress \
    "Z-Image selected Blender weight" \
    "${Z_IMAGE_NUNCHAKU_MODEL_REPO}" \
    prefetch_zimage_nunchaku_weight
  unset NYMPHS3D_PREFETCH_COMPONENT_HINT
fi

if [[ "${fetch_flux_dev}" == "true" ]]; then
  export NYMPHS3D_PREFETCH_COMPONENT_HINT="FLUX.1-dev base model and Nunchaku r32 transformer"
  run_with_hf_download_progress \
    "FLUX.1-dev r32" \
    "black-forest-labs/FLUX.1-dev" \
    prefetch_flux_dev
  unset NYMPHS3D_PREFETCH_COMPONENT_HINT
fi

if [[ "${fetch_flux_kontext}" == "true" ]]; then
  export NYMPHS3D_PREFETCH_COMPONENT_HINT="FLUX.1-Kontext-dev base model and Nunchaku r32 transformer"
  run_with_hf_download_progress \
    "FLUX.1-Kontext-dev r32" \
    "black-forest-labs/FLUX.1-Kontext-dev" \
    prefetch_flux_kontext
  unset NYMPHS3D_PREFETCH_COMPONENT_HINT
fi

save_zimage_generation_preset
echo "Z-Image generation preset saved: precision=${Z_IMAGE_NUNCHAKU_PRECISION} rank=${Z_IMAGE_NUNCHAKU_RANK} file=${ZIMAGE_PRESET_FILE}"
