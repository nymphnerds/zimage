#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

installed=false
runtime_present=false
data_present=false
env_ready=false
models_ready=unknown
base_model_downloaded=unknown
downloaded_models=none
downloaded_weights=none
missing_weights=none
weight_profile_selected=unknown
weight_profiles_available=zimage_int4_r32,zimage_int4_r128,zimage_int4_r256,zimage_fp4_r32,zimage_fp4_r128,qwen_edit_2511_ultimate_speed_int4,qwen_edit_2511_balance_int4,qwen_edit_2511_best_quality_int4,qwen_edit_2511_ultimate_speed_fp4,qwen_edit_2511_balance_fp4,qwen_edit_2511_best_quality_fp4
weight_profiles_downloaded=none
weight_profiles_missing=none
weight_profile_ready=false
zimage_ready=false
qwen_edit_2511_ultimate_speed_int4_ready=false
qwen_edit_2511_balance_int4_ready=false
qwen_edit_2511_best_quality_int4_ready=false
qwen_edit_2511_ultimate_speed_fp4_ready=false
qwen_edit_2511_balance_fp4_ready=false
qwen_edit_2511_best_quality_fp4_ready=false
qwen_edit_ready=false
brain_installed=false
brain_model_configured=false
brain_ready=false
brain_url="${NYMPHS_BRAIN_LLM_API_BASE_URL:-http://127.0.0.1:8000/v1}"
local_parts_ready=false
running=false
version=not-installed
health=unavailable
state=available
marker="${ZIMAGE_INSTALL_ROOT}/.nymph-module-version"
last_log="${ZIMAGE_LOG_DIR}/zimage-server.log"
gpu_vram_mb="${ZIMAGE_STATUS_GPU_VRAM_MB:-unknown}"
recommended_precision="${ZIMAGE_STATUS_RECOMMENDED_PRECISION:-int4}"
status_requested_precision="${Z_IMAGE_NUNCHAKU_PRECISION}"
status_check_precision="${status_requested_precision}"
detail="Not installed."

read_zimage_model_cache_status() {
  local base_cache="${NYMPHS3D_HF_CACHE_DIR}/models--Tongyi-MAI--Z-Image-Turbo/snapshots"
  local base_ref="${NYMPHS3D_HF_CACHE_DIR}/models--Tongyi-MAI--Z-Image-Turbo/refs/main"
  local weight_cache="${NYMPHS3D_HF_CACHE_DIR}/models--nunchaku-ai--nunchaku-z-image-turbo/snapshots"
  local base_downloaded=false
  local downloaded=()
  local missing=()

  if [[ -f "${base_ref}" ]]; then
    base_snapshot="$(cat "${base_ref}" 2>/dev/null || true)"
    if [[ -n "${base_snapshot}" &&
          -f "${base_cache}/${base_snapshot}/model_index.json" &&
          -d "${base_cache}/${base_snapshot}/transformer" &&
          -d "${base_cache}/${base_snapshot}/vae" ]]; then
      base_downloaded=true
    fi
  fi

  local labels=(
    int4_r32
    int4_r128
    int4_r256
    fp4_r32
    fp4_r128
  )
  local files=(
    svdq-int4_r32-z-image-turbo.safetensors
    svdq-int4_r128-z-image-turbo.safetensors
    svdq-int4_r256-z-image-turbo.safetensors
    svdq-fp4_r32-z-image-turbo.safetensors
    svdq-fp4_r128-z-image-turbo.safetensors
  )

  local index
  if [[ -d "${weight_cache}" ]]; then
    for index in "${!labels[@]}"; do
      if [[ -n "$(find -L "${weight_cache}" -mindepth 2 -maxdepth 2 -type f -name "${files[$index]}" -print -quit 2>/dev/null)" ]]; then
        downloaded+=("${labels[$index]}")
      else
        missing+=("${labels[$index]}")
      fi
    done
  else
    missing=("${labels[@]}")
  fi

  local downloaded_models=()
  [[ "${base_downloaded}" == "true" ]] && downloaded_models+=(Base)
  downloaded_models+=("${downloaded[@]}")

  printf 'base_model_downloaded=%s\n' "${base_downloaded}"
  printf 'downloaded_weights=%s\n' "$([[ ${#downloaded[@]} -gt 0 ]] && (IFS=,; printf '%s' "${downloaded[*]}") || printf 'none')"
  printf 'missing_weights=%s\n' "$([[ ${#missing[@]} -gt 0 ]] && (IFS=,; printf '%s' "${missing[*]}") || printf 'none')"
  printf 'downloaded_models=%s\n' "$([[ ${#downloaded_models[@]} -gt 0 ]] && (IFS=,; printf '%s' "${downloaded_models[*]}") || printf 'none')"
}

hf_snapshot_has_path() {
  local repo_id="$1"
  local required_path="$2"
  local repo_dir="${NYMPHS3D_HF_CACHE_DIR}/models--${repo_id//\//--}"
  local ref_file="${repo_dir}/refs/main"
  local snapshots_dir="${repo_dir}/snapshots"
  local snapshot=""
  if [[ -f "${ref_file}" ]]; then
    snapshot="$(cat "${ref_file}" 2>/dev/null || true)"
    if [[ -n "${snapshot}" && -e "${snapshots_dir}/${snapshot}/${required_path}" ]]; then
      return 0
    fi
  fi
  [[ -d "${snapshots_dir}" ]] || return 1
  find -L "${snapshots_dir}" -mindepth 2 -maxdepth 3 -path "*/${required_path}" -print -quit 2>/dev/null | grep -q .
}

hf_snapshot_has_file() {
  local repo_id="$1"
  local filename="$2"
  local snapshots_dir="${NYMPHS3D_HF_CACHE_DIR}/models--${repo_id//\//--}/snapshots"
  [[ -d "${snapshots_dir}" ]] || return 1
  find -L "${snapshots_dir}" -mindepth 2 -maxdepth 2 -type f -name "${filename}" -print -quit 2>/dev/null | grep -q .
}

qwen_edit_base_ready_check() {
  local repo_id="Qwen/Qwen-Image-Edit-2511"
  local required_paths=(
    "model_index.json"
    "scheduler/scheduler_config.json"
    "processor/preprocessor_config.json"
    "processor/tokenizer.json"
    "processor/tokenizer_config.json"
    "text_encoder/config.json"
    "text_encoder/model-00001-of-00004.safetensors"
    "text_encoder/model-00002-of-00004.safetensors"
    "text_encoder/model-00003-of-00004.safetensors"
    "text_encoder/model-00004-of-00004.safetensors"
    "text_encoder/model.safetensors.index.json"
    "tokenizer/merges.txt"
    "tokenizer/tokenizer_config.json"
    "tokenizer/vocab.json"
    "vae/config.json"
    "vae/diffusion_pytorch_model.safetensors"
  )
  local required_path
  for required_path in "${required_paths[@]}"; do
    if ! hf_snapshot_has_path "${repo_id}" "${required_path}"; then
      return 1
    fi
  done
  return 0
}

if [[ -d "${NYMPHS3D_HF_CACHE_DIR}" ]]; then
  while IFS='=' read -r key value; do
    case "${key}" in
      base_model_downloaded) base_model_downloaded="${value}" ;;
      downloaded_models) downloaded_models="${value}" ;;
      downloaded_weights) downloaded_weights="${value}" ;;
      missing_weights) missing_weights="${value}" ;;
    esac
  done < <(read_zimage_model_cache_status)
fi

if [[ "${base_model_downloaded}" == "true" && "${downloaded_weights}" != "none" && -n "${downloaded_weights}" ]]; then
  zimage_ready=true
fi

qwen_edit_base_ready=false
if qwen_edit_base_ready_check; then
  qwen_edit_base_ready=true
fi
if hf_snapshot_has_file "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_ultimate_speed_int4.safetensors"; then
  qwen_edit_2511_ultimate_speed_int4_ready=true
fi
if hf_snapshot_has_file "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_balance_int4.safetensors"; then
  qwen_edit_2511_balance_int4_ready=true
fi
if hf_snapshot_has_file "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_best_quality_int4.safetensors"; then
  qwen_edit_2511_best_quality_int4_ready=true
fi
if hf_snapshot_has_file "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_ultimate_speed_fp4.safetensors"; then
  qwen_edit_2511_ultimate_speed_fp4_ready=true
fi
if hf_snapshot_has_file "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_balance_fp4.safetensors"; then
  qwen_edit_2511_balance_fp4_ready=true
fi
if hf_snapshot_has_file "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511" "nunchaku_qwen_image_edit_2511_best_quality_fp4.safetensors"; then
  qwen_edit_2511_best_quality_fp4_ready=true
fi
if [[ "${qwen_edit_base_ready}" == "true" &&
      ( "${qwen_edit_2511_ultimate_speed_int4_ready}" == "true" ||
        "${qwen_edit_2511_balance_int4_ready}" == "true" ||
        "${qwen_edit_2511_best_quality_int4_ready}" == "true" ||
        "${qwen_edit_2511_ultimate_speed_fp4_ready}" == "true" ||
        "${qwen_edit_2511_balance_fp4_ready}" == "true" ||
        "${qwen_edit_2511_best_quality_fp4_ready}" == "true" ) ]]; then
  qwen_edit_ready=true
fi

profile_downloaded=()
profile_missing=()
if [[ -n "${downloaded_weights}" && "${downloaded_weights}" != "none" ]]; then
  IFS=',' read -ra zimage_downloaded_profiles <<< "${downloaded_weights}"
  for zimage_profile in "${zimage_downloaded_profiles[@]}"; do
    profile_downloaded+=("zimage_${zimage_profile}")
  done
fi
if [[ -n "${missing_weights}" && "${missing_weights}" != "none" ]]; then
  IFS=',' read -ra zimage_missing_profiles <<< "${missing_weights}"
  for zimage_profile in "${zimage_missing_profiles[@]}"; do
    profile_missing+=("zimage_${zimage_profile}")
  done
fi
if [[ "${qwen_edit_base_ready}" == "true" && "${qwen_edit_2511_ultimate_speed_int4_ready}" == "true" ]]; then
  profile_downloaded+=(qwen_edit_2511_ultimate_speed_int4)
else
  profile_missing+=(qwen_edit_2511_ultimate_speed_int4)
fi
if [[ "${qwen_edit_base_ready}" == "true" && "${qwen_edit_2511_balance_int4_ready}" == "true" ]]; then
  profile_downloaded+=(qwen_edit_2511_balance_int4)
else
  profile_missing+=(qwen_edit_2511_balance_int4)
fi
if [[ "${qwen_edit_base_ready}" == "true" && "${qwen_edit_2511_best_quality_int4_ready}" == "true" ]]; then
  profile_downloaded+=(qwen_edit_2511_best_quality_int4)
else
  profile_missing+=(qwen_edit_2511_best_quality_int4)
fi
if [[ "${qwen_edit_base_ready}" == "true" && "${qwen_edit_2511_ultimate_speed_fp4_ready}" == "true" ]]; then
  profile_downloaded+=(qwen_edit_2511_ultimate_speed_fp4)
else
  profile_missing+=(qwen_edit_2511_ultimate_speed_fp4)
fi
if [[ "${qwen_edit_base_ready}" == "true" && "${qwen_edit_2511_balance_fp4_ready}" == "true" ]]; then
  profile_downloaded+=(qwen_edit_2511_balance_fp4)
else
  profile_missing+=(qwen_edit_2511_balance_fp4)
fi
if [[ "${qwen_edit_base_ready}" == "true" && "${qwen_edit_2511_best_quality_fp4_ready}" == "true" ]]; then
  profile_downloaded+=(qwen_edit_2511_best_quality_fp4)
else
  profile_missing+=(qwen_edit_2511_best_quality_fp4)
fi
if [[ ${#profile_downloaded[@]} -gt 0 ]]; then
  weight_profiles_downloaded="$(IFS=,; printf '%s' "${profile_downloaded[*]}")"
else
  weight_profiles_downloaded=none
fi
if [[ ${#profile_missing[@]} -gt 0 ]]; then
  weight_profiles_missing="$(IFS=,; printf '%s' "${profile_missing[*]}")"
else
  weight_profiles_missing=none
fi

downloaded_model_labels=()
if [[ -n "${downloaded_models}" && "${downloaded_models}" != "none" ]]; then
  IFS=',' read -ra zimage_downloaded_model_labels <<< "${downloaded_models}"
  downloaded_model_labels+=("${zimage_downloaded_model_labels[@]}")
fi
[[ "${qwen_edit_ready}" == "true" ]] && downloaded_model_labels+=(Qwen-Image-Edit-2511)
if [[ ${#downloaded_model_labels[@]} -gt 0 ]]; then
  downloaded_models="$(IFS=,; printf '%s' "${downloaded_model_labels[*]}")"
else
  downloaded_models=none
fi

brain_root="${NYMPHS_BRAIN_INSTALL_ROOT:-$HOME/Nymphs-Brain}"
if [[ -f "${brain_root}/.nymph-module-version" ]]; then
  brain_installed=true
fi

if [[ -f "${brain_root}/bin/lms-start" ]]; then
  configured_brain_model="$(sed -n 's/^MODEL_KEY="\([^"]*\)".*/\1/p' "${brain_root}/bin/lms-start" | head -n 1)"
  if [[ -n "${configured_brain_model}" && "${configured_brain_model}" != "none" ]]; then
    brain_model_configured=true
  fi
fi

if [[ "${brain_installed}" == "true" && "${brain_model_configured}" == "true" ]]; then
  brain_ready=true
fi

if [[ "${qwen_edit_ready}" == "true" && "${brain_ready}" == "true" ]]; then
  local_parts_ready=true
fi

if [[ -f "${marker}" ]]; then
  installed=true
  runtime_present=true
  version="$(head -n 1 "${marker}" 2>/dev/null || true)"
  [[ -n "${version}" ]] || version=unknown
  detail="Source installed."
elif [[ -f "${ZIMAGE_INSTALL_ROOT}/api_server.py" && -f "${ZIMAGE_INSTALL_ROOT}/model_manager.py" ]]; then
  installed=true
  runtime_present=true
  if [[ -f "${ZIMAGE_INSTALL_ROOT}/nymph.json" ]]; then
    version="$(
      ZIMAGE_STATUS_MANIFEST="${ZIMAGE_INSTALL_ROOT}/nymph.json" "$(zimage_python)" - <<'PY' 2>/dev/null || true
import json
import os

try:
    with open(os.environ["ZIMAGE_STATUS_MANIFEST"], "r", encoding="utf-8") as handle:
        print(json.load(handle).get("version") or "unknown")
except Exception:
    print("unknown")
PY
    )"
  fi
  [[ -n "${version}" ]] || version=unknown
  detail="Source installed."
fi

if [[ -d "${ZIMAGE_OUTPUT_DIR}" && -n "$(find "${ZIMAGE_OUTPUT_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]] ||
   [[ -d "${ZIMAGE_CONFIG_DIR}" && -n "$(find "${ZIMAGE_CONFIG_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]] ||
   [[ -d "${ZIMAGE_INSTALL_ROOT}/outputs" && -n "$(find "${ZIMAGE_INSTALL_ROOT}/outputs" -mindepth 1 -print -quit 2>/dev/null)" ]] ||
   [[ -d "${ZIMAGE_LOG_DIR}" && -n "$(find "${ZIMAGE_LOG_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]] ||
   [[ -d "${ZIMAGE_INSTALL_ROOT}/logs" && -n "$(find "${ZIMAGE_INSTALL_ROOT}/logs" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  data_present=true
fi

if [[ "${installed}" == "true" && -x "$(zimage_python)" ]]; then
  env_ready=true
  detail="Runtime environment present."
  if [[ "${status_check_precision}" == "auto" ]]; then
    status_check_precision="${recommended_precision}"
  fi
  case "${status_check_precision}" in
    int4|fp4) ;;
    *) status_check_precision="${recommended_precision}" ;;
  esac
fi

if [[ "${installed}" == "true" ]] && zimage_is_running; then
  running=true
  if zimage_probe_url "${Z_IMAGE_SERVER_URL}/health" >/dev/null 2>&1; then
    health=ok
  else
    health=unreachable
  fi
fi

if [[ "${env_ready}" == "true" &&
      "${base_model_downloaded}" == "true" &&
      "${downloaded_weights}" != "none" &&
      -n "${downloaded_weights}" ]]; then
  models_ready=true
  if [[ "${running}" != "true" ]]; then
    health=ok
  fi
  detail="Runtime and cached model files are ready."
  if [[ "${qwen_edit_ready}" != "true" ]]; then
    detail="Runtime and Z-Image files are ready. Qwen Image Edit is not fetched yet; use Fetch Models for local editing and parts extraction."
  fi
elif [[ "${env_ready}" == "true" &&
        "${base_model_downloaded}" == "true" &&
        ( "${downloaded_weights}" == "none" || -z "${downloaded_weights}" ) ]]; then
  models_ready=false
  health=model-download-needed
  detail="Model files need downloading for ${status_check_precision} r${Z_IMAGE_NUNCHAKU_RANK}. Use Fetch Models to download the selected Nunchaku-compatible quantized weight."
elif [[ "${env_ready}" == "true" ]]; then
  if (
    cd "${ZIMAGE_INSTALL_ROOT}"
    export Z_IMAGE_NUNCHAKU_PRECISION="${status_check_precision}"
    "$(zimage_python)" -m py_compile api_server.py brain_client.py image_providers.py model_manager.py nunchaku_compat.py scripts/prefetch_model.py >/dev/null 2>&1
    "$(zimage_python)" scripts/prefetch_model.py --local-files-only >/dev/null 2>&1
  ); then
    models_ready=true
    if [[ "${running}" != "true" ]]; then
      health=ok
    fi
    detail="Runtime and cached model files are ready."
  else
    models_ready=false
    health=model-download-needed
    detail="Model files need downloading for ${status_check_precision} r${Z_IMAGE_NUNCHAKU_RANK}. Use Fetch Models to download the base Z-Image Turbo model and selected Nunchaku-compatible quantized weight."
  fi
fi

if [[ "${installed}" == "true" && "${running}" == "true" ]]; then
  state=running
elif [[ "${installed}" == "true" ]]; then
  state=installed
  if [[ "${env_ready}" != "true" ]]; then
    state=needs_attention
    health=degraded
    detail="Z-Image runtime files are installed, but the Python runtime is missing."
  elif [[ "${models_ready}" == "false" ]]; then
    state=model_download_needed
    health=model-download-needed
  fi
elif [[ "${data_present}" == "true" ]]; then
  detail="Z-Image preserved data remains, but runtime files are not installed."
fi

zimage_weight_profile_selected="${status_check_precision}_r${Z_IMAGE_NUNCHAKU_RANK}"
weight_profile_selected="zimage_${zimage_weight_profile_selected}"
if [[ ",${downloaded_weights}," == *",${zimage_weight_profile_selected},"* ]]; then
  weight_profile_ready=true
else
  weight_profile_ready=false
fi

cat <<EOF
id=zimage
name=Z-Image Turbo
installed=${installed}
runtime_present=${runtime_present}
data_present=${data_present}
version=${version}
env_ready=${env_ready}
models_ready=${models_ready}
zimage_ready=${zimage_ready}
qwen_edit_ready=${qwen_edit_ready}
qwen_edit_base_ready=${qwen_edit_base_ready}
qwen_edit_2511_ultimate_speed_int4_ready=${qwen_edit_2511_ultimate_speed_int4_ready}
qwen_edit_2511_balance_int4_ready=${qwen_edit_2511_balance_int4_ready}
qwen_edit_2511_best_quality_int4_ready=${qwen_edit_2511_best_quality_int4_ready}
qwen_edit_2511_ultimate_speed_fp4_ready=${qwen_edit_2511_ultimate_speed_fp4_ready}
qwen_edit_2511_balance_fp4_ready=${qwen_edit_2511_balance_fp4_ready}
qwen_edit_2511_best_quality_fp4_ready=${qwen_edit_2511_best_quality_fp4_ready}
brain_installed=${brain_installed}
brain_model_configured=${brain_model_configured}
brain_ready=${brain_ready}
brain_url=${brain_url}
local_parts_ready=${local_parts_ready}
base_model_downloaded=${base_model_downloaded}
downloaded_models=${downloaded_models}
downloaded_weights=${downloaded_weights}
missing_weights=${missing_weights}
weight_profile_selected=${weight_profile_selected}
weight_profiles_available=${weight_profiles_available}
weight_profiles_downloaded=${weight_profiles_downloaded}
weight_profiles_missing=${weight_profiles_missing}
weight_profile_ready=${weight_profile_ready}
running=${running}
state=${state}
health=${health}
url=${Z_IMAGE_SERVER_URL}
health_url=${Z_IMAGE_SERVER_URL}/health
server_info_url=${Z_IMAGE_SERVER_URL}/server_info
model_id=${Z_IMAGE_MODEL_ID}
nunchaku_weight_repo=${Z_IMAGE_NUNCHAKU_MODEL_REPO}
nunchaku_rank=${Z_IMAGE_NUNCHAKU_RANK}
nunchaku_precision=${Z_IMAGE_NUNCHAKU_PRECISION}
status_requested_precision=${status_requested_precision}
status_check_precision=${status_check_precision}
recommended_precision=${recommended_precision}
gpu_vram_mb=${gpu_vram_mb}
install_root=${ZIMAGE_INSTALL_ROOT}
outputs_dir=${ZIMAGE_OUTPUT_DIR}
hf_cache_dir=${NYMPHS3D_HF_CACHE_DIR}
venv=${ZIMAGE_VENV_DIR}
logs_dir=${ZIMAGE_LOG_DIR}
last_log=${last_log}
marker=${marker}
detail=${detail}
EOF
