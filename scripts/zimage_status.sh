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
running=false
version=not-installed
health=unavailable
state=available
marker="${ZIMAGE_INSTALL_ROOT}/.nymph-module-version"
last_log="${ZIMAGE_LOG_DIR}/zimage-server.log"
gpu_vram_mb="$(zimage_detect_gpu_vram_mb)"
recommended_precision="$(zimage_recommended_precision "${gpu_vram_mb}")"
status_requested_precision="${Z_IMAGE_NUNCHAKU_PRECISION}"
status_check_precision="${status_requested_precision}"
detail="Not installed."

read_zimage_model_cache_status() {
  if [[ ! -x "$(zimage_python)" ]]; then
    return 0
  fi

  ZIMAGE_STATUS_BASE_MODEL="${Z_IMAGE_MODEL_ID}" \
  ZIMAGE_STATUS_WEIGHT_REPO="${Z_IMAGE_NUNCHAKU_MODEL_REPO}" \
  ZIMAGE_STATUS_CACHE_DIR="${NYMPHS3D_HF_CACHE_DIR}" \
  "$(zimage_python)" - <<'PY' 2>/dev/null || true
import os

base_model = os.getenv("ZIMAGE_STATUS_BASE_MODEL") or "Tongyi-MAI/Z-Image-Turbo"
weight_repo = os.getenv("ZIMAGE_STATUS_WEIGHT_REPO") or "nunchaku-ai/nunchaku-z-image-turbo"
cache_dir = os.getenv("ZIMAGE_STATUS_CACHE_DIR") or None
weights = [
    ("int4_r32", "svdq-int4_r32-z-image-turbo.safetensors"),
    ("int4_r128", "svdq-int4_r128-z-image-turbo.safetensors"),
    ("int4_r256", "svdq-int4_r256-z-image-turbo.safetensors"),
    ("fp4_r32", "svdq-fp4_r32-z-image-turbo.safetensors"),
    ("fp4_r128", "svdq-fp4_r128-z-image-turbo.safetensors"),
]

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception:
    raise SystemExit(0)

base_patterns = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
]

try:
    snapshot_download(
        repo_id=base_model,
        cache_dir=cache_dir,
        local_files_only=True,
        allow_patterns=base_patterns,
    )
    base_downloaded = True
except Exception:
    base_downloaded = False

downloaded = []
missing = []
for label, filename in weights:
    try:
        hf_hub_download(
            repo_id=weight_repo,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        downloaded.append(label)
    except Exception:
        missing.append(label)

downloaded_models = []
if base_downloaded:
    downloaded_models.append("Base")
downloaded_models.extend(downloaded)

print(f"base_model_downloaded={'true' if base_downloaded else 'false'}")
print(f"downloaded_weights={','.join(downloaded) if downloaded else 'none'}")
print(f"missing_weights={','.join(missing) if missing else 'none'}")
print(f"downloaded_models={','.join(downloaded_models) if downloaded_models else 'none'}")
PY
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
    status_check_precision="$(
      NYMPHS_ZIMAGE_RECOMMENDED_PRECISION="${recommended_precision}" "$(zimage_python)" - <<'PY' 2>/dev/null
import os

try:
    from nunchaku.utils import get_precision

    device = os.getenv("Z_IMAGE_DEVICE") or "cuda"
    print(get_precision(precision="auto", device=device))
except Exception:
    print(os.getenv("NYMPHS_ZIMAGE_RECOMMENDED_PRECISION") or "int4")
PY
    )"
    [[ -n "${status_check_precision}" ]] || status_check_precision="${recommended_precision}"
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
    "$(zimage_python)" -m py_compile api_server.py model_manager.py nunchaku_compat.py scripts/prefetch_model.py >/dev/null 2>&1
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

cat <<EOF
id=zimage
name=Z-Image Turbo
installed=${installed}
runtime_present=${runtime_present}
data_present=${data_present}
version=${version}
env_ready=${env_ready}
models_ready=${models_ready}
base_model_downloaded=${base_model_downloaded}
downloaded_models=${downloaded_models}
downloaded_weights=${downloaded_weights}
missing_weights=${missing_weights}
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
gpu_vram_mb=${gpu_vram_mb:-unknown}
install_root=${ZIMAGE_INSTALL_ROOT}
outputs_dir=${ZIMAGE_OUTPUT_DIR}
hf_cache_dir=${NYMPHS3D_HF_CACHE_DIR}
venv=${ZIMAGE_VENV_DIR}
logs_dir=${ZIMAGE_LOG_DIR}
last_log=${last_log}
marker=${marker}
detail=${detail}
EOF
