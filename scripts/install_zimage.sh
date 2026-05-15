#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

echo "Z-Image Turbo: installing module into ${ZIMAGE_INSTALL_ROOT}"

has_apt_candidate() {
  local package_name="$1"
  local candidate
  candidate="$(apt-cache policy "${package_name}" 2>/dev/null | awk '/Candidate:/ {print $2; exit}')"
  [[ -n "${candidate}" && "${candidate}" != "(none)" ]]
}

refresh_cuda_env() {
  if [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
    return 0
  fi

  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path
    local cuda_home
    nvcc_path="$(command -v nvcc)"
    cuda_home="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
    export CUDA_HOME="${CUDA_HOME:-${cuda_home}}"
    export CUDACXX="${CUDACXX:-${nvcc_path}}"
    return 0
  fi

  return 1
}

download_cuda_keyring() {
  local keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
  local keyring_deb="${HOME}/cuda-keyring_1.1-1_all.deb"

  if [[ -f "${keyring_deb}" ]]; then
    printf '%s\n' "${keyring_deb}"
    return 0
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -O "${keyring_deb}" "${keyring_url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "${keyring_deb}" "${keyring_url}"
  else
    echo "Neither wget nor curl is available to download the NVIDIA CUDA keyring." >&2
    return 1
  fi

  printf '%s\n' "${keyring_deb}"
}

ensure_cuda_toolkit() {
  if refresh_cuda_env; then
    echo "CUDA compiler detected:"
    nvcc --version | sed -n '1,4p'
    return 0
  fi

  echo "Nunchaku is being built from the Nymphs fork and needs the CUDA compiler (nvcc)."

  if ! command -v sudo >/dev/null 2>&1 ||
     ! command -v apt-cache >/dev/null 2>&1 ||
     ! command -v dpkg >/dev/null 2>&1; then
    echo "nvcc is missing and automatic CUDA toolkit installation is not available." >&2
    echo "Install CUDA Toolkit 13.0 for WSL, then retry Z-Image Turbo installation." >&2
    exit 1
  fi

  if ! has_apt_candidate cuda-toolkit-13-0; then
    echo "Adding NVIDIA CUDA 13.0 apt repository for WSL..."
    sudo apt update
    local keyring_deb
    keyring_deb="$(download_cuda_keyring)"
    sudo dpkg -i "${keyring_deb}"
    sudo apt update
  fi

  if ! has_apt_candidate cuda-toolkit-13-0; then
    echo "cuda-toolkit-13-0 is still not available from apt after adding the NVIDIA repository." >&2
    echo "Install CUDA Toolkit 13.0 for WSL, then retry Z-Image Turbo installation." >&2
    exit 1
  fi

  echo "Installing CUDA Toolkit 13.0 so Nunchaku can compile native CUDA extensions..."
  sudo apt install -y cuda-toolkit-13-0

  if ! refresh_cuda_env; then
    echo "CUDA Toolkit installation completed, but nvcc was not found." >&2
    echo "Expected /usr/local/cuda-13.0/bin/nvcc. Check the CUDA toolkit install, then retry." >&2
    exit 1
  fi

  echo "CUDA compiler ready:"
  nvcc --version | sed -n '1,4p'
}

ensure_python311() {
  if command -v python3.11 >/dev/null 2>&1 &&
     python3.11 - <<'PY' >/dev/null 2>&1 &&
import venv
PY
     { ! command -v dpkg >/dev/null 2>&1 || dpkg -s python3.11-dev >/dev/null 2>&1; }; then
    return 0
  fi

  echo "python3.11, python3.11-venv, and python3.11-dev are required for the pinned Z-Image Turbo Nunchaku runtime."

  if ! command -v sudo >/dev/null 2>&1 || ! command -v apt-cache >/dev/null 2>&1; then
    echo "python3.11 is missing and automatic apt installation is not available." >&2
    echo "Install python3.11, python3.11-venv, and python3.11-dev, then retry." >&2
    exit 1
  fi

  if ! has_apt_candidate python3.11 || ! has_apt_candidate python3.11-venv || ! has_apt_candidate python3.11-dev; then
    echo "Python 3.11 packages are not available in current apt sources. Adding deadsnakes PPA..."
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
  fi

  echo "Installing Python 3.11 runtime packages..."
  sudo apt install -y python3.11 python3.11-venv python3.11-dev

  if ! command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11 installation did not complete. Install python3.11, python3.11-venv, and python3.11-dev, then retry." >&2
    exit 1
  fi
}

ensure_python311
ensure_cuda_toolkit

module_version="$(python3.11 - "${MODULE_ROOT}/nymph.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

print(str(manifest.get("version", "unknown")).strip() or "unknown")
PY
)"

install_parent="$(dirname "${ZIMAGE_INSTALL_ROOT}")"
mkdir -p "${install_parent}"

STAGING_ROOT="$(mktemp -d "${install_parent}/.zimage-install-staging.XXXXXX")"
OLD_ROOT=""
FILTERED_LOCK_FILE=""

cleanup_on_exit() {
  local exit_code=$?
  if [[ -n "${FILTERED_LOCK_FILE}" ]]; then
    rm -f "${FILTERED_LOCK_FILE}" || true
  fi
  if [[ "${exit_code}" -ne 0 ]]; then
    rm -rf "${STAGING_ROOT}" || true
    if [[ -n "${OLD_ROOT}" && -d "${OLD_ROOT}" && ! -e "${ZIMAGE_INSTALL_ROOT}" ]]; then
      mv "${OLD_ROOT}" "${ZIMAGE_INSTALL_ROOT}" || true
    fi
  fi
}
trap cleanup_on_exit EXIT

mkdir -p "${STAGING_ROOT}/scripts"

install -m 644 "${MODULE_ROOT}/.env.example" "${STAGING_ROOT}/.env.example"
install -m 644 "${MODULE_ROOT}/.gitignore" "${STAGING_ROOT}/.gitignore"
install -m 644 "${MODULE_ROOT}/README.md" "${STAGING_ROOT}/README.md"
install -m 644 "${MODULE_ROOT}/nymph.json" "${STAGING_ROOT}/nymph.json"
install -m 644 "${MODULE_ROOT}/api_server.py" "${STAGING_ROOT}/api_server.py"
install -m 644 "${MODULE_ROOT}/config.py" "${STAGING_ROOT}/config.py"
install -m 644 "${MODULE_ROOT}/image_store.py" "${STAGING_ROOT}/image_store.py"
install -m 644 "${MODULE_ROOT}/model_manager.py" "${STAGING_ROOT}/model_manager.py"
install -m 644 "${MODULE_ROOT}/nunchaku_compat.py" "${STAGING_ROOT}/nunchaku_compat.py"
install -m 644 "${MODULE_ROOT}/progress_state.py" "${STAGING_ROOT}/progress_state.py"
install -m 644 "${MODULE_ROOT}/requirements.lock.txt" "${STAGING_ROOT}/requirements.lock.txt"
install -m 644 "${MODULE_ROOT}/schemas.py" "${STAGING_ROOT}/schemas.py"

for script in \
  _zimage_common.sh \
  prefetch_model.py \
  run_nunchaku_zimage_test.py \
  zimage_delete_models.sh \
  zimage_fetch_models.sh \
  zimage_logs.sh \
  zimage_open.sh \
  zimage_open_outputs.sh \
  zimage_open_weights.sh \
  zimage_smoke_test.sh \
  zimage_start.sh \
  zimage_status.sh \
  zimage_stop.sh \
  zimage_uninstall.sh; do
  install -m 755 "${MODULE_ROOT}/scripts/${script}" "${STAGING_ROOT}/scripts/${script}"
done

if [[ -d "${MODULE_ROOT}/ui" ]]; then
  mkdir -p "${STAGING_ROOT}/ui"
  find "${MODULE_ROOT}/ui" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' ui_file; do
    install -m 644 "${ui_file}" "${STAGING_ROOT}/ui/$(basename "${ui_file}")"
  done
fi

STAGING_VENV_DIR="${STAGING_ROOT}/.venv-nunchaku"
echo "Creating staged runtime venv: ${STAGING_VENV_DIR}"
python3.11 -m venv "${STAGING_VENV_DIR}"

VENV_PYTHON="${STAGING_VENV_DIR}/bin/python"
FILTERED_LOCK_FILE="$(mktemp)"

"${VENV_PYTHON}" -m pip install --upgrade pip "setuptools<82" wheel
"${VENV_PYTHON}" -m pip install torch==2.11.0 torchvision torchaudio --index-url "${ZIMAGE_TORCH_INDEX_URL}"

grep -Evi '(diffusers|safetensors)' "${STAGING_ROOT}/requirements.lock.txt" > "${FILTERED_LOCK_FILE}"
"${VENV_PYTHON}" -m pip install -r "${FILTERED_LOCK_FILE}"

"${VENV_PYTHON}" -m pip install --no-deps --pre "${ZIMAGE_NUNCHAKU_SPEC}"
"${VENV_PYTHON}" -m pip install httpx importlib_metadata einops "peft>=0.17" protobuf sentencepiece
"${VENV_PYTHON}" -m pip install --no-deps --force-reinstall safetensors==0.7.0

diffusers_attempts=3
diffusers_success=0
for diffusers_attempt in $(seq 1 "${diffusers_attempts}"); do
  if [[ "${diffusers_attempt}" -gt 1 ]]; then
    echo "Retrying diffusers install (${diffusers_attempt}/${diffusers_attempts})..."
    sleep $((5 * diffusers_attempt))
  fi

  if "${VENV_PYTHON}" -m pip install --no-deps --force-reinstall "${ZIMAGE_DIFFUSERS_SPEC}"; then
    diffusers_success=1
    break
  fi
done

if [[ "${diffusers_success}" -ne 1 ]]; then
  echo "diffusers install failed after ${diffusers_attempts} attempts." >&2
  exit 1
fi

(
  cd "${STAGING_ROOT}"
  "${VENV_PYTHON}" -m py_compile api_server.py config.py image_store.py model_manager.py nunchaku_compat.py progress_state.py schemas.py scripts/prefetch_model.py scripts/run_nunchaku_zimage_test.py
  "${VENV_PYTHON}" - <<'PY'
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from diffusers.pipelines.z_image.pipeline_z_image_img2img import ZImageImg2ImgPipeline
from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku_compat import patch_zimage_transformer_forward

patched = patch_zimage_transformer_forward(NunchakuZImageTransformer2DModel)
if not patched:
    raise SystemExit("nunchaku forward patch validation failed")
missing_methods = [
    name
    for name in ("update_lora_params", "set_lora_strength", "reset_lora")
    if not hasattr(NunchakuZImageTransformer2DModel, name)
]
if missing_methods:
    raise SystemExit(f"nunchaku LoRA method(s) missing after install: {', '.join(missing_methods)}")
print(f"zimage_pipeline={ZImagePipeline.__name__}")
print(f"zimage_img2img_pipeline={ZImageImg2ImgPipeline.__name__}")
print(f"nunchaku_transformer={NunchakuZImageTransformer2DModel.__name__}")
print("Z-Image Turbo runtime imports validated.")
PY
)

rm -f "${FILTERED_LOCK_FILE}"
FILTERED_LOCK_FILE=""

cat > "${STAGING_VENV_DIR}/.nymphs_nunchaku_runtime.json" <<EOF
{
  "nunchaku_repo": "${ZIMAGE_NUNCHAKU_REPO}",
  "nunchaku_commit": "${ZIMAGE_NUNCHAKU_COMMIT}",
  "diffusers": "${ZIMAGE_DIFFUSERS_SPEC}",
  "torch_index_url": "${ZIMAGE_TORCH_INDEX_URL}"
}
EOF

printf '%s\n' "${module_version}" > "${STAGING_ROOT}/.nymph-module-version"

if [[ -e "${ZIMAGE_INSTALL_ROOT}" && ! -d "${ZIMAGE_INSTALL_ROOT}" ]]; then
  echo "Install root exists but is not a directory: ${ZIMAGE_INSTALL_ROOT}" >&2
  exit 1
fi

"${SCRIPT_DIR}/zimage_stop.sh" || true

for preserve_name in outputs logs; do
  if [[ -e "${ZIMAGE_INSTALL_ROOT}/${preserve_name}" ]]; then
    rm -rf "${STAGING_ROOT:?}/${preserve_name}"
    mv "${ZIMAGE_INSTALL_ROOT}/${preserve_name}" "${STAGING_ROOT}/${preserve_name}"
  fi
done

if [[ -d "${ZIMAGE_INSTALL_ROOT}" ]]; then
  OLD_ROOT="${ZIMAGE_INSTALL_ROOT}.previous.$$"
  rm -rf "${OLD_ROOT}"
  mv "${ZIMAGE_INSTALL_ROOT}" "${OLD_ROOT}"
fi

mv "${STAGING_ROOT}" "${ZIMAGE_INSTALL_ROOT}"

if [[ -n "${OLD_ROOT}" ]]; then
  rm -rf "${OLD_ROOT}"
fi

trap - EXIT
echo "installed_module_version=${module_version}"
echo "Z-Image Turbo install complete."
