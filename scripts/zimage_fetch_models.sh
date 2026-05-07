#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

if [[ ! -x "$(zimage_python)" ]]; then
  echo "Z-Image Turbo runtime is missing. Run scripts/install_zimage.sh first." >&2
  exit 1
fi

(
  cd "${ZIMAGE_INSTALL_ROOT}"
  "$(zimage_python)" scripts/prefetch_model.py
  "$(zimage_python)" - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = os.getenv("Z_IMAGE_NUNCHAKU_MODEL_REPO") or "nunchaku-ai/nunchaku-z-image-turbo"
rank = os.getenv("Z_IMAGE_NUNCHAKU_RANK") or "32"
precision = (os.getenv("Z_IMAGE_NUNCHAKU_PRECISION") or "auto").strip().lower()
if precision == "auto":
    try:
        from nunchaku.utils import get_precision
        precision = get_precision()
    except Exception:
        precision = "int4"

filename = f"svdq-{precision}_r{rank}-z-image-turbo.safetensors"
path = hf_hub_download(repo_id=repo_id, filename=filename)
print(f"nunchaku_weight={path}")
PY
)
