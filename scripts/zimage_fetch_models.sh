#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --precision)
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
      Z_IMAGE_NUNCHAKU_RANK="${2:-}"
      export Z_IMAGE_NUNCHAKU_RANK
      shift 2
      ;;
    --rank=*)
      Z_IMAGE_NUNCHAKU_RANK="${1#*=}"
      export Z_IMAGE_NUNCHAKU_RANK
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: zimage_fetch_models.sh [--precision auto|int4|fp4] [--rank 32|128|256]

Downloads the base Z-Image Turbo model files and the selected Nunchaku
quantized weight.

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

echo "zimage_model=${Z_IMAGE_MODEL_ID}"
echo "nunchaku_weight_repo=${Z_IMAGE_NUNCHAKU_MODEL_REPO}"
echo "nunchaku_precision=${Z_IMAGE_NUNCHAKU_PRECISION}"
echo "nunchaku_rank=${Z_IMAGE_NUNCHAKU_RANK}"

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
