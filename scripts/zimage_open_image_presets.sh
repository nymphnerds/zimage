#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_zimage_common.sh"

NYMPHS_CONFIG_ROOT="${NYMPHS_CONFIG_ROOT:-$NYMPHS_DATA_ROOT/config}"
TARGET_DIR="${NYMPHS_IMAGE_PRESET_DIR:-${NYMPHS_IMAGE_PRESETS_DIR:-$NYMPHS_CONFIG_ROOT/image_presets}}"
mkdir -p "${TARGET_DIR}"

if [[ -d "${MODULE_ROOT}/prompt_presets" ]]; then
  TARGET_DIR="${TARGET_DIR}" PACKAGED_PRESET_DIR="${MODULE_ROOT}/prompt_presets" python3 - <<'PY'
import json
import os
import re
from pathlib import Path


def slug(value: str, fallback: str = "preset") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return cleaned[:52] or fallback


target = Path(os.environ["TARGET_DIR"]).expanduser()
source = Path(os.environ["PACKAGED_PRESET_DIR"]).expanduser()
target.mkdir(parents=True, exist_ok=True)

for path in sorted(source.glob("*.json")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(data, dict):
        continue

    raw_kind = str(data.get("kind") or data.get("type") or "").strip().lower()
    kind = "style" if raw_kind == "style" or "style" in data else "subject"
    preset_id = path.stem
    if kind == "style" and preset_id.endswith("_style"):
        preset_id = preset_id[:-6]

    prompt_text = str(data.get("style") or data.get("prompt") or "").strip()
    if not prompt_text:
        continue

    payload = {
        "name": str(data.get("name") or data.get("label") or preset_id.replace("_", " ").title()).strip(),
        "kind": kind,
        "description": str(data.get("description") or "").strip(),
    }
    if kind == "style":
        payload["style"] = prompt_text
    else:
        payload["prompt"] = prompt_text

    destination = target / f"{kind}__{slug(preset_id)}.json"
    if not destination.exists():
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

(target / ".defaults_seeded").write_text("Defaults seeded.\n", encoding="utf-8")
PY
fi

echo "directory=${TARGET_DIR}"
