from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import uvicorn
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

from config import get_settings
from image_store import save_image_and_metadata
from model_manager import ModelManager
from progress_state import reset as progress_reset
from progress_state import snapshot as progress_snapshot
from progress_state import update as progress_update
from schemas import (
    ActiveTaskResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ServerInfoResponse,
)


VERSION = "0.1.0"
WORKER_ID = uuid.uuid4().hex[:6]
SETTINGS = get_settings()
MODEL_MANAGER = ModelManager(SETTINGS)
NYMPH_UI_PATH = Path(__file__).resolve().parent / "nymph_image.html"
OPENROUTER_API_ROOT = "https://openrouter.ai/api/v1"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PRESET_KINDS = {"subject", "style", "saved", "settings"}

app = FastAPI(
    title="Nymphs2D2 API",
    version=VERSION,
    description="Local Nymphs Stable Diffusion-style backend scaffold.",
)


def _log_stage(message: str, **fields):
    parts = [f"{key}={value}" for key, value in fields.items()]
    suffix = f" {' '.join(parts)}" if parts else ""
    print(f"[nymphs:zimage:stage] {message}{suffix}", flush=True)


def _coerce_dimension(value: int, *, maximum: int, label: str) -> int:
    if value <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    if value > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return max(64, value - (value % 8))


def _decode_base64_image(raw: str) -> Image.Image:
    payload = raw.split(",", 1)[1] if "," in raw else raw
    try:
        image = Image.open(BytesIO(base64.b64decode(payload)))
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid base64 image payload.") from exc


def _openrouter_env_file() -> Path:
    configured = os.getenv("ZIMAGE_OPENROUTER_ENV_FILE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "NymphsData" / "config" / "zimage" / "openrouter.env"


def _config_dir() -> Path:
    configured = os.getenv("ZIMAGE_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "NymphsData" / "config" / "zimage"


def _preset_dir(kind: str) -> Path:
    normalized = (kind or "").strip().lower()
    if normalized not in PRESET_KINDS:
        raise HTTPException(status_code=400, detail="Unsupported preset kind.")
    path = _config_dir() / "presets" / normalized
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_slug(value: str, fallback: str = "preset") -> str:
    return _slugify(value, fallback).replace("-", "_")


def _safe_preset_path(kind: str, preset_id: str) -> Path:
    return _preset_dir(kind) / f"{_safe_slug(preset_id, 'preset')}.json"


def _load_user_presets(kind: str) -> list[dict]:
    presets = []
    for path in sorted(_preset_dir(kind).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        preset_id = path.stem
        presets.append(
            {
                "id": preset_id,
                "name": str(data.get("name") or data.get("label") or preset_id.replace("_", " ").title()).strip(),
                "kind": kind,
                "prompt": str(data.get("prompt") or data.get("style") or "").strip(),
                "values": data.get("values") if isinstance(data.get("values"), dict) else {},
                "description": str(data.get("description") or "").strip(),
            }
        )
    return presets


def _read_openrouter_api_key_file() -> str:
    path = _openrouter_env_file()
    if not path.is_file():
        return ""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def _resolve_openrouter_api_key(payload: dict | None = None) -> str:
    payload = payload or {}
    return (
        (payload.get("api_key") or "").strip()
        or (os.getenv("OPENROUTER_API_KEY") or "").strip()
        or _read_openrouter_api_key_file()
    )


def _batch_metadata(payload: dict, *, default_type: str, default_label: str, default_item_label: str = "") -> dict:
    batch_id = str(payload.get("batch_id") or "").strip()
    if not batch_id:
        return {}
    metadata = {
        "batch_id": batch_id,
        "batch_label": str(payload.get("batch_label") or default_label).strip() or default_label,
        "batch_type": str(payload.get("batch_type") or default_type).strip() or default_type,
        "item_label": str(payload.get("item_label") or default_item_label).strip() or default_item_label,
    }
    for key in ("item_index", "item_total"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            metadata[key] = int(value)
        except Exception:
            pass
    return metadata


def _decode_data_url(raw: str) -> tuple[str, bytes]:
    value = (raw or "").strip()
    if not value:
        raise ValueError("Missing image payload.")
    if value.startswith(("http://", "https://")):
        with urlopen(value, timeout=180) as response:
            mime_type = response.headers.get_content_type() or "image/png"
            return mime_type, response.read()
    mime_type = "image/png"
    encoded = value
    if value.startswith("data:"):
        header, separator, encoded = value.partition(",")
        if not separator:
            raise ValueError("Malformed image data URL.")
        mime_type = header[5:].split(";", 1)[0] or mime_type
    try:
        return mime_type, base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError("Invalid base64 image payload.") from exc


def _image_suffix(mime_type: str) -> str:
    value = (mime_type or "").lower()
    if "jpeg" in value or "jpg" in value:
        return ".jpg"
    if "webp" in value:
        return ".webp"
    if "gif" in value:
        return ".gif"
    return ".png"


def _slugify(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return cleaned[:52] or fallback


def _safe_output_path(relative_path: str) -> Path:
    root = SETTINGS.output_dir.resolve()
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(status_code=404, detail="Output not found.")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Output not found.")
    return candidate


def _output_url(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(SETTINGS.output_dir.resolve()).as_posix()
    except Exception:
        rel = path.name
    return f"/outputs/{quote(rel, safe='/')}"


def _metadata_for(path: Path) -> dict:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _output_record(path: Path) -> dict:
    metadata = _metadata_for(path)
    return {
        "name": path.name,
        "path": str(path),
        "url": _output_url(path),
        "metadata_path": str(path.with_suffix(".json")) if path.with_suffix(".json").is_file() else "",
        "provider": metadata.get("provider") or metadata.get("backend") or "",
        "mode": metadata.get("mode") or "",
        "prompt": metadata.get("prompt") or "",
        "batch_id": metadata.get("batch_id") or "",
        "batch_label": metadata.get("batch_label") or "",
        "batch_type": metadata.get("batch_type") or "",
        "item_label": metadata.get("item_label") or "",
        "item_index": metadata.get("item_index"),
        "item_total": metadata.get("item_total"),
        "created": path.stat().st_mtime,
        "metadata": metadata,
    }


def _recent_outputs(limit: int = 80) -> list[dict]:
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    files = [
        path
        for path in SETTINGS.output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    outputs = []
    for path in files[: max(1, min(limit, 200))]:
        outputs.append(_output_record(path))
    return outputs


def _iter_lora_runs() -> list[dict]:
    root = Path(os.getenv("ZIMAGE_LORA_ROOT") or Path.home() / "LoRA" / "loras").expanduser()
    if not root.exists():
        return []
    runs = []
    seen = set()
    top_level = sorted(root.iterdir(), key=lambda item: item.name.lower())
    for path in top_level:
        if path.is_file() and path.suffix.lower() == ".safetensors":
            resolved = str(path)
            runs.append(
                {
                    "id": resolved,
                    "name": path.name,
                    "latest_file": resolved,
                    "latest_mtime": path.stat().st_mtime,
                    "is_file": True,
                }
            )
            seen.add(resolved)
    for child in top_level:
        if not child.is_dir():
            continue
        latest_file = None
        latest_mtime = -1.0
        for path in child.rglob("*.safetensors"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= latest_mtime:
                latest_file = path
                latest_mtime = mtime
        if latest_file is None:
            continue
        resolved = str(child)
        if resolved in seen:
            continue
        runs.append(
            {
                "id": resolved,
                "name": child.name,
                "latest_file": str(latest_file),
                "latest_mtime": latest_mtime,
                "is_file": False,
            }
        )
    runs.sort(key=lambda item: (item.get("latest_mtime", -1), item.get("name", "").lower()), reverse=True)
    return runs


def _iter_lora_checkpoints(run_id: str) -> list[dict]:
    path = Path(run_id or "").expanduser()
    if path.is_file() and path.suffix.lower() == ".safetensors":
        return [{"id": str(path), "name": path.name, "mtime": path.stat().st_mtime}]
    if not path.is_dir():
        return []
    checkpoints = []
    for checkpoint in path.rglob("*.safetensors"):
        try:
            mtime = checkpoint.stat().st_mtime
        except OSError:
            continue
        checkpoints.append({"id": str(checkpoint), "name": checkpoint.name, "mtime": mtime})
    checkpoints.sort(key=lambda item: (item.get("mtime", -1), item.get("name", "").lower()), reverse=True)
    return checkpoints


def _http_json(method: str, url: str, *, payload: dict, headers: dict | None = None, timeout: int = 1800) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = body.strip()
        try:
            detail = json.loads(body)
            if isinstance(detail, dict):
                message = str((detail.get("error") or {}).get("message") or detail.get("message") or message)
        except Exception:
            pass
        raise RuntimeError(f"OpenRouter request failed ({exc.code}): {message}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc


def _openrouter_text_from_image(api_key: str, model_id: str, image_data_url: str, prompt: str) -> tuple[str, dict]:
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        detail = _http_json("POST", f"{OPENROUTER_API_ROOT}/chat/completions", payload=payload, headers=headers)
    except RuntimeError as exc:
        if "response_format" not in str(exc).lower():
            raise
        payload.pop("response_format", None)
        detail = _http_json("POST", f"{OPENROUTER_API_ROOT}/chat/completions", payload=payload, headers=headers)

    text_parts = []
    for choice in detail.get("choices", []) or []:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content.strip())
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text") or "").strip())
    text = "\n".join(part for part in text_parts if part).strip()
    if not text:
        raise RuntimeError("Planner did not return text.")
    return text, detail


def _safe_openrouter_response(detail: dict) -> dict:
    safe_detail = json.loads(json.dumps(detail))
    for choice in safe_detail.get("choices", []) or []:
        message = choice.get("message") or {}
        for image in message.get("images", []) or []:
            image_url = image.get("image_url") or image.get("imageUrl") or {}
            if isinstance(image_url, dict) and "url" in image_url:
                image_url["url"] = "[image data omitted]"
    return safe_detail


def _gemini_request_image(payload: dict, prompt: str, output_label: str) -> list[dict]:
    api_key = _resolve_openrouter_api_key(payload)
    if not api_key:
        raise ValueError("Enter an OpenRouter API key or save one in the Manager.")
    model_id = (payload.get("model_id") or "google/gemini-2.5-flash-image").strip()
    image_config = {"aspect_ratio": (payload.get("aspect_ratio") or "1:1").strip()}
    image_size = (payload.get("image_size") or "").strip()
    if image_size and model_id in {"google/gemini-3.1-flash-image-preview", "google/gemini-3-pro-image-preview"}:
        image_config["image_size"] = image_size

    message_content: str | list[dict] = prompt
    guide_image = (payload.get("guide_image") or "").strip()
    if guide_image:
        message_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": guide_image}},
        ]

    request_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": message_content}],
        "modalities": ["image", "text"],
        "stream": False,
        "image_config": image_config,
    }
    detail = _http_json(
        "POST",
        f"{OPENROUTER_API_ROOT}/chat/completions",
        payload=request_payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )

    text_parts = []
    image_urls = []
    finish_reasons = []
    native_finish_reasons = []
    for choice in detail.get("choices", []) or []:
        if choice.get("finish_reason"):
            finish_reasons.append(str(choice.get("finish_reason")))
        if choice.get("native_finish_reason"):
            native_finish_reasons.append(str(choice.get("native_finish_reason")))
        message = choice.get("message") or {}
        if message.get("content"):
            text_parts.append(str(message["content"]).strip())
        for image in message.get("images", []) or []:
            image_url = image.get("image_url") or image.get("imageUrl") or {}
            if isinstance(image_url, dict) and image_url.get("url"):
                image_urls.append(image_url["url"])

    if not image_urls:
        message = "Gemini Flash did not return an image."
        reason = (detail.get("error") or {}).get("message", "")
        if reason:
            message += f" {reason}"
        if native_finish_reasons:
            message += f" Provider finish reason: {', '.join(dict.fromkeys(native_finish_reasons))}."
        elif finish_reasons:
            message += f" Finish reason: {', '.join(dict.fromkeys(finish_reasons))}."
        if text_parts:
            message += f" Response: {' '.join(text_parts)[:500]}"
        raise RuntimeError(message)

    saved = []
    total = len(image_urls)
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    for index, image_url in enumerate(image_urls, start=1):
        mime_type, image_bytes = _decode_data_url(image_url)
        label = output_label if total == 1 else f"{output_label}-{index}"
        path = SETTINGS.output_dir / f"{uuid.uuid4().hex[:8]}-{_slugify(label, 'gemini')}{_image_suffix(mime_type)}"
        path.write_bytes(image_bytes)
        metadata = {
            "provider": "Gemini Flash",
            "mode": "gemini",
            "model_id": model_id,
            "prompt": prompt,
            "aspect_ratio": image_config.get("aspect_ratio"),
            "image_size": image_config.get("image_size", ""),
            "guide_image": bool(guide_image),
            "mime_type": mime_type,
            "text": [part for part in text_parts if part],
            "response": _safe_openrouter_response(detail),
            **_batch_metadata(payload, default_type="gemini", default_label="Gemini Variants", default_item_label=label),
        }
        metadata_path = path.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        saved.append(
            {
                "name": path.name,
                "path": str(path),
                "url": _output_url(path),
                "metadata_path": str(metadata_path),
                "batch_id": metadata.get("batch_id", ""),
                "batch_label": metadata.get("batch_label", ""),
                "batch_type": metadata.get("batch_type", ""),
                "item_label": metadata.get("item_label", ""),
                "item_index": metadata.get("item_index"),
                "item_total": metadata.get("item_total"),
                "metadata": metadata,
            }
        )
    return saved


def _extract_json_payload(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _normalize_bbox(value) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    normalized = []
    for item in value:
        try:
            normalized.append(max(0.0, min(1.0, float(item))))
        except Exception:
            return []
    return normalized


def _normalize_part_plan(raw_plan: dict, max_parts: int = 8) -> dict:
    raw_parts = raw_plan.get("parts") if isinstance(raw_plan, dict) else None
    if not isinstance(raw_parts, list):
        raise RuntimeError("Planner JSON did not contain a parts list.")
    parts = []
    seen = set()
    for index, raw_part in enumerate(raw_parts, start=1):
        if not isinstance(raw_part, dict):
            continue
        display_name = str(raw_part.get("display_name") or raw_part.get("name") or "").strip()
        part_id = _slugify(str(raw_part.get("id") or display_name), f"part-{index:02d}").replace("-", "_")
        if part_id in seen:
            part_id = f"{part_id}_{index:02d}"
        seen.add(part_id)
        if not display_name:
            display_name = part_id.replace("_", " ").title()
        extraction_prompt = str(raw_part.get("extraction_prompt") or raw_part.get("prompt") or "").strip()
        if not extraction_prompt:
            extraction_prompt = (
                f"Extract only the {display_name}. Remove unrelated body parts, heads, mannequins, labels, "
                "extra objects, and background elements."
            )
        try:
            priority = int(raw_part.get("priority") or index)
        except Exception:
            priority = index
        parts.append(
            {
                "id": part_id,
                "display_name": display_name,
                "category": str(raw_part.get("category") or "part").strip().lower(),
                "priority": priority,
                "selected": True,
                "symmetry": bool(raw_part.get("symmetry", False)),
                "normalized_bbox": _normalize_bbox(raw_part.get("normalized_bbox")),
                "extraction_prompt": extraction_prompt,
            }
        )
    if not parts:
        raise RuntimeError("Planner did not identify any extractable parts.")
    parts.sort(key=lambda item: (item.get("priority", 999), item.get("id", "")))
    return {"parts": parts[: max(1, int(max_parts or 8))]}


def _part_planning_prompt(guidance: str, *, base_face: bool, base_eyes: bool, eye_part: bool) -> str:
    face_rule = (
        "- For anatomy_base, keep facial features on the base body and match the source face structure.\n"
        if base_face
        else "- For anatomy_base, keep the head feature-neutral with no finished face details.\n"
    )
    eyes_rule = ""
    if base_face and base_eyes:
        eyes_rule = "- For anatomy_base, include finished eyes on the base body.\n"
    elif base_face:
        eyes_rule = "- For anatomy_base, keep face structure but do not include finished eyes.\n"
    eye_rule = (
        "- Include one reusable Eyeball part with category face_feature: one isolated spherical eyeball only.\n"
        if eye_part
        else ""
    )
    return (
        "Look at the master character reference image and plan separate asset extractions for a 3D game asset workflow.\n\n"
        "Return JSON only. Schema: {\"parts\":[{\"id\":\"short_slug\",\"display_name\":\"Name\",\"category\":\"anatomy_base | hair | clothing | armor | accessory | weapon | prop | face_feature\",\"priority\":1,\"normalized_bbox\":[0,0,1,1],\"extraction_prompt\":\"specific instruction\"}]}\n\n"
        "Rules:\n"
        "- Include one anatomy_base part first when a body/base mesh is visible or inferable.\n"
        "- For anatomy_base, ask for body/base mesh only, not the dressed character.\n"
        "- Include major garments, armor, hair, weapons, carried props, pouches, belts, and accessories that matter for 3D asset creation.\n"
        f"{face_rule}{eyes_rule}{eye_rule}"
        "- Do not include scenery, shadows, background decorations, labels, duplicate variants, or combined multi-item parts.\n"
        "- Prefer the most important 4 to 8 parts.\n"
        "- Each extraction_prompt must ask for exactly one isolated target item and remove unrelated body parts, heads, mannequins, labels, extra objects, and background.\n\n"
        f"Global extraction guidance: {(guidance or 'Preserve the source design, scale relationship, silhouette, materials, and media style.').strip()}"
    )


def _part_extraction_prompt(part: dict, payload: dict) -> str:
    display_name = part.get("display_name") or part.get("id") or "character part"
    category = (part.get("category") or "").strip().lower()
    instruction = (part.get("extraction_prompt") or "").strip()
    guidance = (payload.get("guidance") or "Preserve the master image design and media style.").strip()
    style_text = (payload.get("style_text") or "").strip() if payload.get("style_lock", True) else ""
    style_block = f"\nStyle lock: {style_text}" if style_text else ""
    symmetry = (
        "\nSymmetry lock: make the item left-right symmetrical and front-readable."
        if part.get("symmetry")
        else ""
    )
    if "eye" in f"{part.get('id','')} {display_name} {category}".lower():
        instruction = (
            "Create exactly one isolated spherical eyeball asset from the source character. Show only sclera, iris, pupil, cornea highlight, and painted surface detail. "
            "Do not include eyelids, skin, brow, socket, surrounding flesh, face, head, or hair."
        )
    return (
        "Using the master character reference image, create one clean isolated asset reference image.\n\n"
        f"Target part: {display_name}\n"
        f"Extraction instruction: {instruction}\n"
        f"Global guidance: {guidance}{style_block}{symmetry}\n\n"
        "Output rules:\n"
        "- Show exactly one centered target item.\n"
        "- Preserve the original design language, material details, color palette, and scale relationship.\n"
        "- Use a plain background only to isolate the item.\n"
        "- Do not create a parts sheet, grid, lineup, collage, catalog page, labels, text, scenery, or unrelated objects."
    )


def _gemini_generate_worker(payload: dict) -> dict:
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Enter an image-generation prompt first.")
        prompts = [prompt for _ in range(max(1, min(int(payload.get("variant_count") or 1), 8)))]

    outputs = []
    total = len(prompts)
    batch_id = str(payload.get("batch_id") or f"gemini-{uuid.uuid4().hex[:8]}").strip()
    for index, prompt in enumerate(prompts, start=1):
        progress_update(
            status="processing",
            stage="gemini_image",
            detail=f"Generating Gemini image {index}/{total}",
            progress_current=index - 1,
            progress_total=total,
            progress_percent=((index - 1) / total) * 100.0,
        )
        item_payload = {
            **payload,
            "batch_id": batch_id,
            "batch_label": payload.get("batch_label") or "Gemini Variants",
            "batch_type": payload.get("batch_type") or "gemini",
            "item_label": payload.get("item_label") or f"G {index}",
            "item_index": index,
            "item_total": total,
        }
        outputs.extend(_gemini_request_image(item_payload, prompt, f"gemini-{index}"))

    progress_update(
        status="idle",
        stage="complete",
        detail="Gemini generation complete",
        progress_current=total,
        progress_total=total,
        progress_percent=100.0,
        last_output_path=outputs[-1]["path"] if outputs else None,
    )
    return {"status": "ok", "batch_id": batch_id, "outputs": outputs}


def _part_plan_worker(payload: dict) -> dict:
    api_key = _resolve_openrouter_api_key(payload)
    source_image = (payload.get("source_image") or "").strip()
    if not api_key:
        raise ValueError("Enter an OpenRouter API key or save one in the Manager.")
    if not source_image:
        raise ValueError("Choose a source image first.")
    max_parts = max(1, min(int(payload.get("max_parts") or 8), 16))
    planner_model = (payload.get("planner_model") or "google/gemini-2.5-flash").strip()
    prompt = _part_planning_prompt(
        payload.get("guidance") or "",
        base_face=bool(payload.get("base_face")),
        base_eyes=bool(payload.get("base_eyes")),
        eye_part=bool(payload.get("eye_part")),
    )
    progress_update(status="processing", stage="planning_parts", detail="Planning character parts", progress_percent=12.0)
    response_text, detail = _openrouter_text_from_image(api_key, planner_model, source_image, prompt)
    plan = _normalize_part_plan(_extract_json_payload(response_text), max_parts=max_parts)
    metadata = {
        "provider": "Gemini Flash",
        "mode": "part_plan",
        "planner_model": planner_model,
        "parts": plan["parts"],
        "guidance": payload.get("guidance") or "",
        "response_text": response_text,
        "response": _safe_openrouter_response(detail),
    }
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = SETTINGS.output_dir / f"{uuid.uuid4().hex[:8]}-parts-plan.json"
    plan_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    progress_update(status="idle", stage="complete", detail="Part plan ready", progress_percent=100.0)
    return {"status": "ok", "plan_path": str(plan_path), "parts": plan["parts"]}


def _part_extract_worker(payload: dict) -> dict:
    source_image = (payload.get("source_image") or "").strip()
    parts = payload.get("parts")
    if not source_image:
        raise ValueError("Choose a source image first.")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Plan and select at least one part first.")
    payload = {**payload, "guide_image": source_image}
    outputs = []
    total = len(parts)
    batch_id = str(payload.get("batch_id") or f"parts-{uuid.uuid4().hex[:8]}").strip()
    for index, part in enumerate(parts, start=1):
        name = part.get("display_name") or part.get("id") or f"Part {index}"
        progress_update(
            status="processing",
            stage="extracting_parts",
            detail=f"Extracting {index}/{total}: {name}",
            progress_current=index - 1,
            progress_total=total,
            progress_percent=((index - 1) / total) * 100.0,
        )
        label = f"part-{index:02d}-{_slugify(name, f'part-{index:02d}')}"
        item_payload = {
            **payload,
            "batch_id": batch_id,
            "batch_label": payload.get("batch_label") or "Image Parts",
            "batch_type": payload.get("batch_type") or "parts",
            "item_label": name,
            "item_index": index,
            "item_total": total,
        }
        outputs.extend(_gemini_request_image(item_payload, _part_extraction_prompt(part, payload), label))
    progress_update(
        status="idle",
        stage="complete",
        detail="Part extraction complete",
        progress_current=total,
        progress_total=total,
        progress_percent=100.0,
        last_output_path=outputs[-1]["path"] if outputs else None,
    )
    return {"status": "ok", "batch_id": batch_id, "outputs": outputs}


def _resize_init_image(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _normalize_request(payload: GenerateRequest) -> GenerateRequest:
    width = _coerce_dimension(payload.width, maximum=SETTINGS.max_width, label="width")
    height = _coerce_dimension(payload.height, maximum=SETTINGS.max_height, label="height")
    steps = payload.steps or SETTINGS.default_steps
    guidance_scale = payload.guidance_scale if payload.guidance_scale is not None else SETTINGS.default_guidance_scale
    strength = payload.strength if payload.strength is not None else SETTINGS.default_strength
    lora_path = (payload.lora_path or "").strip() or None
    lora_scale = payload.lora_scale if lora_path is not None else None
    if lora_path is not None and lora_scale is None:
        lora_scale = 1.0

    if steps <= 0:
        raise ValueError("steps must be greater than zero.")
    if payload.mode == "img2img" and not payload.image:
        raise ValueError("img2img mode requires an input image.")
    if payload.mode == "img2img" and not MODEL_MANAGER.supports_img2img(payload.model_id):
        raise ValueError("Current runtime supports txt2img only.")
    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")
    if lora_scale is not None and lora_scale < 0.0:
        raise ValueError("lora_scale must be zero or greater.")

    return payload.model_copy(
        update={
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "negative_prompt": payload.negative_prompt or SETTINGS.default_negative_prompt,
            "lora_path": lora_path,
            "lora_scale": lora_scale,
        }
    )


def _generate(payload: GenerateRequest) -> GenerateResponse:
    started_at = perf_counter()
    _log_stage(
        "generate.begin",
        mode=payload.mode,
        steps=payload.steps,
        width=payload.width,
        height=payload.height,
        lora=bool(payload.lora_path),
    )
    progress_update(
        status="processing",
        stage="loading_model",
        detail="Loading or reusing model",
        model_id=payload.model_id or SETTINGS.default_model_id,
        progress_current=0,
        progress_total=3,
        progress_percent=0.0,
    )

    if payload.mode == "txt2img":
        progress_update(
            status="processing",
            stage="generating_image",
            detail="Running txt2img",
            progress_current=1,
            progress_total=3,
            progress_percent=33.0,
        )
        _log_stage("txt2img.call.begin")
        image, model_id = MODEL_MANAGER.generate_text_to_image(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            width=payload.width,
            height=payload.height,
            steps=payload.steps,
            guidance_scale=payload.guidance_scale,
            seed=payload.seed,
            model_id=payload.model_id,
            lora_path=payload.lora_path,
            lora_scale=payload.lora_scale,
        )
        _log_stage("txt2img.call.end", elapsed=f"{perf_counter() - started_at:.2f}s")
    else:
        init_image = _decode_base64_image(payload.image or "")
        init_image = _resize_init_image(init_image, payload.width, payload.height)
        progress_update(
            status="processing",
            stage="generating_image",
            detail="Running img2img",
            progress_current=1,
            progress_total=3,
            progress_percent=33.0,
        )
        _log_stage("img2img.call.begin")
        image, model_id = MODEL_MANAGER.generate_image_to_image(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            image=init_image,
            width=payload.width,
            height=payload.height,
            steps=payload.steps,
            guidance_scale=payload.guidance_scale,
            strength=payload.strength,
            seed=payload.seed,
            model_id=payload.model_id,
            lora_path=payload.lora_path,
            lora_scale=payload.lora_scale,
        )
        _log_stage("img2img.call.end", elapsed=f"{perf_counter() - started_at:.2f}s")

    progress_update(
        status="processing",
        stage="saving_output",
        detail="Saving output image",
        model_id=model_id,
        progress_current=2,
        progress_total=3,
        progress_percent=66.0,
    )

    metadata = {
        "backend": "Nymphs2D2",
        "version": VERSION,
        "worker_id": WORKER_ID,
        "runtime": MODEL_MANAGER.loaded_runtime or SETTINGS.runtime,
        "mode": payload.mode,
        "model_id": model_id,
        "prompt": payload.prompt,
        "negative_prompt": payload.negative_prompt,
        "width": payload.width,
        "height": payload.height,
        "steps": payload.steps,
        "guidance_scale": payload.guidance_scale,
        "seed": payload.seed,
        "lora_path": payload.lora_path,
        "lora_scale": payload.lora_scale,
        "strength": payload.strength,
        **_batch_metadata(
            payload.model_dump() if hasattr(payload, "model_dump") else payload.dict(),
            default_type="zimage",
            default_label="Z-Image Variants",
            default_item_label=payload.item_label or "Z-Image",
        ),
    }
    _log_stage("save.begin", output_dir=SETTINGS.output_dir)
    output_path, metadata_path = save_image_and_metadata(
        image,
        SETTINGS.output_dir,
        mode=payload.mode,
        prompt=payload.prompt,
        metadata=metadata,
    )
    _log_stage(
        "save.end",
        output_path=output_path,
        metadata_path=metadata_path,
        elapsed=f"{perf_counter() - started_at:.2f}s",
    )

    progress_update(
        status="idle",
        stage="complete",
        detail="Generation complete",
        model_id=model_id,
        progress_current=3,
        progress_total=3,
        progress_percent=100.0,
        last_output_path=str(output_path),
    )
    progress_reset()
    progress_update(last_output_path=str(output_path), model_id=model_id)

    _log_stage("generate.end", elapsed=f"{perf_counter() - started_at:.2f}s")
    return GenerateResponse(
        status="ok",
        worker_id=WORKER_ID,
        mode=payload.mode,
        model_id=model_id,
        output_path=str(output_path),
        metadata_path=str(metadata_path),
        url=_output_url(output_path),
        batch_id=metadata.get("batch_id"),
        batch_label=metadata.get("batch_label"),
        batch_type=metadata.get("batch_type"),
        item_label=metadata.get("item_label"),
        item_index=metadata.get("item_index"),
        item_total=metadata.get("item_total"),
    )


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health_check():
    return JSONResponse({"status": "healthy", "worker_id": WORKER_ID}, status_code=200)


@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def root_ui():
    return HTMLResponse(NYMPH_UI_PATH.read_text(encoding="utf-8"))


@app.get("/nymph", response_class=HTMLResponse, tags=["ui"])
async def nymph_ui():
    return HTMLResponse(NYMPH_UI_PATH.read_text(encoding="utf-8"))


@app.get("/server_info", response_model=ServerInfoResponse, tags=["status"])
async def server_info():
    supported_modes = MODEL_MANAGER.supported_modes()
    return ServerInfoResponse(
        backend="Nymphs2D2",
        version=VERSION,
        worker_id=WORKER_ID,
        configured_model_id=SETTINGS.default_model_id,
        loaded_model_id=MODEL_MANAGER.loaded_model_id,
        device=SETTINGS.device,
        dtype=SETTINGS.dtype,
        output_dir=str(SETTINGS.output_dir),
        supported_modes=supported_modes,
        extra={
            "max_width": SETTINGS.max_width,
            "max_height": SETTINGS.max_height,
            "runtime": MODEL_MANAGER.loaded_runtime or SETTINGS.runtime,
            "configured_runtime": SETTINGS.runtime,
            "supports_lora": MODEL_MANAGER.supports_lora(),
            "nunchaku_rank": SETTINGS.nunchaku_rank,
            "nunchaku_precision": SETTINGS.nunchaku_precision,
            **MODEL_MANAGER.loaded_runtime_extra,
        },
    )


@app.get("/active_task", response_model=ActiveTaskResponse, tags=["status"])
async def active_task():
    return ActiveTaskResponse(**progress_snapshot())


@app.get("/outputs/{relative_path:path}", tags=["ui"])
async def output_file(relative_path: str):
    path = _safe_output_path(relative_path)
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/outputs", tags=["ui"])
async def recent_outputs(limit: int = 80):
    return JSONResponse({"outputs": _recent_outputs(limit)})


@app.post("/api/outputs/clear", tags=["ui"])
async def clear_outputs():
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    removed = 0
    for path in SETTINGS.output_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES and path.suffix.lower() != ".json":
            continue
        try:
            path.unlink()
            removed += 1
        except Exception:
            pass
    return JSONResponse({"status": "ok", "removed": removed, "outputs": []})


@app.get("/api/presets", tags=["ui"])
async def presets():
    return JSONResponse({kind: _load_user_presets(kind) for kind in sorted(PRESET_KINDS)})


@app.post("/api/presets/{kind}", tags=["ui"])
async def save_preset(kind: str, request: FastAPIRequest):
    payload = await request.json()
    name = str(payload.get("name") or payload.get("label") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Preset name is required.")
    preset_id = _safe_slug(str(payload.get("id") or name), "preset")
    data = {
        "name": name,
        "kind": kind,
        "description": str(payload.get("description") or "").strip(),
    }
    if kind == "settings":
        values = payload.get("values")
        if not isinstance(values, dict):
            raise HTTPException(status_code=400, detail="Settings preset values are required.")
        data["values"] = values
    else:
        data["prompt"] = str(payload.get("prompt") or "").strip()
        if not data["prompt"]:
            raise HTTPException(status_code=400, detail="Preset prompt is required.")
    path = _safe_preset_path(kind, preset_id)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return JSONResponse({"status": "ok", "preset": {"id": path.stem, **data}})


@app.delete("/api/presets/{kind}/{preset_id}", tags=["ui"])
async def delete_preset(kind: str, preset_id: str):
    path = _safe_preset_path(kind, preset_id)
    if path.is_file():
        path.unlink()
    return JSONResponse({"status": "ok"})


@app.get("/api/loras", tags=["ui"])
async def list_loras():
    runs = _iter_lora_runs()
    checkpoints = {run["id"]: _iter_lora_checkpoints(run["id"]) for run in runs[:80]}
    return JSONResponse({"runs": runs, "checkpoints": checkpoints})


@app.get("/api/openrouter/status", tags=["ui"])
async def openrouter_status():
    return JSONResponse({"configured": bool(_resolve_openrouter_api_key())})


@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
async def generate(payload: GenerateRequest):
    try:
        normalized = _normalize_request(payload)
    except ValueError as exc:
        print(f"[nymphs:zimage:error] normalize.value_error detail={exc}", flush=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return await run_in_threadpool(_generate, normalized)
    except HTTPException:
        raise
    except ValueError as exc:
        print(f"[nymphs:zimage:error] generate.value_error detail={exc}", flush=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        progress_update(status="error", stage="failed", detail=detail)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {detail}") from exc


@app.post("/api/gemini/generate", tags=["gemini"])
async def gemini_generate(request: FastAPIRequest):
    payload = await request.json()
    try:
        return JSONResponse(await run_in_threadpool(_gemini_generate_worker, payload))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        progress_update(status="error", stage="failed", detail=detail)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Gemini generation failed: {detail}") from exc


@app.post("/api/parts/plan", tags=["gemini"])
async def plan_parts(request: FastAPIRequest):
    payload = await request.json()
    try:
        return JSONResponse(await run_in_threadpool(_part_plan_worker, payload))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        progress_update(status="error", stage="failed", detail=detail)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Part planning failed: {detail}") from exc


@app.post("/api/parts/extract", tags=["gemini"])
async def extract_parts(request: FastAPIRequest):
    payload = await request.json()
    try:
        return JSONResponse(await run_in_threadpool(_part_extract_worker, payload))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        progress_update(status="error", stage="failed", detail=detail)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Part extraction failed: {detail}") from exc


def main():
    parser = argparse.ArgumentParser(description="Run the Nymphs2D2 API server.")
    parser.add_argument("--host", default=SETTINGS.host)
    parser.add_argument("--port", type=int, default=SETTINGS.port)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
