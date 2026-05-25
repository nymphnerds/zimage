from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_BRAIN_BASE_URL = "http://127.0.0.1:8000/v1"


def brain_base_url() -> str:
    return (os.getenv("NYMPHS_BRAIN_LLM_API_BASE_URL") or DEFAULT_BRAIN_BASE_URL).rstrip("/")


def _brain_model_id(timeout: float = 10.0) -> str:
    try:
        with urlopen(f"{brain_base_url()}/models", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError("Brain is required for local parts planning. Start Brain with a vision-capable model first.") from exc

    for item in payload.get("data", []) or []:
        if isinstance(item, dict):
            model_id = str(item.get("id") or item.get("name") or item.get("model") or "").strip()
            if model_id:
                return model_id
    return "local-model"


def brain_status() -> dict:
    root = Path(os.getenv("NYMPHS_BRAIN_INSTALL_ROOT") or Path.home() / "Nymphs-Brain")
    installed = (root / ".nymph-module-version").is_file()
    model_configured = False
    start_script = root / "bin" / "lms-start"
    if start_script.is_file():
        for line in start_script.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MODEL_KEY="):
                value = line.split("=", 1)[1].strip().strip('"')
                model_configured = bool(value and value != "none")
                break
    reachable = False
    if installed and model_configured:
        try:
            _brain_model_id(timeout=1.0)
            reachable = True
        except Exception:
            reachable = False
    return {
        "label": "Brain Vision",
        "ready": installed and model_configured and reachable,
        "installed": installed,
        "model_configured": model_configured,
        "reachable": reachable,
        "base_url": brain_base_url(),
        "tasks": ["caption", "parts_plan"],
    }


def brain_chat_image(image_data_url: str, prompt: str, *, temperature: float = 0.1, max_tokens: int = 1600) -> str:
    status = brain_status()
    if not status["installed"]:
        raise RuntimeError("Brain is required for local parts planning. Install Brain and configure a vision-capable model.")
    if not status["model_configured"]:
        raise RuntimeError("Brain is required for local parts planning. Configure a vision-capable Brain model first.")
    model_id = _brain_model_id()
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
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = Request(
        f"{brain_base_url()}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode("utf-8", errors="replace"))
    texts = []
    for choice in body.get("choices", []) or []:
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, str):
            texts.append(content.strip())
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"].strip())
    text = "\n".join(item for item in texts if item).strip()
    if not text:
        raise RuntimeError("Brain returned no vision text.")
    return text
