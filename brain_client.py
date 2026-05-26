from __future__ import annotations

from contextlib import contextmanager
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_BRAIN_BASE_URL = "http://127.0.0.1:8000/v1"
_BRAIN_VISION_LOCK = threading.Lock()
_VISION_MODEL_MARKERS = (
    "-vl-",
    "vision",
    "llava",
    "minicpmv",
    "internvl",
    "cogvlm",
    "pixtral",
    "glm4v",
    "gemma4v",
    "qwen2vl",
    "qwen3vl",
    "molmo",
    "smolvlm",
)


def brain_base_url() -> str:
    return (os.getenv("NYMPHS_BRAIN_LLM_API_BASE_URL") or DEFAULT_BRAIN_BASE_URL).rstrip("/")


def _brain_install_root() -> Path:
    return Path(
        os.getenv("NYMPHS_BRAIN_INSTALL_ROOT")
        or os.getenv("BRAIN_INSTALL_ROOT")
        or Path.home() / "Nymphs-Brain"
    ).expanduser()


def _brain_start_script(root: Path) -> Path:
    return root / "bin" / "lms-start"


def _brain_stop_script(root: Path) -> Path:
    return root / "bin" / "lms-stop"


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


def _brain_api_ready(timeout: float = 2.0) -> bool:
    try:
        _brain_model_id(timeout=timeout)
        return True
    except Exception:
        return False


def _read_model_key(start_script: Path) -> str:
    if not start_script.is_file():
        return ""
    for line in start_script.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MODEL_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    return ""


def _write_model_key(start_script: Path, model_key: str) -> None:
    text = start_script.read_text(encoding="utf-8", errors="ignore")
    replacement = f'MODEL_KEY="{model_key}"'
    if re.search(r"^MODEL_KEY=.*$", text, flags=re.MULTILINE):
        text = re.sub(r"^MODEL_KEY=.*$", replacement, text, count=1, flags=re.MULTILINE)
    else:
        text = f"{replacement}\n{text}"
    start_script.write_text(text, encoding="utf-8")
    start_script.chmod(start_script.stat().st_mode | 0o111)


def _model_is_probably_vision(model_key: str) -> bool:
    normalized = (model_key or "").strip().lower()
    return any(marker in normalized for marker in _VISION_MODEL_MARKERS)


def _model_slug(model_key: str) -> str:
    model_name = (model_key or "").strip().split("/")[-1]
    model_name = re.sub(r"-gguf$", "", model_name, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", model_name.lower())


def _find_model_dir_for_key(root: Path, model_key: str) -> Path | None:
    models_dir = root / "models"
    slug = _model_slug(model_key)
    if not slug or not models_dir.is_dir():
        return None
    for model_dir in sorted(path for path in models_dir.glob("*/*") if path.is_dir()):
        comparable = re.sub(r"-gguf", "", str(model_dir).lower())
        comparable = re.sub(r"[^a-z0-9]", "", comparable)
        if slug in comparable:
            return model_dir
    return None


def _model_dir_is_vision_ready(model_dir: Path | None) -> bool:
    if model_dir is None or not model_dir.is_dir():
        return False
    has_model = any(path.is_file() and path.suffix.lower() == ".gguf" and "mmproj" not in path.name.lower() for path in model_dir.iterdir())
    has_mmproj = any(path.is_file() and "mmproj" in path.name.lower() for path in model_dir.iterdir())
    return has_model and has_mmproj


def _find_compatible_vision_model_key(root: Path) -> str:
    models_dir = root / "models"
    if not models_dir.is_dir():
        return ""
    for model_dir in sorted(path for path in models_dir.glob("*/*") if path.is_dir()):
        candidate_key = f"{model_dir.parent.name}/{model_dir.name}"
        if _model_is_probably_vision(candidate_key) and _model_dir_is_vision_ready(model_dir):
            return candidate_key
    return ""


def _start_brain(start_script: Path) -> None:
    result = subprocess.run(
        [str(start_script)],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Brain vision server failed to start. {detail[:800]}")
    for _ in range(30):
        if _brain_api_ready(timeout=2.0):
            return
        time.sleep(1)
    raise RuntimeError("Brain vision server started but did not become reachable.")


def _stop_brain(stop_script: Path) -> None:
    subprocess.run(
        [str(stop_script)],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


@contextmanager
def _brain_vision_session():
    root = _brain_install_root()
    start_script = _brain_start_script(root)
    stop_script = _brain_stop_script(root)
    if not (root / ".nymph-module-version").is_file():
        raise RuntimeError("Brain is required for local parts planning. Install Brain first.")
    if not start_script.is_file() or not os.access(start_script, os.X_OK):
        raise RuntimeError(f"Brain start script is missing: {start_script}")
    if not stop_script.is_file() or not os.access(stop_script, os.X_OK):
        raise RuntimeError(f"Brain stop script is missing: {stop_script}")

    with _BRAIN_VISION_LOCK:
        original_script = start_script.read_text(encoding="utf-8", errors="ignore")
        original_model_key = _read_model_key(start_script)
        selected_model_key = ""
        original_model_dir = _find_model_dir_for_key(root, original_model_key)
        original_server_running = _brain_api_ready(timeout=2.0)
        started_temporary_server = False

        if _model_is_probably_vision(original_model_key) and _model_dir_is_vision_ready(original_model_dir):
            selected_model_key = original_model_key
        if not selected_model_key:
            selected_model_key = _find_compatible_vision_model_key(root)
        if not selected_model_key:
            raise RuntimeError(
                f"Brain could not find a downloaded vision model with a matching mmproj file under {root / 'models'}. "
                "Download a Brain vision model such as Qwen2.5-VL first."
            )

        try:
            if not original_server_running:
                _write_model_key(start_script, selected_model_key)
                _start_brain(start_script)
                started_temporary_server = True
            elif selected_model_key != original_model_key:
                _stop_brain(stop_script)
                _write_model_key(start_script, selected_model_key)
                _start_brain(start_script)
                started_temporary_server = True
            yield
        finally:
            if start_script.exists():
                start_script.write_text(original_script, encoding="utf-8")
                start_script.chmod(start_script.stat().st_mode | 0o111)
            if started_temporary_server:
                _stop_brain(stop_script)
            if original_server_running and not _brain_api_ready(timeout=2.0):
                _start_brain(start_script)


def brain_status() -> dict:
    root = _brain_install_root()
    installed = (root / ".nymph-module-version").is_file()
    start_script = _brain_start_script(root)
    configured_model = _read_model_key(start_script)
    model_configured = bool(configured_model and configured_model != "none")
    configured_model_dir = _find_model_dir_for_key(root, configured_model)
    configured_vision_ready = _model_is_probably_vision(configured_model) and _model_dir_is_vision_ready(configured_model_dir)
    available_vision_model = configured_model if configured_vision_ready else _find_compatible_vision_model_key(root)
    reachable = False
    if installed and model_configured:
        try:
            _brain_model_id(timeout=1.0)
            reachable = True
        except Exception:
            reachable = False
    return {
        "label": "Brain Vision",
        "ready": installed and bool(available_vision_model),
        "installed": installed,
        "model_configured": model_configured,
        "reachable": reachable,
        "vision_model_available": bool(available_vision_model),
        "vision_model": available_vision_model,
        "auto_start": True,
        "base_url": brain_base_url(),
        "tasks": ["caption", "parts_plan"],
    }


def brain_chat_image(image_data_url: str, prompt: str, *, temperature: float = 0.1, max_tokens: int = 1600) -> str:
    with _brain_vision_session():
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
        try:
            with urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8", errors="replace"))
        except URLError as exc:
            raise RuntimeError(f"Brain vision endpoint was not reachable: {exc}") from exc
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
