from __future__ import annotations

import gc
from pathlib import Path
from threading import Lock

from PIL import Image


FLUX_DEV_MODEL_ID = "black-forest-labs/FLUX.1-dev"
FLUX_KONTEXT_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX_DEV_WEIGHT_REPO = "nunchaku-tech/nunchaku-flux.1-dev"
FLUX_KONTEXT_WEIGHT_REPO = "nunchaku-tech/nunchaku-flux.1-kontext-dev"
FLUX_RANK = 32


def _repo_cache_root(cache_dir: Path | None, repo_id: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"models--{repo_id.replace('/', '--')}"


def _snapshot_ready(cache_dir: Path | None, repo_id: str, required_paths: list[str]) -> bool:
    repo_root = _repo_cache_root(cache_dir, repo_id)
    if repo_root is None:
        return False
    ref_file = repo_root / "refs" / "main"
    snapshots_dir = repo_root / "snapshots"
    if ref_file.exists():
        snapshot = ref_file.read_text(encoding="utf-8", errors="ignore").strip()
        candidates = [snapshots_dir / snapshot] if snapshot else []
    else:
        candidates = [path for path in snapshots_dir.iterdir()] if snapshots_dir.exists() else []
    for snapshot_dir in candidates:
        if snapshot_dir.is_dir() and all((snapshot_dir / item).exists() for item in required_paths):
            return True
    return False


def _hf_file_ready(cache_dir: Path | None, repo_id: str, filename: str) -> bool:
    return _hf_cached_file(cache_dir, repo_id, filename) is not None


def _hf_cached_file(cache_dir: Path | None, repo_id: str, filename: str) -> Path | None:
    repo_root = _repo_cache_root(cache_dir, repo_id)
    snapshots_dir = repo_root / "snapshots" if repo_root is not None else None
    if snapshots_dir is None or not snapshots_dir.exists():
        return None
    for candidate in snapshots_dir.glob(f"*/{filename}"):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _empty_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _torch_dtype(settings):
    import torch

    dtype_name = (settings.dtype or "bfloat16").strip().lower()
    if dtype_name in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    return torch.bfloat16


def _get_precision(settings) -> str:
    precision = (settings.nunchaku_precision or "auto").strip().lower()
    if precision != "auto":
        return precision
    try:
        from nunchaku.utils import get_precision

        return get_precision(precision="auto", device=settings.device)
    except Exception:
        return "int4"


def _flux_weight_filename(model_name: str, settings) -> str:
    precision = _get_precision(settings)
    return f"svdq-{precision}_r{FLUX_RANK}-{model_name}.safetensors"


def _flux_transformer_class():
    import nunchaku

    return getattr(nunchaku, "NunchakuFluxTransformer2dModel", None) or getattr(
        nunchaku, "NunchakuFluxTransformer2DModelV2"
    )


class ImageServiceCoordinator:
    def __init__(self, settings, zimage_manager):
        self.settings = settings
        self.zimage_manager = zimage_manager
        self._lock = Lock()
        self._flux_dev = None
        self._flux_kontext = None
        self._flux_dev_model_id = None
        self._flux_kontext_model_id = None

    @staticmethod
    def normalize_provider(provider: str | None, mode: str = "txt2img") -> str:
        normalized = (provider or "zimage").strip().lower().replace("_", "-")
        if normalized in {"", "zimage", "z-image"}:
            return "zimage"
        if normalized in {"flux", "flux-dev", "flux-txt2img"}:
            return "flux_kontext" if mode == "img2img" else "flux_dev"
        if normalized in {"flux-kontext", "flux-edit", "flux-img2img", "kontext"}:
            return "flux_kontext"
        raise ValueError(f"Unknown image provider: {provider}.")

    def unload_all(self, keep: str | None = None) -> None:
        if keep != "zimage" and hasattr(self.zimage_manager, "_unload_pipelines"):
            self.zimage_manager._unload_pipelines()
        if keep != "flux_dev":
            self._flux_dev = None
            self._flux_dev_model_id = None
        if keep != "flux_kontext":
            self._flux_kontext = None
            self._flux_kontext_model_id = None
        _empty_cuda_cache()

    def _flux_dev_ready(self) -> bool:
        filename = _flux_weight_filename("flux.1-dev", self.settings)
        return _snapshot_ready(
            self.settings.hf_cache_dir,
            FLUX_DEV_MODEL_ID,
            ["model_index.json"],
        ) and _hf_file_ready(self.settings.hf_cache_dir, FLUX_DEV_WEIGHT_REPO, filename)

    def _flux_kontext_ready(self) -> bool:
        filename = _flux_weight_filename("flux.1-kontext-dev", self.settings)
        return _snapshot_ready(
            self.settings.hf_cache_dir,
            FLUX_KONTEXT_MODEL_ID,
            ["model_index.json"],
        ) and _hf_file_ready(self.settings.hf_cache_dir, FLUX_KONTEXT_WEIGHT_REPO, filename)

    def providers_info(self, supported_modes: list[str], supports_lora: bool) -> dict:
        runtime = self.zimage_manager.loaded_runtime or self.settings.runtime
        zimage_loaded = self.zimage_manager.loaded_model_id is not None
        flux_dev_filename = _flux_weight_filename("flux.1-dev", self.settings)
        flux_kontext_filename = _flux_weight_filename("flux.1-kontext-dev", self.settings)
        return {
            "zimage": {
                "label": "Z-Image Turbo",
                "ready": True,
                "loaded": zimage_loaded,
                "model_id": self.settings.default_model_id,
                "loaded_model_id": self.zimage_manager.loaded_model_id,
                "modes": supported_modes,
                "supports_txt2img": True,
                "supports_img2img": "img2img" in supported_modes,
                "supports_reference_edit": False,
                "supports_lora": supports_lora,
                "supports_parts_extract": False,
                "defaults": {
                    "width": 1024,
                    "height": 1024,
                    "steps": self.settings.default_steps,
                    "guidance_scale": self.settings.default_guidance_scale,
                    "strength": self.settings.default_strength,
                },
                "lora_family": "zimage",
                "memory_mode": runtime,
            },
            "flux_dev": {
                "label": "FLUX.1-dev",
                "ready": self._flux_dev_ready(),
                "loaded": self._flux_dev is not None,
                "model_id": FLUX_DEV_MODEL_ID,
                "weight_repo": FLUX_DEV_WEIGHT_REPO,
                "weight_file": flux_dev_filename,
                "modes": ["txt2img"],
                "supports_txt2img": True,
                "supports_img2img": False,
                "supports_reference_edit": False,
                "supports_lora": False,
                "supports_parts_extract": False,
                "defaults": {"width": 1024, "height": 1024, "steps": 20, "guidance_scale": 3.5},
                "lora_family": "flux",
                "memory_mode": "single_provider",
            },
            "flux_kontext": {
                "label": "FLUX.1-Kontext-dev",
                "ready": self._flux_kontext_ready(),
                "loaded": self._flux_kontext is not None,
                "model_id": FLUX_KONTEXT_MODEL_ID,
                "weight_repo": FLUX_KONTEXT_WEIGHT_REPO,
                "weight_file": flux_kontext_filename,
                "modes": ["img2img", "reference_edit", "parts_extract"],
                "supports_txt2img": False,
                "supports_img2img": True,
                "supports_reference_edit": True,
                "supports_lora": False,
                "supports_parts_extract": True,
                "defaults": {"width": 1024, "height": 1024, "steps": 20, "guidance_scale": 2.5, "strength": 0.75},
                "lora_family": "flux",
                "memory_mode": "single_provider",
            },
        }

    def _load_flux_dev(self):
        if not self._flux_dev_ready():
            raise RuntimeError("Fetch FLUX.1-dev before generating with provider=flux_dev.")
        if self._flux_dev is not None:
            return self._flux_dev
        self.unload_all(keep="flux_dev")
        import torch
        from diffusers import FluxPipeline

        transformer_cls = _flux_transformer_class()
        filename = _flux_weight_filename("flux.1-dev", self.settings)
        weight_path = _hf_cached_file(self.settings.hf_cache_dir, FLUX_DEV_WEIGHT_REPO, filename)
        if weight_path is None:
            raise RuntimeError("Fetch FLUX.1-dev before generating with provider=flux_dev.")
        transformer = transformer_cls.from_pretrained(
            str(weight_path),
            torch_dtype=_torch_dtype(self.settings),
        )
        pipe = FluxPipeline.from_pretrained(
            FLUX_DEV_MODEL_ID,
            transformer=transformer,
            torch_dtype=_torch_dtype(self.settings),
            cache_dir=str(self.settings.hf_cache_dir) if self.settings.hf_cache_dir else None,
            local_files_only=True,
            token=self.settings.hf_token,
        )
        if self.settings.device == "cuda" and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        elif self.settings.device:
            pipe = pipe.to(self.settings.device)
        self._flux_dev = pipe
        self._flux_dev_model_id = FLUX_DEV_MODEL_ID
        return pipe

    def _load_flux_kontext(self):
        if not self._flux_kontext_ready():
            raise RuntimeError("Fetch FLUX.1-Kontext-dev before generating with provider=flux_kontext.")
        if self._flux_kontext is not None:
            return self._flux_kontext
        self.unload_all(keep="flux_kontext")
        from diffusers import FluxKontextPipeline

        transformer_cls = _flux_transformer_class()
        filename = _flux_weight_filename("flux.1-kontext-dev", self.settings)
        weight_path = _hf_cached_file(self.settings.hf_cache_dir, FLUX_KONTEXT_WEIGHT_REPO, filename)
        if weight_path is None:
            raise RuntimeError("Fetch FLUX.1-Kontext-dev before generating with provider=flux_kontext.")
        transformer = transformer_cls.from_pretrained(
            str(weight_path),
            torch_dtype=_torch_dtype(self.settings),
        )
        pipe = FluxKontextPipeline.from_pretrained(
            FLUX_KONTEXT_MODEL_ID,
            transformer=transformer,
            torch_dtype=_torch_dtype(self.settings),
            cache_dir=str(self.settings.hf_cache_dir) if self.settings.hf_cache_dir else None,
            local_files_only=True,
            token=self.settings.hf_token,
        )
        if self.settings.device == "cuda" and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        elif self.settings.device:
            pipe = pipe.to(self.settings.device)
        self._flux_kontext = pipe
        self._flux_kontext_model_id = FLUX_KONTEXT_MODEL_ID
        return pipe

    def _build_generator(self, seed: int | None):
        if seed is None:
            return None
        import torch

        device = self.settings.device if self.settings.device != "mps" else "cpu"
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def generate_text_to_image(self, payload, progress_callback=None):
        with self._lock:
            pipe = self._load_flux_dev()
            steps = int(payload.steps or 20)
            kwargs = {
                "prompt": payload.prompt,
                "width": payload.width,
                "height": payload.height,
                "num_inference_steps": steps,
                "guidance_scale": payload.guidance_scale if payload.guidance_scale is not None else 3.5,
                "generator": self._build_generator(payload.seed),
            }
            if progress_callback is not None:
                def _on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
                    progress_callback(int(step_index) + 1, steps)
                    return callback_kwargs

                kwargs["callback_on_step_end"] = _on_step_end
            result = pipe(**kwargs)
            return result.images[0], FLUX_DEV_MODEL_ID

    def generate_image_to_image(self, payload, image: Image.Image, progress_callback=None):
        with self._lock:
            pipe = self._load_flux_kontext()
            steps = int(payload.steps or 20)
            kwargs = {
                "prompt": payload.prompt,
                "image": image,
                "width": payload.width,
                "height": payload.height,
                "num_inference_steps": steps,
                "guidance_scale": payload.guidance_scale if payload.guidance_scale is not None else 2.5,
                "generator": self._build_generator(payload.seed),
            }
            if progress_callback is not None:
                def _on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
                    progress_callback(int(step_index) + 1, steps)
                    return callback_kwargs

                kwargs["callback_on_step_end"] = _on_step_end
            result = pipe(**kwargs)
            return result.images[0], FLUX_KONTEXT_MODEL_ID
