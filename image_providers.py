from __future__ import annotations

import gc
from pathlib import Path
from threading import Lock

from PIL import Image


QWEN_EDIT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
QWEN_EDIT_WEIGHT_REPO = "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511"
QWEN_EDIT_WEIGHT_FILES = [
    "nunchaku_qwen_image_edit_2511_balance_int4.safetensors",
    "nunchaku_qwen_image_edit_2511_ultimate_speed_int4.safetensors",
    "nunchaku_qwen_image_edit_2511_best_quality_int4.safetensors",
    "nunchaku_qwen_image_edit_2511_balance_fp4.safetensors",
    "nunchaku_qwen_image_edit_2511_ultimate_speed_fp4.safetensors",
    "nunchaku_qwen_image_edit_2511_best_quality_fp4.safetensors",
]
QWEN_EDIT_BASE_REQUIRED_PATHS = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "processor/preprocessor_config.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


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


def _qwen_transformer_class():
    import nunchaku

    return getattr(nunchaku, "NunchakuQwenImageTransformer2DModel")


def _patch_qwen_transformer_txt_seq_lens(transformer) -> None:
    if getattr(transformer, "_nymphs_txt_seq_lens_patch", False):
        return
    original_forward = transformer.forward

    def forward_with_txt_seq_lens(*args, **kwargs):
        if kwargs.get("txt_seq_lens") is None:
            mask = kwargs.get("encoder_hidden_states_mask")
            hidden = kwargs.get("encoder_hidden_states")
            if mask is not None:
                try:
                    seq_lens = mask.sum(dim=1).detach().to("cpu").tolist()
                    kwargs["txt_seq_lens"] = [int(value) for value in seq_lens]
                except Exception:
                    kwargs["txt_seq_lens"] = [int(mask.shape[-1])]
            elif hidden is not None:
                batch = int(hidden.shape[0]) if len(hidden.shape) > 1 else 1
                kwargs["txt_seq_lens"] = [int(hidden.shape[1])] * batch
        return original_forward(*args, **kwargs)

    transformer.forward = forward_with_txt_seq_lens
    transformer._nymphs_txt_seq_lens_patch = True


class ImageServiceCoordinator:
    def __init__(self, settings, zimage_manager):
        self.settings = settings
        self.zimage_manager = zimage_manager
        self._lock = Lock()
        self._qwen_edit = None
        self._qwen_edit_model_id = None

    @staticmethod
    def normalize_provider(provider: str | None, mode: str = "txt2img") -> str:
        normalized = (provider or "zimage").strip().lower().replace("_", "-")
        if normalized in {"", "zimage", "z-image"}:
            return "zimage"
        if normalized in {"qwen", "qwen-edit", "qwen-img2img", "qwen-image-edit"}:
            return "qwen_edit"
        raise ValueError(f"Unknown image provider: {provider}.")

    def unload_all(self, keep: str | None = None) -> None:
        if keep != "zimage" and hasattr(self.zimage_manager, "_unload_pipelines"):
            self.zimage_manager._unload_pipelines()
        if keep != "qwen_edit":
            self._qwen_edit = None
            self._qwen_edit_model_id = None
        _empty_cuda_cache()

    def _qwen_edit_weight_file(self) -> str | None:
        for filename in QWEN_EDIT_WEIGHT_FILES:
            if _hf_file_ready(self.settings.hf_cache_dir, QWEN_EDIT_WEIGHT_REPO, filename):
                return filename
        return None

    def _qwen_edit_ready(self) -> bool:
        return _snapshot_ready(
            self.settings.hf_cache_dir,
            QWEN_EDIT_MODEL_ID,
            QWEN_EDIT_BASE_REQUIRED_PATHS,
        ) and self._qwen_edit_weight_file() is not None

    def providers_info(self, supported_modes: list[str], supports_lora: bool) -> dict:
        runtime = self.zimage_manager.loaded_runtime or self.settings.runtime
        zimage_loaded = self.zimage_manager.loaded_model_id is not None
        qwen_edit_filename = self._qwen_edit_weight_file()
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
            "qwen_edit": {
                "label": "Qwen Image Edit 2511",
                "ready": self._qwen_edit_ready(),
                "loaded": self._qwen_edit is not None,
                "model_id": QWEN_EDIT_MODEL_ID,
                "weight_repo": QWEN_EDIT_WEIGHT_REPO,
                "weight_file": qwen_edit_filename,
                "modes": ["img2img", "reference_edit", "parts_extract"],
                "supports_txt2img": False,
                "supports_img2img": True,
                "supports_reference_edit": True,
                "supports_lora": False,
                "supports_parts_extract": True,
                "defaults": {"width": 1024, "height": 1024, "steps": 20, "guidance_scale": 1.0, "strength": 0.75},
                "true_cfg_scale": 4.0,
                "lora_family": None,
                "memory_mode": "single_provider",
            },
        }

    def _load_qwen_edit(self):
        if not self._qwen_edit_ready():
            raise RuntimeError("Fetch Qwen Image Edit 2511 before generating with provider=qwen_edit.")
        if self._qwen_edit is not None:
            return self._qwen_edit
        self.unload_all(keep="qwen_edit")
        from diffusers import QwenImageEditPlusPipeline
        from nunchaku.utils import get_gpu_memory

        transformer_cls = _qwen_transformer_class()
        filename = self._qwen_edit_weight_file()
        weight_path = _hf_cached_file(self.settings.hf_cache_dir, QWEN_EDIT_WEIGHT_REPO, filename or "")
        if weight_path is None:
            raise RuntimeError("Fetch Qwen Image Edit 2511 before generating with provider=qwen_edit.")
        transformer = transformer_cls.from_pretrained(
            str(weight_path),
            torch_dtype=_torch_dtype(self.settings),
        )
        _patch_qwen_transformer_txt_seq_lens(transformer)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            QWEN_EDIT_MODEL_ID,
            transformer=transformer,
            torch_dtype=_torch_dtype(self.settings),
            cache_dir=str(self.settings.hf_cache_dir) if self.settings.hf_cache_dir else None,
            local_files_only=True,
            token=self.settings.hf_token,
        )
        if self.settings.device == "cuda" and get_gpu_memory() <= 18:
            if hasattr(transformer, "set_offload"):
                transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            if hasattr(pipe, "_exclude_from_cpu_offload"):
                pipe._exclude_from_cpu_offload.append("transformer")
            pipe.enable_sequential_cpu_offload()
        elif self.settings.device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif self.settings.device:
            pipe = pipe.to(self.settings.device)
        self._qwen_edit = pipe
        self._qwen_edit_model_id = QWEN_EDIT_MODEL_ID
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
        raise RuntimeError("Qwen Image Edit supports img2img/reference editing only. Use Z-Image for txt2img.")

    def generate_image_to_image(self, payload, image: Image.Image, progress_callback=None):
        with self._lock:
            pipe = self._load_qwen_edit()
            steps = int(payload.steps or 20)
            kwargs = {
                "prompt": payload.prompt,
                "image": [image],
                "num_inference_steps": steps,
                "guidance_scale": payload.guidance_scale if payload.guidance_scale is not None else 1.0,
                "true_cfg_scale": 4.0,
                "negative_prompt": payload.negative_prompt or " ",
                "num_images_per_prompt": 1,
                "generator": self._build_generator(payload.seed),
            }
            if progress_callback is not None:
                def _on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
                    progress_callback(int(step_index) + 1, steps)
                    return callback_kwargs

                kwargs["callback_on_step_end"] = _on_step_end
            result = pipe(**kwargs)
            return result.images[0], QWEN_EDIT_MODEL_ID
