from __future__ import annotations

import gc
import os
from threading import RLock

import torch
import torch.nn as nn
from safetensors.torch import load_file

from config import Settings
from nunchaku_compat import patch_zimage_transformer_forward


DEFAULT_ZIMAGE_CONTROLNET_REPO = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union"
DEFAULT_ZIMAGE_CONTROLNET_FILE = "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"


def _experimental_nunchaku_img2img_enabled() -> bool:
    raw = os.getenv("Z_IMAGE_NUNCHAKU_IMG2IMG") or os.getenv("NYMPHS2D2_NUNCHAKU_IMG2IMG")
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


class DeferredNunchakuLoraWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._desired_lora_path: str | None = None
        self._desired_lora_scale: float = 1.0
        self._applied_lora_path: str | None = None
        self._applied_lora_scale: float | None = None

    def __getattr__(self, name):
        if name != "model":
            modules = self.__dict__.get("_modules", {})
            model = modules.get("model")
            if model is not None:
                try:
                    return getattr(model, name)
                except AttributeError:
                    pass
        return super().__getattr__(name)

    def update_lora_params(self, path_or_state_dict: str | dict[str, torch.Tensor]):
        if not isinstance(path_or_state_dict, str):
            raise TypeError("DeferredNunchakuLoraWrapper currently expects a LoRA file path.")
        self._desired_lora_path = path_or_state_dict

    def set_lora_strength(self, strength: float = 1.0):
        self._desired_lora_scale = float(strength)

    def reset_lora(self):
        self._desired_lora_path = None
        self._desired_lora_scale = 1.0

    def _sync_lora_state(self):
        desired_path = self._desired_lora_path
        desired_scale = self._desired_lora_scale
        if desired_path == self._applied_lora_path and (
            desired_path is None or desired_scale == self._applied_lora_scale
        ):
            return

        if hasattr(self.model, "reset_lora"):
            self.model.reset_lora()

        if desired_path:
            print(
                f"[nymphs:zimage:lora] wrapper.compose path={desired_path} strength={desired_scale}",
                flush=True,
            )
            self.model.update_lora_params(desired_path)
            self.model.set_lora_strength(desired_scale)
            self._applied_lora_path = desired_path
            self._applied_lora_scale = desired_scale
        else:
            print("[nymphs:zimage:lora] wrapper.reset", flush=True)
            self._applied_lora_path = None
            self._applied_lora_scale = None

    def forward(self, *args, **kwargs):
        self._sync_lora_state()
        return self.model(*args, **kwargs)


def _wrap_pipeline_transformer_for_deferred_lora(pipeline):
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or isinstance(transformer, DeferredNunchakuLoraWrapper):
        return pipeline
    pipeline.transformer = DeferredNunchakuLoraWrapper(transformer)
    return pipeline


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._txt2img = None
        self._img2img = None
        self._controlnet_edit = None
        self._loaded_model_id = None
        self._dtype = self._resolve_torch_dtype(settings.dtype)
        if settings.device == "cpu" and self._dtype != torch.float32:
            self._dtype = torch.float32
        self._loaded_model_family = None
        self._loaded_runtime = None
        self._loaded_runtime_extra = {}

    @property
    def loaded_model_id(self) -> str | None:
        return self._loaded_model_id

    @property
    def loaded_runtime(self) -> str | None:
        return self._loaded_runtime

    @property
    def loaded_runtime_extra(self) -> dict:
        return dict(self._loaded_runtime_extra)

    def _loaded_runtime_config_matches(self, model_id: str, runtime: str) -> bool:
        if self._loaded_model_id != model_id or self._loaded_runtime != runtime:
            return False
        if runtime != "nunchaku":
            return True

        extra = self._loaded_runtime_extra or {}
        try:
            loaded_rank = int(extra.get("nunchaku_rank") or 0)
        except (TypeError, ValueError):
            loaded_rank = 0
        if loaded_rank != int(self.settings.nunchaku_rank):
            return False

        configured_precision = (self.settings.nunchaku_precision or "auto").strip().lower()
        loaded_precision = str(extra.get("nunchaku_precision") or "").strip().lower()
        if configured_precision in {"int4", "fp4"} and loaded_precision != configured_precision:
            return False
        return True

    def _loaded_pipeline_matches(self, model_id: str, runtime: str) -> bool:
        if self._txt2img is None:
            return False
        return self._loaded_runtime_config_matches(model_id, runtime)

    def _loaded_model_state_matches(self, model_id: str, runtime: str) -> bool:
        if self._txt2img is None and self._img2img is None and self._controlnet_edit is None:
            return False
        return self._loaded_runtime_config_matches(model_id, runtime)

    def _model_family(self, model_id: str | None) -> str:
        normalized = (model_id or self.settings.default_model_id or "").strip().lower()
        if "z-image" in normalized:
            return "zimage"
        return "generic"

    def _is_zimage_turbo_model(self, model_id: str | None) -> bool:
        normalized = (model_id or self.settings.default_model_id or "").strip().lower()
        return normalized.endswith("/z-image-turbo")

    def _resolve_torch_dtype(self, dtype_name: str):
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(dtype_name.lower(), torch.float16)

    def _resolve_runtime(self, model_id: str | None) -> str:
        runtime = (self.settings.runtime or "standard").strip().lower()
        if runtime != "nunchaku":
            return "standard"
        if self._model_family(model_id) != "zimage" or not self._is_zimage_turbo_model(model_id):
            raise RuntimeError("Nunchaku runtime currently supports Tongyi-MAI/Z-Image-Turbo only.")
        return "nunchaku"

    def supports_img2img(self, requested_model_id: str | None = None) -> bool:
        if self._resolve_runtime(requested_model_id or self.settings.default_model_id) != "nunchaku":
            return True
        return _experimental_nunchaku_img2img_enabled()

    def supported_modes(self, requested_model_id: str | None = None) -> list[str]:
        modes = ["txt2img"]
        if self.supports_controlnet_edit(requested_model_id):
            modes.append("controlnet_edit")
        if self.supports_img2img(requested_model_id):
            modes.append("img2img")
        return modes

    def supports_controlnet_edit(self, requested_model_id: str | None = None) -> bool:
        if self._resolve_runtime(requested_model_id or self.settings.default_model_id) != "nunchaku":
            return False
        return self._model_family(requested_model_id or self.settings.default_model_id) == "zimage"

    def supports_lora(self, requested_model_id: str | None = None) -> bool:
        runtime = self._resolve_runtime(requested_model_id or self.settings.default_model_id)
        if runtime != "nunchaku":
            return True
        transformer = None
        if self._txt2img is not None:
            transformer = getattr(self._txt2img, "transformer", None)
        if transformer is not None:
            return (
                hasattr(transformer, "update_lora_params")
                and hasattr(transformer, "set_lora_strength")
                and hasattr(transformer, "reset_lora")
            )
        try:
            from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel
        except Exception:
            return False
        return (
            hasattr(NunchakuZImageTransformer2DModel, "update_lora_params")
            and hasattr(NunchakuZImageTransformer2DModel, "set_lora_strength")
            and hasattr(NunchakuZImageTransformer2DModel, "reset_lora")
        )

    def _pipeline_kwargs(self, model_id: str | None, runtime: str) -> dict:
        model_family = self._model_family(model_id)
        kwargs = {
            "torch_dtype": self._dtype,
        }
        if model_family == "zimage":
            kwargs["low_cpu_mem_usage"] = False
        elif runtime != "nunchaku" and self.settings.variant:
            kwargs["variant"] = self.settings.variant
        if self.settings.hf_cache_dir:
            kwargs["cache_dir"] = str(self.settings.hf_cache_dir)
        if self.settings.hf_token:
            kwargs["token"] = self.settings.hf_token
        if self.settings.use_safetensors:
            kwargs["use_safetensors"] = True
        return kwargs

    def _prepare_pipeline(self, pipeline, runtime: str, *, cpu_offload: bool = True):
        if runtime == "nunchaku":
            if hasattr(pipeline, "remove_all_hooks"):
                pipeline.remove_all_hooks()
            if cpu_offload and self.settings.device != "cpu" and hasattr(pipeline, "enable_sequential_cpu_offload"):
                pipeline.enable_sequential_cpu_offload()
                pipeline._nymphs_nunchaku_offload_enabled = True
                return pipeline
            if self.settings.device:
                pipeline = pipeline.to(self.settings.device)
            pipeline._nymphs_nunchaku_offload_enabled = False
            return pipeline

        if self.settings.device:
            pipeline = pipeline.to(self.settings.device)
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        return pipeline

    def _set_nunchaku_lora_execution_mode(self, pipeline, lora_active: bool):
        return

    def _unload_pipelines(self):
        self._txt2img = None
        self._img2img = None
        self._controlnet_edit = None
        self._loaded_model_id = None
        self._loaded_model_family = None
        self._loaded_runtime = None
        self._loaded_runtime_extra = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_nunchaku_dtype(self):
        if self.settings.device == "cpu":
            return torch.float32
        if self._dtype == torch.float32:
            return torch.float32
        try:
            from nunchaku.utils import is_turing

            if is_turing(self.settings.device):
                return torch.float16
        except Exception:
            pass
        return self._dtype

    def _nunchaku_rank_path(self) -> tuple[str, str]:
        from nunchaku.utils import get_precision

        precision = self.settings.nunchaku_precision or "auto"
        if precision == "auto":
            precision = get_precision(precision="auto", device=self.settings.device)
        rank_path = (
            f"{self.settings.nunchaku_model_repo}/"
            f"svdq-{precision}_r{self.settings.nunchaku_rank}-z-image-turbo.safetensors"
        )
        return rank_path, precision

    def _resolve_nunchaku_rank_path(self, rank_path: str) -> str:
        if self.settings.hf_cache_dir is None:
            return rank_path
        repo_id, filename = rank_path.rsplit("/", 1)
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(self.settings.hf_cache_dir),
            token=self.settings.hf_token,
        )
        return str(path)

    def _load_nunchaku_transformer(self):
        try:
            from nunchaku import NunchakuZImageTransformer2DModel
        except ImportError as exc:
            raise RuntimeError("Nunchaku runtime dependencies are not installed in this environment.") from exc

        patch_zimage_transformer_forward(NunchakuZImageTransformer2DModel)
        rank_path, precision = self._nunchaku_rank_path()
        local_rank_path = self._resolve_nunchaku_rank_path(rank_path)
        dtype = self._resolve_nunchaku_dtype()
        transformer = NunchakuZImageTransformer2DModel.from_pretrained(local_rank_path, torch_dtype=dtype)
        return transformer, local_rank_path, precision, dtype

    def _load_txt2img_pipeline(self, model_id: str, runtime: str):
        if runtime == "nunchaku":
            try:
                from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
            except ImportError as exc:
                raise RuntimeError("Nunchaku runtime dependencies are not installed in this environment.") from exc

            transformer, rank_path, precision, dtype = self._load_nunchaku_transformer()
            self._loaded_runtime_extra = {
                "runtime": "nunchaku",
                "nunchaku_rank": self.settings.nunchaku_rank,
                "nunchaku_precision": precision,
                "nunchaku_rank_path": rank_path,
                "runtime_dtype": str(dtype).replace("torch.", ""),
                "zimage_forward_shim": True,
                "experimental_img2img": _experimental_nunchaku_img2img_enabled(),
                "controlnet_edit": True,
            }
            pipeline = ZImagePipeline.from_pretrained(
                model_id,
                transformer=transformer,
                **self._pipeline_kwargs(model_id, runtime),
            )
            return _wrap_pipeline_transformer_for_deferred_lora(pipeline)

        if self._model_family(model_id) == "zimage":
            try:
                from diffusers import ZImagePipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Current diffusers build does not include Z-Image support. "
                    "Install a newer diffusers build before loading Tongyi-MAI/Z-Image models."
                ) from exc
            return ZImagePipeline.from_pretrained(model_id, **self._pipeline_kwargs(model_id, runtime))

        from diffusers import AutoPipelineForText2Image

        return AutoPipelineForText2Image.from_pretrained(model_id, **self._pipeline_kwargs(model_id, runtime))

    def ensure_model(self, requested_model_id: str | None = None) -> str:
        model_id = requested_model_id or self.settings.default_model_id
        with self._lock:
            runtime = self._resolve_runtime(model_id)
            if self._loaded_pipeline_matches(model_id, runtime):
                return model_id

            self._unload_pipelines()
            self._txt2img = self._load_txt2img_pipeline(model_id, runtime)
            self._txt2img = self._prepare_pipeline(self._txt2img, runtime)
            self._loaded_model_id = model_id
            self._loaded_model_family = self._model_family(model_id)
            self._loaded_runtime = runtime
            return model_id

    def ensure_controlnet_model(self, requested_model_id: str | None = None) -> str:
        model_id = requested_model_id or self.settings.default_model_id
        with self._lock:
            runtime = self._resolve_runtime(model_id)
            if runtime != "nunchaku" or self._model_family(model_id) != "zimage":
                raise RuntimeError("Z-Image ControlNet edit requires Nunchaku runtime with Tongyi-MAI/Z-Image-Turbo.")
            if self._loaded_model_state_matches(model_id, runtime):
                return model_id

            self._unload_pipelines()
            self._loaded_model_id = model_id
            self._loaded_model_family = self._model_family(model_id)
            self._loaded_runtime = runtime
            self._loaded_runtime_extra = {
                "runtime": "nunchaku",
                "nunchaku_rank": self.settings.nunchaku_rank,
                "nunchaku_precision": self.settings.nunchaku_precision,
                "controlnet_edit": True,
            }
            return model_id

    def _resolve_controlnet_path(self) -> tuple[str, str, str]:
        configured_path = (
            os.getenv("Z_IMAGE_CONTROLNET_PATH")
            or os.getenv("NYMPHS2D2_CONTROLNET_PATH")
            or ""
        ).strip()
        if configured_path:
            return configured_path, "local", configured_path

        repo_id = (
            os.getenv("Z_IMAGE_CONTROLNET_REPO")
            or os.getenv("NYMPHS2D2_CONTROLNET_REPO")
            or DEFAULT_ZIMAGE_CONTROLNET_REPO
        ).strip()
        filename = (
            os.getenv("Z_IMAGE_CONTROLNET_FILE")
            or os.getenv("NYMPHS2D2_CONTROLNET_FILE")
            or DEFAULT_ZIMAGE_CONTROLNET_FILE
        ).strip()
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(self.settings.hf_cache_dir) if self.settings.hf_cache_dir else None,
            token=self.settings.hf_token,
        )
        return str(path), repo_id, filename

    def _ensure_controlnet_edit(self):
        if self._controlnet_edit is not None:
            return self._controlnet_edit

        if self._loaded_runtime != "nunchaku" or self._loaded_model_family != "zimage":
            raise RuntimeError("Z-Image ControlNet edit requires Nunchaku runtime with Tongyi-MAI/Z-Image-Turbo.")

        self._txt2img = None
        self._img2img = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            from diffusers import ZImageControlNetModel, ZImageControlNetPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Current diffusers build does not include Z-Image ControlNet support. "
                "Install diffusers 0.37.1 or newer before using controlnet_edit."
            ) from exc

        transformer, rank_path, precision, dtype = self._load_nunchaku_transformer()
        controlnet_path, controlnet_repo, controlnet_file = self._resolve_controlnet_path()
        controlnet = ZImageControlNetModel.from_single_file(controlnet_path, torch_dtype=dtype)
        self._controlnet_edit = ZImageControlNetPipeline.from_pretrained(
            self._loaded_model_id,
            transformer=transformer,
            controlnet=controlnet,
            **self._pipeline_kwargs(self._loaded_model_id, self._loaded_runtime or "nunchaku"),
        )
        self._controlnet_edit = _wrap_pipeline_transformer_for_deferred_lora(self._controlnet_edit)
        self._controlnet_edit = self._prepare_pipeline(
            self._controlnet_edit,
            self._loaded_runtime or "nunchaku",
            cpu_offload=False,
        )
        self._loaded_runtime_extra.update(
            {
                "nunchaku_precision": precision,
                "nunchaku_rank_path": rank_path,
                "runtime_dtype": str(dtype).replace("torch.", ""),
                "controlnet_edit": True,
                "controlnet_repo": controlnet_repo,
                "controlnet_file": controlnet_file,
            }
        )
        return self._controlnet_edit

    def _ensure_img2img(self):
        if self._img2img is not None:
            return self._img2img

        if self._loaded_runtime == "nunchaku":
            if not _experimental_nunchaku_img2img_enabled():
                raise RuntimeError(
                    "Nunchaku img2img is experimental. Set Z_IMAGE_NUNCHAKU_IMG2IMG=1 to enable it."
                )
            if self._loaded_model_family != "zimage":
                raise RuntimeError("Nunchaku img2img currently supports Z-Image models only.")

            self._txt2img = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                from diffusers import ZImageImg2ImgPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Current diffusers build does not include Z-Image img2img support. "
                    "Install a newer diffusers build before using Z-Image img2img."
                ) from exc
            transformer, rank_path, precision, dtype = self._load_nunchaku_transformer()
            self._loaded_runtime_extra.update(
                {
                    "nunchaku_precision": precision,
                    "nunchaku_rank_path": rank_path,
                    "runtime_dtype": str(dtype).replace("torch.", ""),
                    "experimental_img2img": True,
                }
            )
            self._img2img = ZImageImg2ImgPipeline.from_pretrained(
                self._loaded_model_id,
                transformer=transformer,
                **self._pipeline_kwargs(self._loaded_model_id, self._loaded_runtime or "nunchaku"),
            )
            self._img2img = _wrap_pipeline_transformer_for_deferred_lora(self._img2img)
            self._img2img = self._prepare_pipeline(self._img2img, self._loaded_runtime or "nunchaku")
            return self._img2img

        if self._loaded_model_family == "zimage":
            # Z-Image img2img uses a separate pipeline class. Drop the txt2img
            # pipeline first so we do not keep two full 6B-class pipelines on
            # the GPU at once during iterative edit workflows.
            self._txt2img = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                from diffusers import ZImageImg2ImgPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Current diffusers build does not include Z-Image img2img support. "
                    "Install a newer diffusers build before using Tongyi-MAI/Z-Image models."
                ) from exc
            self._img2img = ZImageImg2ImgPipeline.from_pretrained(
                self._loaded_model_id,
                **self._pipeline_kwargs(self._loaded_model_id, self._loaded_runtime or "standard"),
            )
            self._img2img = self._prepare_pipeline(self._img2img, self._loaded_runtime or "standard")
            return self._img2img

        from diffusers import AutoPipelineForImage2Image

        try:
            self._img2img = AutoPipelineForImage2Image.from_pipe(self._txt2img)
            self._img2img = self._prepare_pipeline(self._img2img, self._loaded_runtime or "standard")
        except AttributeError:
            self._img2img = AutoPipelineForImage2Image.from_pretrained(
                self._loaded_model_id,
                **self._pipeline_kwargs(self._loaded_model_id, self._loaded_runtime or "standard"),
            )
            self._img2img = self._prepare_pipeline(self._img2img, self._loaded_runtime or "standard")
        return self._img2img

    def _build_generator(self, seed: int | None):
        if seed is None:
            return None

        device = self.settings.device if self.settings.device != "mps" else "cpu"
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def _load_lora_with_alpha_fallback(self, pipeline, lora_path: str, adapter_name: str):
        try:
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
            return
        except KeyError as exc:
            if ".alpha" not in str(exc):
                raise

        state_dict = load_file(lora_path)
        if any(key.endswith(".alpha") for key in state_dict):
            raise

        patched_state_dict = dict(state_dict)
        added_alpha = False
        for key, value in state_dict.items():
            if key.endswith(".lora_A.default.weight"):
                alpha_key = key.replace(".lora_A.default.weight", ".alpha")
                if alpha_key not in patched_state_dict:
                    patched_state_dict[alpha_key] = torch.tensor(
                        float(value.shape[0]), dtype=value.dtype, device=value.device
                    )
                    added_alpha = True

        if not added_alpha:
            raise

        pipeline.load_lora_weights(patched_state_dict, adapter_name=adapter_name)

    def _configure_pipeline_lora(self, pipeline, lora_path: str | None, lora_scale: float | None):
        desired_path = (lora_path or "").strip()
        loaded_path = getattr(pipeline, "_nymphs_lora_path", None)

        if not desired_path:
            if loaded_path:
                if self._loaded_runtime == "nunchaku":
                    transformer = getattr(pipeline, "transformer", None)
                    if transformer is not None and hasattr(transformer, "reset_lora"):
                        transformer.reset_lora()
                    self._set_nunchaku_lora_execution_mode(pipeline, False)
                else:
                    pipeline.unload_lora_weights()
                pipeline._nymphs_lora_path = None
                pipeline._nymphs_lora_scale = None
            return

        if not os.path.isfile(desired_path):
            raise RuntimeError(f"LoRA file not found: {desired_path}")

        desired_scale = float(1.0 if lora_scale is None else lora_scale)
        if self._loaded_runtime == "nunchaku":
            self._set_nunchaku_lora_execution_mode(pipeline, True)
            transformer = getattr(pipeline, "transformer", None)
            if (
                transformer is None
                or not hasattr(transformer, "update_lora_params")
                or not hasattr(transformer, "set_lora_strength")
            ):
                raise ValueError(
                    "Current Z-Image Nunchaku runtime does not include LoRA support yet. "
                    "Repair Z-Image after updating to the NymphNerds Nunchaku fork build."
                )
            if loaded_path != desired_path:
                if loaded_path and hasattr(transformer, "reset_lora"):
                    transformer.reset_lora()
                transformer.update_lora_params(desired_path)
                pipeline._nymphs_lora_path = desired_path
            transformer.set_lora_strength(desired_scale)
            pipeline._nymphs_lora_scale = desired_scale
            return

        if loaded_path != desired_path:
            if loaded_path:
                pipeline.unload_lora_weights()
            self._load_lora_with_alpha_fallback(pipeline, desired_path, adapter_name="nymphs_user_lora")
            pipeline._nymphs_lora_path = desired_path

        if hasattr(pipeline, "set_adapters"):
            pipeline.set_adapters(["nymphs_user_lora"], [desired_scale])
        pipeline._nymphs_lora_scale = desired_scale

    def generate_text_to_image(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        model_id: str | None,
        nunchaku_rank: int | None,
        nunchaku_precision: str | None,
        lora_path: str | None,
        lora_scale: float | None,
        progress_callback=None,
    ):
        with self._lock:
            active_model_id = self.ensure_model(model_id)
            self._configure_pipeline_lora(self._txt2img, lora_path, lora_scale)
            generator = self._build_generator(seed)
            kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            if self._loaded_runtime != "nunchaku":
                kwargs["negative_prompt"] = negative_prompt
            if progress_callback is not None:
                def _on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
                    progress_callback(int(step_index) + 1, steps)
                    return callback_kwargs

                kwargs["callback_on_step_end"] = _on_step_end
            print("[nymphs:zimage:stage] pipeline.txt2img.begin", flush=True)
            result = self._txt2img(**kwargs)
            print("[nymphs:zimage:stage] pipeline.txt2img.returned", flush=True)
            image = result.images[0]
            print("[nymphs:zimage:stage] pipeline.txt2img.image_extracted", flush=True)
            return image, active_model_id

    def generate_controlnet_edit(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        control_image,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        controlnet_conditioning_scale: float,
        seed: int | None,
        model_id: str | None,
        nunchaku_rank: int | None,
        nunchaku_precision: str | None,
        lora_path: str | None,
        lora_scale: float | None,
        progress_callback=None,
    ):
        with self._lock:
            active_model_id = self.ensure_controlnet_model(model_id)
            pipeline = self._ensure_controlnet_edit()
            self._configure_pipeline_lora(pipeline, lora_path, lora_scale)
            generator = self._build_generator(seed)
            callback_kwargs = {}
            if progress_callback is not None:
                def _on_step_end(_pipeline, step_index, _timestep, callback_state):
                    progress_callback(int(step_index) + 1, steps)
                    return callback_state

                callback_kwargs["callback_on_step_end"] = _on_step_end
            print("[nymphs:zimage:stage] pipeline.controlnet_edit.begin", flush=True)
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=control_image,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                **callback_kwargs,
            )
            print("[nymphs:zimage:stage] pipeline.controlnet_edit.returned", flush=True)
            output_image = result.images[0]
            print("[nymphs:zimage:stage] pipeline.controlnet_edit.image_extracted", flush=True)
            return output_image, active_model_id

    def generate_image_to_image(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        image,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        strength: float,
        seed: int | None,
        model_id: str | None,
        nunchaku_rank: int | None,
        nunchaku_precision: str | None,
        lora_path: str | None,
        lora_scale: float | None,
        progress_callback=None,
    ):
        with self._lock:
            active_model_id = self.ensure_model(model_id)
            pipeline = self._ensure_img2img()
            self._configure_pipeline_lora(pipeline, lora_path, lora_scale)
            generator = self._build_generator(seed)
            callback_kwargs = {}
            if progress_callback is not None:
                def _on_step_end(_pipeline, step_index, _timestep, callback_state):
                    progress_callback(int(step_index) + 1, steps)
                    return callback_state

                callback_kwargs["callback_on_step_end"] = _on_step_end
            print("[nymphs:zimage:stage] pipeline.img2img.begin", flush=True)
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
                **callback_kwargs,
            )
            print("[nymphs:zimage:stage] pipeline.img2img.returned", flush=True)
            output_image = result.images[0]
            print("[nymphs:zimage:stage] pipeline.img2img.image_extracted", flush=True)
            return output_image, active_model_id
