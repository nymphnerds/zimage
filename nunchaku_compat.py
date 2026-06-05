from __future__ import annotations

import inspect
from typing import Any


def _patch_zimage_lora_rank_slots(transformer_module: Any) -> bool:
    """Keep Z-Image Nunchaku LoRA tensors inside the active rank slot.

    Some Nunchaku Z-Image builds expose fixed low-rank CUDA slots even when the
    selected checkpoint and external LoRA use a smaller rank. The upstream
    helper replaces those parameters with the raw merged tensor, which can make
    accelerate fail with shape errors such as ``qkv_proj_down`` 32 -> 128.
    """

    if getattr(transformer_module, "_nymphs2d2_zimage_lora_rank_slot_shim", False):
        return False

    original_replace = getattr(transformer_module, "_replace_module_parameter", None)
    if original_replace is None:
        return False

    def adapt_to_slot(tensor, slot):
        if getattr(slot, "shape", None) is None or tuple(tensor.shape) == tuple(slot.shape):
            return tensor
        if tensor.ndim != 2 or slot.ndim != 2:
            return tensor
        if tensor.shape[0] != slot.shape[0]:
            return tensor

        import torch

        target_rank = int(slot.shape[1])
        current_rank = int(tensor.shape[1])
        if current_rank == target_rank:
            return tensor
        if current_rank < target_rank:
            pad = torch.zeros(
                tensor.shape[0],
                target_rank - current_rank,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            return torch.cat([tensor, pad], dim=1)
        return tensor[:, :target_rank].contiguous()

    def rank_safe_replace(module, attr_name: str, tensor):
        import torch.nn as nn

        old_param = getattr(module, attr_name)
        tensor = adapt_to_slot(tensor, old_param)
        target_device = tensor.device if old_param.device.type == "meta" else old_param.device
        new_param = nn.Parameter(
            tensor.to(device=target_device, dtype=old_param.dtype),
            requires_grad=old_param.requires_grad,
        )
        setattr(module, attr_name, new_param)
        if attr_name.endswith(("proj_down", "qkv_proj_down")) and hasattr(module, "rank"):
            try:
                module.rank = int(new_param.shape[1])
            except Exception:
                pass

    transformer_module._replace_module_parameter = rank_safe_replace
    transformer_module._nymphs2d2_zimage_lora_rank_slot_shim = True
    transformer_module._nymphs2d2_zimage_lora_original_replace = original_replace
    return True


def patch_zimage_transformer_forward(transformer_cls: type) -> bool:
    """Patch Nunchaku's Z-Image wrapper to call diffusers with keyword args.

    Recent diffusers builds inserted optional parameters before ``patch_size`` in
    ``ZImageTransformer2DModel.forward``. Nunchaku 1.3.0 still forwards those
    arguments positionally, which makes latest diffusers interpret an integer
    patch size as ``controlnet_block_samples``.
    """

    transformer_module = inspect.getmodule(transformer_cls)
    if transformer_module is not None:
        _patch_zimage_lora_rank_slots(transformer_module)

    if getattr(transformer_cls, "_nymphs2d2_zimage_forward_shim", False):
        return False

    parent_forward = None
    for base_cls in transformer_cls.__mro__[1:]:
        if base_cls.__name__ == "ZImageTransformer2DModel":
            parent_forward = base_cls.forward
            break
    if parent_forward is None:
        return False

    parent_parameters = inspect.signature(parent_forward).parameters

    def forward(
        self: Any,
        x,
        t,
        cap_feats,
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
        **kwargs,
    ):
        from nunchaku.models.transformers.transformer_zimage import NunchakuZImageRopeHook

        call_kwargs = {
            "x": x,
            "t": t,
            "cap_feats": cap_feats,
        }
        optional_values = {
            "return_dict": return_dict,
            "patch_size": patch_size,
            "f_patch_size": f_patch_size,
            "controlnet_block_samples": kwargs.get("controlnet_block_samples"),
            "siglip_feats": kwargs.get("siglip_feats"),
            "image_noise_mask": kwargs.get("image_noise_mask"),
        }
        for name, value in optional_values.items():
            if name in parent_parameters:
                call_kwargs[name] = value

        rope_hook = NunchakuZImageRopeHook()
        self.register_rope_hook(rope_hook)
        try:
            return parent_forward(self, **call_kwargs)
        finally:
            self.unregister_rope_hook()
            del rope_hook

    transformer_cls.forward = forward
    transformer_cls._nymphs2d2_zimage_forward_shim = True
    return True
