from __future__ import annotations

import inspect
from typing import Any


def patch_zimage_transformer_forward(transformer_cls: type) -> bool:
    """Patch Nunchaku's Z-Image wrapper to call diffusers with keyword args.

    Recent diffusers builds inserted optional parameters before ``patch_size`` in
    ``ZImageTransformer2DModel.forward``. Nunchaku 1.3.0 still forwards those
    arguments positionally, which makes latest diffusers interpret an integer
    patch size as ``controlnet_block_samples``.
    """

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
