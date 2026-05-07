#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from nunchaku import NunchakuZImageTransformer2DModel

from nunchaku_compat import patch_zimage_transformer_forward


DEFAULT_PROMPT = (
    "single fantasy goblin adventurer, full body character concept, centered, "
    "neutral standing pose, plain light background, painted illustration, clean "
    "silhouette, detailed game art"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local Nunchaku Z-Image-Turbo smoke test."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--base-model", default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument(
        "--rank-path",
        default="nunchaku-ai/nunchaku-z-image-turbo/svdq-int4_r32-z-image-turbo.safetensors",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output",
        default="/home/nymphs3d/Nymphs2D2/outputs/nunchaku-r32-zimage-test.png",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default="/home/nymphs3d/.cache/huggingface/hub",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16

    print("loading transformer...")
    t0 = time.perf_counter()
    patched_forward = patch_zimage_transformer_forward(NunchakuZImageTransformer2DModel)
    print(f"zimage_forward_shim={patched_forward}")
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(
        args.rank_path,
        torch_dtype=dtype,
    )
    print(f"transformer_loaded_sec={time.perf_counter() - t0:.2f}")

    print("loading pipeline...")
    t1 = time.perf_counter()
    pipe = ZImagePipeline.from_pretrained(
        args.base_model,
        transformer=transformer,
        torch_dtype=dtype,
        cache_dir=args.hf_cache_dir,
        low_cpu_mem_usage=False,
    )
    pipe.enable_sequential_cpu_offload()
    print(f"pipeline_loaded_sec={time.perf_counter() - t1:.2f}")

    print("running inference...")
    t2 = time.perf_counter()
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=torch.Generator().manual_seed(args.seed),
    ).images[0]
    print(f"inference_sec={time.perf_counter() - t2:.2f}")

    image.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
