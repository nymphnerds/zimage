# Z-Image Turbo

Z-Image Turbo is the NymphsCore 2D image generation backend packaged as an installable Nymph module.

It provides a local FastAPI service for:

- `txt2img`
- `img2img`
- local output files
- Z-Image Turbo through the Nunchaku runtime
- LoRA-compatible runtime hooks used by NymphsCore

This repo replaces the older internal `Nymphs2D2` module shape. The compatibility environment variables are still supported so the current manager and Blender-side tools can migrate without a hard cutover.

## Runtime Layout

Expected in-distro layout:

- module repo: `~/Z-Image`
- runtime venv: `~/Z-Image/.venv-nunchaku`
- outputs: `~/NymphsData/outputs/zimage`
- logs: `~/NymphsData/logs/zimage`
- Hugging Face cache: `~/NymphsData/cache/huggingface`

The manager should install or update this repo as a module. It should not copy virtual environments, generated images, model weights, or cache folders into git.

## Manager Contract

The manager discovers this module through `nymph.json` and calls scripts in `scripts/`.

Useful scripts:

```bash
scripts/install_zimage.sh
scripts/zimage_status.sh
scripts/zimage_start.sh
scripts/zimage_stop.sh
scripts/zimage_open.sh
scripts/zimage_logs.sh
scripts/zimage_fetch_models.sh
scripts/zimage_delete_models.sh
scripts/zimage_smoke_test.sh
```

The module page uses the standard NymphsCore install/detail shell before install.
After install, the Manager renders the compact native model fetch UI declared in
`ui.manager_action_groups`.

Z-Image owns the fetch choices, source links, script arguments, validation, and
runtime preset file. The Manager only renders the generic controls, stores the
shared Hugging Face token, and routes the declared action.

The native Fetch Models UI downloads the base Z-Image Turbo model plus one
Nunchaku-compatible quantized `.safetensors` weight:

- `svdq-int4_r32-z-image-turbo.safetensors`
- `svdq-int4_r128-z-image-turbo.safetensors`
- `svdq-int4_r256-z-image-turbo.safetensors`
- `svdq-fp4_r32-z-image-turbo.safetensors`
- `svdq-fp4_r128-z-image-turbo.safetensors`

FP4 r256 is not listed because the published r256 weight is INT4-only. These
choices are Z-Image Turbo inference weights for the Nunchaku runtime, not LoRA
training precision. LoRA training BF16 is handled separately by the training
stack.

The Fetch Models UI also has an optional `HF Token` password field. In the
NymphsCore Manager, that token is persisted in the Windows user profile under:

```text
%LOCALAPPDATA%\NymphsCore\shared-secrets.json
```

The token is passed into model downloads as `NYMPHS3D_HF_TOKEN` and must not be
printed to logs.

If the runtime is installed but the selected model cache is missing, status stays
`installed` and reports `models_ready=false` with a plain download message. That
state should not be treated as a broken install.

The module still owns `scripts/zimage_delete_models.sh` for local model cache
cleanup. Outputs, logs, LoRAs, and the runtime install are preserved.

## Important Dependency

Z-Image Turbo depends on the Nymph Nerds Nunchaku fork:

```text
git@github.com:nymphnerds/nunchaku.git
commit: a2a4f2444a092974ba53323ba0681a523ff98031
```

That fork is not vendored into this repo. The installer pins it into `.venv-nunchaku`.

The fork currently builds native CUDA extensions during install, so the installer
also checks for `nvcc` and installs CUDA Toolkit 13.0 for WSL through apt when it
is missing. If the NVIDIA apt repository cannot be added automatically, install
CUDA Toolkit 13.0 manually and rerun the module install.

## Environment Variables

Primary variables:

- `ZIMAGE_INSTALL_ROOT`
- `Z_IMAGE_PORT`
- `Z_IMAGE_RUNTIME`
- `Z_IMAGE_MODEL_ID`
- `Z_IMAGE_DEVICE`
- `Z_IMAGE_DTYPE`
- `Z_IMAGE_OUTPUT_DIR`
- `Z_IMAGE_NUNCHAKU_MODEL_REPO`
- `Z_IMAGE_NUNCHAKU_RANK`
- `Z_IMAGE_NUNCHAKU_PRECISION`
- `NYMPHS3D_HF_CACHE_DIR`
- `NYMPHS3D_HF_TOKEN`

Compatibility variables still accepted by the backend:

- `NYMPHS2D2_MODEL_ID`
- `NYMPHS2D2_RUNTIME`
- `NYMPHS2D2_OUTPUT_DIR`
- `NYMPHS2D2_PORT`

Default model:

```text
Tongyi-MAI/Z-Image-Turbo
```

Default Nunchaku-compatible Z-Image Turbo weights:

```text
nunchaku-ai/nunchaku-z-image-turbo
svdq-*_r32-z-image-turbo.safetensors
```

## Run Manually

```bash
scripts/zimage_start.sh
scripts/zimage_status.sh
scripts/zimage_logs.sh
scripts/zimage_fetch_models.sh --precision auto --rank 32
scripts/zimage_delete_models.sh --scope weights --yes
```

For private or gated Hugging Face downloads:

```bash
NYMPHS3D_HF_TOKEN=hf_xxx scripts/zimage_fetch_models.sh --precision int4 --rank 32
```

The default local URL is:

```text
http://127.0.0.1:8090
```

## Endpoints

- `GET /health`
- `GET /server_info`
- `GET /active_task`
- `POST /generate`

`POST /generate` supports:

- `mode="txt2img"`
- `mode="img2img"`

## Repo Rule

This repo should stay clean:

- keep source code, scripts, lockfiles, docs, and `nymph.json`
- do not commit `.venv-nunchaku`
- do not commit `outputs`
- do not commit `NymphsData`
- do not commit Hugging Face model cache
- do not commit user-generated images or metadata
