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
After install, the Manager can host the local module-owned UI declared in:

```text
ui/manager.html
```

That UI describes backend controls and quantized model fetching. The Manager should only act as a generic host for this local installed file; it should not hardcode Z-Image controls.

The current Fetch Models UI exposes all published Nunchaku Z-Image Turbo
generation weights:

- INT4 r32
- INT4 r128
- INT4 r256
- FP4 r32
- FP4 r128

`auto` is allowed for r32/r128 only, because r256 is INT4-only. These choices
are Nunchaku generation weights, not LoRA training precision. LoRA training BF16
is handled separately by the training stack.

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

The Fetch Models UI can also delete local model cache files. `delete_weights`
removes the Nunchaku quantized weight cache, while `delete_all_models` removes
both the base Z-Image Turbo cache and the Nunchaku weight cache. Outputs, logs,
LoRAs, and the runtime install are preserved.

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

Default Nunchaku weights:

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
