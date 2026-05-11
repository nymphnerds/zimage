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
scripts/zimage_smoke_test.sh
```

The module page uses the standard NymphsCore detail shell. Z-Image owns the center surface declared in:

```text
ui/manager.surface.json
```

That surface describes backend readiness, quantized model fetching, and endpoint facts. The Manager should keep the standard right rail and lifecycle contract buttons.

## Important Dependency

Z-Image Turbo depends on the Nymph Nerds Nunchaku fork:

```text
git@github.com:nymphnerds/nunchaku.git
commit: a2a4f2444a092974ba53323ba0681a523ff98031
```

That fork is not vendored into this repo. The installer pins it into `.venv-nunchaku`.

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
