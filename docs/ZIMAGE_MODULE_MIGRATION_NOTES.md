# Z-Image Turbo Module Migration Notes

This repo is the clean module home for the current Z-Image backend.

Source copied from:

```text
/home/nymph/Z-Image
```

Current source HEAD at copy time:

```text
2109c9d Stabilize Z-Image Nunchaku LoRA runtime path
```

The old manager name was `Nymphs2D2`; the new module identity is:

```text
id: zimage
name: Z-Image Turbo
short name: ZI
repo: github.com/nymphnerds/zimage
install path: ~/Z-Image
```

## What Carries Over

- `api_server.py`
- `config.py`
- `image_store.py`
- `model_manager.py`
- `nunchaku_compat.py`
- `progress_state.py`
- `schemas.py`
- `requirements.lock.txt`
- `scripts/prefetch_model.py`
- `scripts/run_nunchaku_zimage_test.py`

## What Stays Out Of Git

- `.venv`
- `.venv-nunchaku`
- model weights
- Hugging Face cache
- generated images
- runtime logs
- user LoRAs

## Nunchaku Dependency

The working runtime depends on the Nymph Nerds Nunchaku fork:

```text
https://github.com/nymphnerds/nunchaku.git
a2a4f2444a092974ba53323ba0681a523ff98031
```

That fork should remain its own dependency. Do not copy the Nunchaku source tree into this repo.

## Manager Migration

The current manager installer logic lives in:

```text
NymphsCore/Manager/scripts/install_nymphs2d2.sh
NymphsCore/Manager/scripts/runtime_tools_status.sh
NymphsCore/Manager/scripts/prefetch_models.sh
NymphsCore/Manager/scripts/smoke_test_server.sh
```

The module repo now owns equivalent scripts under `scripts/`.

The manager should eventually stop special-casing `install_nymphs2d2.sh` and instead call the module contract from `nymph.json`.

## Native Model Fetch UI Rule

The Z-Image Turbo model fetch controls should be native Manager controls
declared by the module manifest, not a WebView2/local HTML page. The manifest
tells the Manager how to discover/install/run the module and how to render the
compact model fetch action group.

Current module-owned UI expectations:

```text
ui.manager_action_groups[id=model_fetch]
entrypoint = fetch_models
```

The Fetch Models surface stays module-owned. It downloads the base Z-Image Turbo
model plus one Nunchaku-compatible quantized `.safetensors` weight:

```text
svdq-int4_r32-z-image-turbo.safetensors
svdq-int4_r128-z-image-turbo.safetensors
svdq-int4_r256-z-image-turbo.safetensors
svdq-fp4_r32-z-image-turbo.safetensors
svdq-fp4_r128-z-image-turbo.safetensors
```

Those options are Z-Image Turbo inference weights for the Nunchaku runtime, not
LoRA training precision. BF16 belongs to the LoRA/training path.

The Manager may hydrate the optional `HF Token` field from its shared local
secrets file and passes it to downloads through `NYMPHS3D_HF_TOKEN`. Do not log
the token, commit it, or write it into the module repo.
