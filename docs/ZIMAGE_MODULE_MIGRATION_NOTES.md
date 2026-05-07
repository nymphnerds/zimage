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

## Custom Page Rule

The Z-Image Turbo manager page should be custom, not the generic fallback facts page. The manifest tells the manager how to discover/install/run the module; it does not replace the designed module page.
