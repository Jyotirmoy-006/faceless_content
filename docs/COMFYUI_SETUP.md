# ComfyUI Setup, Patch Documentation & Environment Reproducibility

## 1. Overview & Root Cause Analysis

During setup of local ComfyUI on Python 3.11 with PyTorch 2.6 (`torch==2.6.0+cu124`), two upstream compatibility issues were identified:

### Issue A: `comfy_kitchen` PEP 585 Schema Inference Error
- **Root Cause**: `comfy_kitchen` (v0.2.34) uses PEP 585 type hints (e.g. `list[int]`, `Tensor | None`) inside `@torch.library.custom_op` decorator signatures (e.g., `comfy_kitchen/backends/eager/conv3d.py`).
- PyTorch 2.6's `torch._library.infer_schema.py` module strictly expects `typing.List[int]` or `typing.Optional[Tensor]`. When Python 3.11 built-in parameterized generics are encountered in custom ops, PyTorch crashes during schema inference.
- **Resolution**: `comfy_kitchen` is an optional accelerator package specifically intended for experimental FP8/FP4 operations on CUDA 13+ devices. For our RTX 3050 (Ampere architecture, Compute Capability 8.6, CUDA 12.4), Stable Diffusion 1.5 (`DreamShaper_8_pruned.safetensors`) operates in FP16 native mode. `comfy_kitchen` is neither needed nor supported on 20/30-series GPUs.
- Therefore, `comfy-kitchen` was uninstalled from the environment:
  ```bash
  pip uninstall -y comfy-kitchen
  ```

### Issue B: Upstream `ComfyUI/comfy/ldm/modules/attention.py` Unconditional Import & NameError
- **Root Cause**: Upstream ComfyUI `attention.py` unconditionally attempted `import comfy_kitchen`, and conditionally referenced `model_management.force_upcast_attention_dtype()` without having `model_management` reliably imported at module scope when optional acceleration modules were absent.
- **Resolution**: A surgical compatibility patch was created:
  - Wraps `import comfy_kitchen` in a try/except guard.
  - Adds explicit `from comfy import model_management`.
  - Catches missing optional backends (`sageattention`, `flash_attn`, `xformers`) softly so standard PyTorch execution proceeds without failure.

---

## 2. Checked-In Patch File

The patch is checked in at:
[`comfyui_setup/comfyui_pytorch26_compat.patch`](../comfyui_setup/comfyui_pytorch26_compat.patch)

### How to apply on a fresh clone:
```bash
git apply ../comfyui_setup/comfyui_pytorch26_compat.patch
```

---

## 3. Automated One-Command Setup

For a fresh checkout, run the automated setup script:
```bash
python comfyui_setup/setup_comfyui.py
```

This script:
1. Clones ComfyUI into `ComfyUI_server/` (if not already present).
2. Verifies `comfy-kitchen` is uninstalled.
3. Automatically applies `comfyui_setup/comfyui_pytorch26_compat.patch`.
4. Hardlinks or verifies the SD1.5 model `DreamShaper_8_pruned.safetensors` into `ComfyUI_server/models/checkpoints/`.

---

## 4. Running ComfyUI for Low VRAM (RTX 3050 4GB)

Always launch ComfyUI with `--lowvram`:
```bash
python ComfyUI_server/main.py --listen 127.0.0.1 --port 8188 --lowvram
```

This ensures ComfyUI loads weights into VRAM dynamically only during active inference steps, allowing seamless coexistence with our GPU lock and subprocess isolation.
