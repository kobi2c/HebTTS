# HebTTS Multi-Band Diffusion (MBD) Profiling Suite

This directory contains the complete set of profiling scripts, raw benchmark outputs, structured comparison tables, and investigative insights created to diagnose and optimize Multi-Band Diffusion (`mbd=1`) performance in HebTTS.

---

## Performance Summary: The 555s $\rightarrow$ 63s Journey

```
Initial (CPU Fallback):                [===================================================] 555.6s
GPU Migration (with VRAM Thrashing):   [==============================] 343.5s
Optimized (GPU + Dynamic Offload):     [======] 63.4s  (Cold Start Init: 45s + Generation: 18s)
Subsequent Chunks (Warm / Cached):     [=] 18.0s
```

---

## Index of Profiling Scripts and Results

| # | Script | Results Document | Core Question Investigated | Primary Finding |
| :-: | :--- | :--- | :--- | :--- |
| **01** | [`01_pipeline_stages.py`](01_pipeline_stages.py) | [`01_pipeline_stages-results.md`](01_pipeline_stages-results.md) | Where is time spent in the end-to-end pipeline? | MBD decode alone was taking 182s per chunk. |
| **02** | [`02_mbd_device_check.py`](02_mbd_device_check.py) | [`02_mbd_device_check-results.md`](02_mbd_device_check-results.md) | Are any MBD weights or buffers still on CPU? | All $1.15\text{B}$ parameters and beta schedules were verified on `cuda:0`. |
| **03** | [`03_unet_raw_compute.py`](03_unet_raw_compute.py) | [`03_unet_raw_compute-results.md`](03_unet_raw_compute-results.md) | How fast is the raw GPU tensor math for one U-Net step? | Single pass takes 40.5ms; 80 passes take only 3.24s of raw GPU compute. |
| **04** | [`04_mbd_tokens_to_wav.py`](04_mbd_tokens_to_wav.py) | [`04_mbd_tokens_to_wav-results.md`](04_mbd_tokens_to_wav-results.md) | How fast does MBD run when loaded alone in GPU memory? | When alone on GPU, MBD decode completes in **4.87s** total. |
| **05** | [`05_vram_allocation.py`](05_vram_allocation.py) | [`05_vram_allocation-results.md`](05_vram_allocation-results.md) | How much VRAM do VALL-E and MBD consume together? | Total allocated + reserved is $8.15\text{ GB}$ (99.5% of 8GB VRAM), forcing PCIe paging. |
| **06** | [`06_vram_offloading.py`](06_vram_offloading.py) | [`06_vram_offloading-results.md`](06_vram_offloading-results.md) | Does offloading VALL-E to CPU eliminate PCIe thrashing? | Freeing 1.8 GB VRAM reduced MBD decode from **182.2s $\rightarrow$ 6.4s** ($28\times$ faster). |
| **07** | [`07_checkpoint_loading.py`](07_checkpoint_loading.py) | [`07_checkpoint_loading-results.md`](07_checkpoint_loading-results.md) | What is the physical disk read bound for the 4.58 GB weights? | Reading and unpickling 4.58 GB takes 20-22s from NVMe storage; one-time cost. |
| **08** | [`08_sample_maker_e2e.py`](08_sample_maker_e2e.py) | [`08_sample_maker_e2e-results.md`](08_sample_maker_e2e-results.md) | What is the total CLI synthesis duration with the fix? | Full CLI run with cold-start init dropped to **63.36s** (and ~18s for warm chunks). |

---

## How to Run Any Profiling Script

All scripts are self-contained and run inside the GPU container with the Hugging Face and PyTorch cache mounts:

```powershell
docker run --gpus all --rm `
    -v "${PWD}:/app" `
    -v "${PWD}/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" `
    -v "${PWD}/.cache/huggingface:/root/.cache/huggingface" `
    -w /app hebtts:latest python profiling/<script_name>.py
```
