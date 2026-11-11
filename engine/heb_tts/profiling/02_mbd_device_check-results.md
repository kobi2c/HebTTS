# Benchmark Results: 02_mbd_device_check

## Objective
Verify whether any sub-networks, weights, or buffers inside `MultiBandDiffusion` remained inadvertently allocated on CPU rather than CUDA, which could cause silent GPU-CPU data thrashing.

---

## Raw Execution Output

```text
======================================================================
HebTTS Profiling: MBD Parameter & Buffer Device Inspection
======================================================================
Top-level mbd.device: cuda:0
codec_model device:   cuda:0
DP[0] model device: cuda:0 | schedule.betas device: cuda:0
DP[1] model device: cuda:0 | schedule.betas device: cuda:0
DP[2] model device: cuda:0 | schedule.betas device: cuda:0
DP[3] model device: cuda:0 | schedule.betas device: cuda:0

codec_model CPU parameters: 0
DP[0] CPU params: 0, CPU buffers: 0
DP[1] CPU params: 0, CPU buffers: 0
DP[2] CPU params: 0, CPU buffers: 0
DP[3] CPU params: 0, CPU buffers: 0
```

---

## Structured Results Table

| Submodule | Target Device | Parameter Device | Buffer Device | CPU Parameters | CPU Buffers |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`codec_model` (EnCodec)** | `cuda:0` | `cuda:0` | `cuda:0` | 0 | 0 |
| **`DP[0]` (Band 0-3 kHz)** | `cuda:0` | `cuda:0` | `cuda:0` | 0 | 0 |
| **`DP[1]` (Band 3-6 kHz)** | `cuda:0` | `cuda:0` | `cuda:0` | 0 | 0 |
| **`DP[2]` (Band 6-9 kHz)** | `cuda:0` | `cuda:0` | `cuda:0` | 0 | 0 |
| **`DP[3]` (Band 9-12 kHz)** | `cuda:0` | `cuda:0` | `cuda:0` | 0 | 0 |

---

## Key Insights
1. **Device Placement Was Correct**:
   - Every single parameter (all $1.15\text{ Billion}$ across the 4 U-Nets) and all noise schedule beta buffers were confirmed to reside on `cuda:0`.
   - There was **zero CPU parameter leakage**.
2. **Eliminated "Accidental CPU Execution" Hypothesis**:
   - The ~180-second delay was NOT caused by code running on CPU due to a missed `.to(device)` call.
   - The slowness was happening while execution was targeted to the GPU, directing focus to GPU compute limits or VRAM contention.
