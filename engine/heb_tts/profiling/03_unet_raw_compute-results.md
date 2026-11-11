# Benchmark Results: 03_unet_raw_compute

## Objective
Measure the raw GPU tensor computation time of a single forward pass of the 286M parameter U-Net to determine whether the 182-second MBD delay was due to intrinsic GPU compute bottlenecks or external interference.

---

## Raw Execution Output

```text
======================================================================
HebTTS Profiling: Single U-Net Forward Pass Raw GPU Compute
======================================================================
Audio Length:                        4.0s (96,000 samples)
Conditioning Frames:                 300 frames (128-dim)
U-Net Parameters:                    286,000,000
Average time per single pass:        0.0405s
Theoretical total for 80 passes:     3.24s (4 bands x 20 steps)
```

---

## Structured Results Table

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Model Size** | 286,000,000 parameters | Single band 1D U-Net diffusion architecture |
| **Input Audio Size** | 96,000 samples (4.0s @ 24kHz) | Waveform dimension `[1, 1, 96000]` |
| **Conditioning Size** | 300 frames @ 75Hz | Acoustic embedding dimension `[1, 128, 300]` |
| **Single Forward Pass Latency** | **0.0405s** (40.5 ms) | Average over 10 synchronized GPU iterations |
| **Diffusion Loop Math (80 Passes)** | **3.24s** | 4 frequency bands $\times$ 20 subsampled steps |

---

## Key Insights
1. **GPU Compute is Blazing Fast**:
   - The RTX 5070 Laptop GPU computes one complete 286M-parameter U-Net forward pass in just **40.5 milliseconds**.
   - Doing all 80 diffusion evaluations takes only **3.24 seconds** of raw GPU compute.
2. **The Discrepancy**:
   - $3.24\text{s}$ theoretical vs. $182.22\text{s}$ observed represents a **$56\times$ slowdown**.
   - The GPU was clearly NOT compute-bound; something was stalling execution between or during iterations.
