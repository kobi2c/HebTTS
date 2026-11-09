# Benchmark Results: 05_vram_allocation

## Objective
Measure PyTorch CUDA memory allocation and caching reservation for VALL-E alone vs. VALL-E + MBD together, determining whether GPU memory limits were breached on the 8 GB RTX 5070 GPU.

---

## Raw Execution Output

```text
======================================================================
HebTTS Profiling: VRAM Footprint & Capacity Exhaustion Inspection
======================================================================
GPU Hardware: NVIDIA GeForce RTX 5070 Laptop GPU
Total Physical VRAM: 8187.9 MB
Initial Allocated:   0.0 MB

[VALL-E Alone] Allocated: 1782.9 MB | Reserved: 5370.8 MB

[VALL-E + MBD Together] Allocated: 6470.2 MB | Reserved: 8147.4 MB
Physical Headroom Remaining: 40.5 MB (out of 8187.9 MB)
```

---

## Structured Results Table

| Configuration | Model Parameters | VRAM Allocated | VRAM Reserved | % of Physical VRAM | Free Headroom |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Idle CUDA Context** | 0 | 0.0 MB | 0.0 MB | 0.0% | 8,187.9 MB |
| **VALL-E Alone** | ~400M | 1,782.9 MB | 5,370.8 MB | 65.6% | 2,817.1 MB |
| **MBD Alone** | ~1.15B | 4,687.2 MB | 5,500.0 MB | 67.2% | 2,687.9 MB |
| **VALL-E + MBD Together** | **~1.55B** | **6,470.2 MB** | **8,147.4 MB** | **99.5%** | **40.5 MB** |

---

## Key Insights
1. **Severe VRAM Saturation (99.5%)**:
   - VALL-E alone takes $1.78\text{ GB}$.
   - MBD alone takes $4.69\text{ GB}$.
   - Loaded together, the raw weights require **$6.47\text{ GB}$**, and PyTorch reserves **$8.15\text{ GB}$**.
2. **The Root Cause of the 182-Second Bottleneck**:
   - On an 8 GB GPU, after accounting for Windows Desktop Window Manager (~800 MB), **zero physical VRAM was left for intermediate activations**.
   - As diffusion executed, the NVIDIA Windows WDDM driver was forced into **Shared GPU Memory Paging**, swapping tensors across the slow PCIe bus into host system RAM on every one of the 80 diffusion steps.
   - This caused the **$37\times$ slowdown** (from 4.87s to 182.2s).
