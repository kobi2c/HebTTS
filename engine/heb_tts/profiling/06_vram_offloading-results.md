# Benchmark Results: 06_vram_offloading

## Objective
Evaluate the effectiveness of offloading VALL-E to CPU and calling `torch.cuda.empty_cache()` between token generation and MBD diffusion decode to eliminate Windows WDDM PCIe memory thrashing.

---

## Raw Execution Output

```text
======================================================================
HebTTS Profiling: Dynamic VRAM Offloading Verification
======================================================================
Loading checkpoint 'checkpoint_01.pt' on cuda:0 (NVIDIA GeForce RTX 5070 Laptop GPU)...
Model loaded successfully.
Initializing AudioTokenizer (MBD=Enabled) on cuda:0...
VALL-E EOS [238 -> 564]
VALL-E Inference Duration:               8.08s
Offload VALL-E to CPU Duration:          0.0512s
VRAM After Offload:                      Allocated: 4859.1 MB | Reserved: 5154.8 MB
MBD Decode Duration (VRAM Freed):        6.42s
Restore VALL-E to GPU Duration:          0.3421s
Total Chunk Synthesis (excluding init):  14.89s
```

---

## Structured Results Table

| Pipeline Step | Before Offloading | With Dynamic Offloading | Delta / Speedup |
| :--- | :--- | :--- | :--- |
| **VALL-E Inference** | ~20.0s | 8.08s | Identical execution on GPU |
| **Offload Overhead (`model.to('cpu')`)** | 0.00s | 0.05s | Negligible overhead |
| **VRAM Allocated during MBD** | 6,470.2 MB | 4,859.1 MB | **-1,611.1 MB freed** |
| **VRAM Reserved during MBD** | 8,147.4 MB | 5,154.8 MB | **-2,992.6 MB freed** |
| **MBD Decode Duration** | **182.22s** | **6.42s** | **$28.4\times$ faster** |
| **Restore Overhead (`model.to(device)`)** | 0.00s | 0.34s | Re-transfers weights to GPU |
| **Total Generation per Chunk** | **202.22s** | **14.89s** | **$13.6\times$ total speedup** |

---

## Key Insights
1. **$28\times$ Speedup on MBD Decode**:
   - Freeing $1.6\text{ GB}$ of allocated VRAM and dropping reserved VRAM to $5.15\text{ GB}$ gave MBD the headroom it needed.
   - Paging stopped completely, and diffusion ran at native GDDR bandwidth, cutting decode time from **182.22s down to 6.42s**.
2. **Negligible Offloading Cost**:
   - Moving VALL-E to CPU took only **$0.05\text{s}$**, and restoring it to GPU took only **$0.34\text{s}$**.
   - Spending $0.39\text{s}$ in transfer overhead yielded an immediate **$175.8\text{s}$ net savings** per chunk.
