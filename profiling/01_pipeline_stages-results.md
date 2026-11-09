# Benchmark Results: 01_pipeline_stages

## Objective
Measure the exact time spent in every phase of the HebTTS generation pipeline with Multi-Band Diffusion (`mbd=1`) enabled, distinguishing between the one-time cold-start model initialization and subsequent warm chunk synthesis.

---

## Raw Execution Output

```text
======================================================================
HebTTS Profiling: Pipeline Stages & Cached Chunks Benchmark
======================================================================
Loading checkpoint 'checkpoint_01.pt' on cuda:0 (NVIDIA GeForce RTX 5070 Laptop GPU)...
Model loaded successfully.
Initializing AudioTokenizer (MBD=Enabled) on cuda:0...
VALL-E EOS [238 -> 501]

--- Benchmark Results ---
1. Init VoiceMaker/GradioPage:        0.13s
2. Load Models (VALL-E + MBD):        45.09s
3. Chunk 1 VALL-E Inference:          20.06s (8 frames)
4. Chunk 1 MBD Decode:                182.22s
Total Chunk 1 Time:                   247.94s
5. Chunk 2 VALL-E Inference (Warm):   20.33s
6. Chunk 2 MBD Decode (Warm):         162.69s
Total Chunk 2 Time (Warm):            183.02s
```

---

## Structured Results Table

| Pipeline Stage | Chunk 1 (Cold Start) | Chunk 2 (Cached Warm) | % of Total Time |
| :--- | :--- | :--- | :--- |
| **GradioPage / Config Init** | 0.13s | 0.00s | < 0.1% |
| **Model Weight Loading (Disk $\rightarrow$ GPU)** | 45.09s | 0.00s (cached) | 18.2% |
| **VALL-E Autoregressive Generation** | 20.06s | 20.33s | 8.1% |
| **MBD Diffusion Vocoder Decode** | **182.22s** | **162.69s** | **73.5%** |
| **Total Synthesis Time** | **247.94s** | **183.02s** | **100.0%** |

---

## Key Insights
1. **Model Loading is Amortized**:
   - Model loading took **45.09s** on cold start, but **0.00s** for Chunk 2 because weights are held in memory.
   - For batch runs with many chunks, this 45s cost is paid only once.
2. **The Real Bottleneck Was MBD Decode (162s–182s)**:
   - While VALL-E autoregression took a consistent ~20s, MBD diffusion decode consumed **>73% of the total runtime** (~182s for Chunk 1 and ~163s for Chunk 2).
   - Even with warm caches, generating a ~3.5-second audio snippet took nearly 3 minutes.
   - This prompted immediate investigation into why MBD diffusion was so slow on an RTX 5070 GPU.
