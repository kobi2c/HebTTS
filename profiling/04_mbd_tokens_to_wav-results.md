# Benchmark Results: 04_mbd_tokens_to_wav

## Objective
Isolate the sub-components within `MultiBandDiffusion.tokens_to_wav` when MBD is loaded alone into GPU memory, determining which step takes how long under optimal memory conditions.

---

## Raw Execution Output

```text
======================================================================
HebTTS Profiling: MBD tokens_to_wav Sub-component Breakdown
======================================================================

--- Sub-component Timing Results ---
1. EnCodec decode:           0.2404s
2. Embedding projection:     0.0005s
3. DP[0] generate:           1.4734s
3. DP[1] generate:           0.8639s
3. DP[2] generate:           1.1175s
3. DP[3] generate:           0.8275s
4. Multiband Re-EQ:          0.3462s
----------------------------------------
Total Standalone Decode:     4.8696s
```

---

## Structured Results Table

| Pipeline Component | Duration (s) | % of Decode Time | Purpose |
| :--- | :--- | :--- | :--- |
| **1. EnCodec Decoder** | 0.2404s | 4.9% | Converts discrete codes to reference EnCodec audio |
| **2. Embedding Projection** | 0.0005s | < 0.1% | Prepares conditioning tensor from acoustic codes |
| **3. Band 0 Diffusion (`DP[0]`)** | 1.4734s | 30.3% | 20 diffusion steps for low frequencies (0-3 kHz) |
| **4. Band 1 Diffusion (`DP[1]`)** | 0.8639s | 17.7% | 20 diffusion steps for mid frequencies (3-6 kHz) |
| **5. Band 2 Diffusion (`DP[2]`)** | 1.1175s | 23.0% | 20 diffusion steps for high-mid frequencies (6-9 kHz) |
| **6. Band 3 Diffusion (`DP[3]`)** | 0.8275s | 17.0% | 20 diffusion steps for high frequencies (9-12 kHz) |
| **7. Julius Multiband Re-EQ** | 0.3462s | 7.1% | 32-band equalizing filter matching reference audio |
| **Total Standalone MBD Decode** | **4.8696s** | **100.0%** | Full waveform synthesis from acoustic tokens |

---

## Key Insights
1. **MBD Alone Takes Only 4.87s**:
   - When Multi-Band Diffusion operates without VALL-E sharing GPU memory, **all 4 frequency bands and 80 diffusion steps finish in less than 5 seconds**.
2. **The "Smoking Gun"**:
   - In `01_pipeline_stages`, MBD took **182.22s**.
   - Here, the exact same code took **4.87s**.
   - The only difference: In `01_pipeline_stages`, the VALL-E model was also loaded into GPU memory. This pointed directly to VRAM exhaustion and driver memory paging.
