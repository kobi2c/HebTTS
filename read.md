# HebTTS Docker Usage

Mounting `%CD%:/app` ensures that scripts and generated outputs sync with your host workspace. Mounting `%CD%/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints` and `%CD%/.cache/huggingface:/root/.cache/huggingface` ensures downloaded PyTorch / EnCodec and HuggingFace / Multi-Band Diffusion (MBD) weights are cached persistently across container runs.

> [!TIP]
> You can also use the included helper scripts **`docker-run.bat`** (Windows) or **`./docker-run.sh`** (Linux / macOS / WSL) to run any command with all GPU and volume settings preconfigured:
> ```bash
> # Run default sample:
> docker-run.bat
> # Or on Linux/macOS:
> ./docker-run.sh
>
> # Pass custom commands/arguments:
> docker-run.bat python make_sample.py --speaker geek --text "שלום עולם"
> ```
> Or use Docker Compose:
> ```bash
> docker compose up
> ```

---

### 1. Gradio Web Interface (`GradioPage.py`)
```bash
docker-run  python GradioPage.py
```
Open your browser at `http://localhost:7860`.

### 2. Quick Sample Generator (`make_sample.py`)
```bash
# Default run (shaul speaker, reads samples/sample.txt, saves to samples/sample.wav)
docker-run python make_sample.py

# List available speakers
docker-run python make_sample.py --speaker-list

# List available checkpoints
docker-run python make_sample.py --checkpoint-list

# Custom text, speaker, and output (supports Hebrew Nikud)
docker-run python make_sample.py --speaker geek --text "שָׁלוֹם עָלֵיכֶם" --output sample.wav

# Synthesize from text file (defaults output to <filename>.wav)
docker-run python make_sample.py --input samples/sample.txt
```

### 3. Download Model Checkpoint
```bash
docker-run gdown 11NoOJzMLRX9q1C_Q4sX0w2b9miiDjGrv
```


### 4. Advanced Inference (`infer.py`)
```bash
docker-run infer.py --checkpoint checkpoint.pt --output-dir ./out --text "שלום עולם"
```

### 5. Batch Voice & Subtitle Generator (`make_voice.py`)
Synthesize speech and generate synchronized `.srt` subtitles from `.mki` script files containing dynamic voice directives (`@` for updates, `!` for reset):
```bash
docker-run python make_voice.py -i samples/sample.mki
```
Outputs are automatically written to `<input_dir>/out/`:
- `{basename}_{idx:03d}.wav` & `{basename}_{idx:03d}.txt`: individual sentence chunks.
- `{basename}.wav`: full merged audio with inter-sentence pauses.
- `{basename}.srt`: synchronized subtitle file.
