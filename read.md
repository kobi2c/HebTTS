# HebTTS Docker Usage

### 1. Download Model Checkpoint
```bash
docker run --rm -v %CD%:/app hebtts:latest gdown 11NoOJzMLRX9q1C_Q4sX0w2b9miiDjGrv
```

### 2. Quick Sample Generator (`sample.py`)
```bash
# Default run (shaul speaker, reads sample.txt, saves to sample.wav)
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py

# List available speakers
docker run --rm -v %CD%:/app hebtts:latest python sample.py --speaker-list

# List available checkpoints
docker run --rm -v %CD%:/app hebtts:latest python sample.py --checkpoint-list

# Custom text, speaker, and output (supports Hebrew Nikud)
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py --speaker geek --text "שָׁלוֹם עָלֵיכֶם" --output sample.wav

# Synthesize from text file (defaults output to <filename>.wav)
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py --input samples/sample.txt
```

### 3. Advanced Inference (`infer.py`)
```bash
docker run --gpus all --rm -v %CD%:/app hebtts:latest python infer.py --checkpoint checkpoint.pt --output-dir ./out --text "שלום עולם"
```

### 4. Gradio Web Interface (`GradioPage.py`)
```bash
docker run --gpus all --rm -p 7860:7860 -v %CD%:/app hebtts:latest python GradioPage.py
```
Open your browser at `http://localhost:7860`.

### 5. Batch Voice & Subtitle Generator (`make_voice.py`)
Synthesize speech and generate synchronized `.srt` subtitles from `.mki` script files containing dynamic voice directives (`@` for updates, `!` for reset):
```bash
docker run --gpus all --rm -v %CD%:/app hebtts:latest python make_voice.py -i samples/sample.mki
```
Outputs are automatically written to `<input_dir>/out/`:
- `{basename}_{idx:03d}.wav` & `{basename}_{idx:03d}.txt`: individual sentence chunks.
- `{basename}.wav`: full merged audio with inter-sentence pauses.
- `{basename}.srt`: synchronized subtitle file.
