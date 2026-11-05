# HebTTS Docker Usage

### 1. Download Model Checkpoint
```bash
docker run --rm -v %CD%:/app hebtts:latest gdown 11NoOJzMLRX9q1C_Q4sX0w2b9miiDjGrv
```

### 2. Quick Sample Generator (`sample.py`)
```bash
# Default run (shaul speaker, reads sample_he.txt, saves to sample.wav)
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py

# List available speakers
docker run --rm -v %CD%:/app hebtts:latest python sample.py --speaker-list

# List available checkpoints
docker run --rm -v %CD%:/app hebtts:latest python sample.py --checkpoint-list

# Custom text, speaker, and output (supports Hebrew Nikud)
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py --speaker geek --text "שָׁלוֹם עָלֵיכֶם" --output sample.wav

# Synthesize from text file (defaults output to <filename>.wav)
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py --input sample.txt
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