# HebTTS Sample Generator (`sample.py`)

`sample.py` provides an easy-to-use CLI and Python API (`Sampler` class) to synthesize Hebrew speech using the VALL-E model and reference audio prompts.

---

## Quick Start (Docker)

Run commands from the project root directory.

### 1. View Available Speakers and Checkpoints

```bash
# List available speakers (ID, audio prompt, reference text)
docker run --rm -v %CD%:/app hebtts:latest python sample.py --speaker-list

# List predefined checkpoints and their download status
docker run --rm -v %CD%:/app hebtts:latest python sample.py --checkpoint-list
```

### 2. Synthesize from Text File

When using `--input`, the output audio file defaults automatically to the base name of the input file (e.g. `sample.txt` -> `sample.wav`):

```bash
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py --input sample.txt
```

> **Note**: Multi-line files are synthesized sentence-by-sentence with a 300ms pause inserted between lines, preventing attention length overflow.

### 3. Synthesize Direct Text

```bash
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py --speaker 1 --text "שלום, זהו מבחן דיבור עבור מודל הדיבור בעברית." --output sample.wav
```

docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py --speaker 1 \
  --text "שלום, זהו מבחן דיבור עבור מודל הדיבור בעברית." \
  --output sample.wav

---

## CLI Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--speaker-list` | Flag | `False` | Display formatted list of available speakers |
| `--checkpoint-list`| Flag | `False` | Display predefined checkpoints and their status |
| `--speaker` | `str` | `1` | Speaker number (`1`, `2`, `3`) or speaker ID (`shaul`, `osim`, `geek`) |
| `--checkpoint` | `str` | `1` | Checkpoint number (`1`) or path to `.pt` checkpoint file |
| `--input`, `--in-file` | `str` | `None` | Path to text file (output defaults to `<stem>.wav`) |
| `--text` | `str` | `None` | Direct Hebrew text to synthesize |
| `--output` | `str` | `None` | Custom output `.wav` path (defaults to input stem or `sample.wav`) |
| `--speaker-audio` | `str` | `None` | Path to custom reference audio for voice cloning |
| `--speaker-text` | `str` | `None` | Transcript of the custom reference audio |
| `--temperature` | `float`| `1.0` | Sampling temperature (lower = more deterministic) |
| `--top-k` | `int` | `50` | Top-K sampling cutoff |
| `--mbd` | Flag | `False` | Enable Multi-Band Diffusion vocoder |

---

## Advanced Examples

### Selecting Speaker by Number or Name
```bash
# Using numeric index (1: osim, 2: geek, 3: shaul)
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py --speaker 2 --input sample.txt

# Using speaker name
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py --speaker shaul --input sample.txt
```

### Custom Voice Cloning
To clone a custom voice, supply both the prompt audio and its corresponding transcript:
```bash
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py \
  --speaker-audio my_voice.wav \
  --speaker-text "טקסט התואם להקלטת הקול" \
  --input sample.txt \
  --output cloned_output.wav
```

### Adjusting Generation Parameters
```bash
docker run --gpus all --rm -v %CD%:/app hebtts:latest python sample.py \
  --input sample.txt \
  --temperature 0.8 \
  --top-k 30 \
  --mbd
```

---

## Python API Usage

The `Sampler` class can also be integrated directly into Python scripts:

```python
from sample import Sampler

sampler = Sampler()

# Run via programmatic arguments
sampler.run([
    "--speaker", "shaul",
    "--input", "sample.txt",
    "--output", "output.wav"
])
```
