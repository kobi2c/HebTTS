# HebTTS Sample Generator (`sample.py`)

`sample.py` provides an easy-to-use CLI and Python API (`Sampler` class) to synthesize Hebrew speech using the VALL-E model and reference audio prompts.

---

## Quick Start (Docker)

Run commands from the project root directory.

### 1. View Available Speakers and Checkpoints

```bash
# List available speakers (ID, audio prompt, reference text)
docker-run python sample.py --speaker-list

# List predefined checkpoints and their download status
docker-run python sample.py --checkpoint-list
```

### 2. Synthesize from Text File

When using `--input`, the output audio file defaults automatically to the same directory and stem as the input file (e.g. `path/to/sample.txt` -> `path/to/sample.wav`):

```bash
docker-run python sample.py --input sample.txt
```

> **Note**: Multi-line files are synthesized sentence-by-sentence with a 300ms pause inserted between lines, preventing attention length overflow.

### 3. Synthesize Direct Text

```bash
docker-run python sample.py --speaker 1 --text "שלום, זהו מבחן דיבור עבור מודל הדיבור בעברית." --output sample.wav
```


---

## CLI Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--speaker-list` | Flag | `False` | Display formatted list of available speakers |
| `--checkpoint-list`| Flag | `False` | Display predefined checkpoints and their status |
| `--speaker` | `str` | `1` | Speaker number (`1`, `2`, `3`) or speaker ID (`shaul`, `osim`, `geek`) |
| `--checkpoint` | `str` | `1` | Checkpoint number (`1`) or path to `.pt` checkpoint file |
| `--input`, `--in-file` | `str` | `None` | Path to text file (output defaults to `<input_dir>/<input_stem>.wav`) |
| `--text` | `str` | `None` | Direct Hebrew text to synthesize |
| `--output` | `str` | `None` | Custom output `.wav` path (defaults to same directory/stem as input or `sample.wav`) |
| `--clone-wav` | `str` | `None` | Reference audio file path for voice cloning (.wav) |
| `--clone-txt` | `str` | `None` | Reference text file path whose contents provide the clone transcript |
| `--temperature` | `float`| `1.0` | Sampling temperature (lower = more deterministic) |
| `--top-k` | `int` | `50` | Top-K sampling cutoff |
| `--mbd` | Flag | `False` | Enable Multi-Band Diffusion vocoder |
| `--nikud-converter` | Flag | `True` | Enable automatic phonetic Nikud / Ktiv-Male disambiguation (use `--no-nikud-converter` to disable) |

---

## Advanced Examples

### Selecting Speaker by Number or Name
```bash
# Using numeric index (1: osim, 2: geek, 3: shaul)
docker-run python sample.py --speaker 2 --input sample.txt

# Using speaker name
docker-run python python sample.py --speaker shaul --input sample.txt
```

### Custom Voice Cloning
To clone a custom voice, supply both the prompt audio and its corresponding transcript:
```bash
docker-run python sample.py --clone-wav clone.wav --clone-txt clone.txt --input sample.txt
```

#### Preparing Audio Prompts (MP3 to WAV)
HebTTS (VALL-E / EnCodec) operates on **24 kHz mono 16-bit WAV**. You can convert MP3 files using ffmpeg directly through Docker:
```bash
# Convert MP3 to 24kHz mono WAV
docker-run ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav
```

### Hebrew Nikud (Vocalization) Support
When your input text contains Hebrew Nikud (vocalization marks like `שָׁלוֹם`, `סֵפֶר` vs `סַפָּר`), HebTTS automatically disambiguates pronunciation using `phonikud`:
- Vocalized words are converted into phonetically unambiguous Ktiv-Male representations (e.g. inserting appropriate `ו`/`י` vowel markers and mapping Sin `שׂ` to `ס`).
- Unvocalized words remain intact for AlefBERT's contextual tokenizer.
- Enabled by default; pass `--no-nikud-converter` if you wish to pass raw unvocalized text directly.

### Adjusting Generation Parameters
```bash
docker-run python sample.py --input sample.txt --temperature 0.8 --top-k 30 --mbd
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

## Samples

```bash
docker-run python sample.py --speaker-list
docker-run python sample.py --checkpoint-list
docker-run python sample.py
docker-run python sample.py --speaker geek
docker-run python sample.py --speaker 3
docker-run python sample.py --text "שלום לכולם"
docker-run python sample.py --clone-wav samples/clone.wav --clone-txt samples/clone.txt
```
