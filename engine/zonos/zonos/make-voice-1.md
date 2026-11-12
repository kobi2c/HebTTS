# Make Voice with Docker

This document explains how to run the `make_voice.py` script inside a Docker container. This is useful if you want to generate audio without having to install the PyTorch and CUDA dependencies on your local machine directly.

## 1. Build the Docker Image

First, build the docker image using the provided `Dockerfile`. You only need to do this once, or whenever the dependencies change. Run this from the root of the project:

```bash
docker build -t zonos .
```

## 2. Run the Script

To run `make_voice.py`, you need to pass your GPU to the container, and mount a volume so that the input files (`input.txt`, `config.json`) are accessible inside the container, and the output `wav` files are saved back to your host machine.

Assuming your current directory contains `input.txt` and `config.json`, run:

```bash
docker run --gpus all -v "$(pwd):/app/data" zonos python make_voice.py --text /app/data/input.txt --config /app/data/config.json --output_dir /app/data/output_wavs
# docker run --rm --gpus=all -v "%cd%:/app" -v zonos_cache:/root/.cache/huggingface zonos python make_voice.py --text /app/data/a.txt --config /app/data/config1.json --output_dir /app/data/
```



### Breakdown of the command:
* `--gpus all`: Enables the container to use your host's NVIDIA GPUs for fast generation.
* `-v "$(pwd):/app/data"`: Mounts your current directory (`pwd`) into the container at `/app/data`. This allows the script to read your text/config and write the audio files back to your machine.
* `python make_voice.py ...`: The actual command that runs inside the container, pointing to the paths within the mounted `/app/data` directory.

> **Note for Windows users**: If you are using Command Prompt or PowerShell, replace `$(pwd)` with `%cd%` (CMD) or `${PWD}` (PowerShell).
