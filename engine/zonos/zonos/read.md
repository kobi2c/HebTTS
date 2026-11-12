
```
docker build -t zonos .
docker run --rm --gpus=all -v "%cd%:/app" -v zonos_cache:/root/.cache/huggingface zonos bash -c "python sample.py"
```

```
docker build -t zonos .
docker run --rm --gpus=all -p 7860:7860 -v "%cd%:/app" -v zonos_cache:/root/.cache/huggingface zonos python gradio_interface.py
```