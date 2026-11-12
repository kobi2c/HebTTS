set image=zonos:v1
docker run --rm --gpus=all -v "%cd%:/app" -v zonos_cache:/root/.cache/huggingface %image% python3 make-voice-1.py %*