set image=zonos
docker run --rm --gpus=all -v "%cd%:/app" -v zonos_cache:/root/.cache/huggingface %image% python3 make-voice-2.py %*