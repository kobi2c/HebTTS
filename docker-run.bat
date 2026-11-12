@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if not exist "%SCRIPT_DIR%.cache\torch\hub\checkpoints" (
    mkdir "%SCRIPT_DIR%.cache\torch\hub\checkpoints"
)
if not exist "%SCRIPT_DIR%.cache\huggingface" (
    mkdir "%SCRIPT_DIR%.cache\huggingface"
)

if "%~1"=="" (
    docker run --gpus all --rm -it -v "%SCRIPT_DIR%:/app" -v "%SCRIPT_DIR%.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" -v "%SCRIPT_DIR%.cache/huggingface:/root/.cache/huggingface" -p 7861:7860 make-voice:hebtts python GradioPage.py
) else (
    docker run --gpus all --rm -it -v "%SCRIPT_DIR%:/app" -v "%SCRIPT_DIR%.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" -v "%SCRIPT_DIR%.cache/huggingface:/root/.cache/huggingface" make-voice:hebtts %*
)
