@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT_DIR=%%~fI"

if not exist "%ROOT_DIR%\.cache\torch\hub\checkpoints" (
    mkdir "%ROOT_DIR%\.cache\torch\hub\checkpoints"
)
if not exist "%ROOT_DIR%\.cache\huggingface" (
    mkdir "%ROOT_DIR%\.cache\huggingface"
)

if "%~1"=="" (
    docker run --gpus all --rm -it -v "%ROOT_DIR%:/app" -v "%ROOT_DIR%\.cache\torch\hub\checkpoints:/root/.cache/torch/hub/checkpoints" -v "%ROOT_DIR%\.cache\huggingface:/root/.cache/huggingface" -p 7861:7860 make-voice:hebtts python GradioPage.py
) else (
    docker run --gpus all --rm -it -v "%ROOT_DIR%:/app" -v "%ROOT_DIR%\.cache\torch\hub\checkpoints:/root/.cache/torch/hub/checkpoints" -v "%ROOT_DIR%\.cache\huggingface:/root/.cache/huggingface" make-voice:hebtts %*
)
