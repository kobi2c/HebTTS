@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if not exist "%SCRIPT_DIR%.cache\torch\hub\checkpoints" (
    mkdir "%SCRIPT_DIR%.cache\torch\hub\checkpoints"
)

if "%~1"=="" (
    docker run --gpus all --rm -it -v "%SCRIPT_DIR%:/app" -v "%SCRIPT_DIR%.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" -p 7860:7860 hebtts:latest python GradioPage.py
) else (
    docker run --gpus all --rm -it -v "%SCRIPT_DIR%:/app" -v "%SCRIPT_DIR%.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints" hebtts:latest %*
)
