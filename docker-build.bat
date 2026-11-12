@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if /i "%~1"=="hebtts" (
    shift
    echo Building HebTTS image (make-voice:hebtts)...
    docker build -f docker/hebtts/Dockerfile -t make-voice:hebtts . %*
) else if /i "%~1"=="zonos" (
    shift
    echo Building Zonos image (make-voice:zonos)...
    docker build -f docker/zonos/Dockerfile -t make-voice:zonos . %*
) else (
    echo Building images via docker-compose...
    docker compose build %*
)

