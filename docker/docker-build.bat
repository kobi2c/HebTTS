@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"

if /i "%~1"=="v1" (
    shift
    echo Building Unified image [make-voice:v1]...
    docker build -f "%ROOT_DIR%\Dockerfile" -t make-voice:v1 "%ROOT_DIR%" %*
) else if /i "%~1"=="hebtts" (
    shift
    echo Building HebTTS image [make-voice:hebtts]...
    docker build -f "%SCRIPT_DIR%hebtts\Dockerfile" -t make-voice:hebtts "%ROOT_DIR%" %*
) else if /i "%~1"=="zonos" (
    shift
    echo Building Zonos image [make-voice:zonos]...
    docker build -f "%SCRIPT_DIR%zonos\Dockerfile" -t make-voice:zonos "%ROOT_DIR%" %*
) else (
    echo Building images via docker-compose...
    docker compose -f "%ROOT_DIR%\docker-compose.yml" build %*
)

