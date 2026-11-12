@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT_DIR=%%~fI"

echo Building HebTTS image (make-voice:hebtts)...
docker build -f "%SCRIPT_DIR%Dockerfile" -t make-voice:hebtts %* "%ROOT_DIR%"

