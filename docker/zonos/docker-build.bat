@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT_DIR=%%~fI"

echo Building Zonos image (make-voice:zonos)...
docker build -f "%SCRIPT_DIR%Dockerfile" -t make-voice:zonos %* "%ROOT_DIR%"

