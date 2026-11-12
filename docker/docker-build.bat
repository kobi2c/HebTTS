@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"

set "TARGET=%~1"

if /i "%TARGET%"=="main" (
  call :parse_args
  echo Building Main image [make-voice:v1]...
  docker build -f "%ROOT_DIR%\Dockerfile" -t make-voice:v1 %EXTRA_ARGS% "%ROOT_DIR%"
  exit /b %ERRORLEVEL%
)

if /i "%TARGET%"=="hebtts" (
  call :parse_args
  echo Building HebTTS image [make-voice:hebtts]...
  docker build -f "%SCRIPT_DIR%hebtts\Dockerfile" -t make-voice:hebtts %EXTRA_ARGS% "%ROOT_DIR%"
  exit /b %ERRORLEVEL%
)

if /i "%TARGET%"=="zonos" (
  call :parse_args
  echo Building Zonos image [make-voice:zonos]...
  docker build -f "%SCRIPT_DIR%zonos\Dockerfile" -t make-voice:zonos %EXTRA_ARGS% "%ROOT_DIR%"
  exit /b %ERRORLEVEL%
)

echo Building images via docker compose...
echo docker compose -f "%ROOT_DIR%\docker-compose.yml" build %*
exit /b %ERRORLEVEL%


:parse_args
  shift
  set "EXTRA_ARGS="
:args_loop
  if "%~1"=="" exit /b
  set "EXTRA_ARGS=%EXTRA_ARGS% %1"
  shift
  goto args_loop
