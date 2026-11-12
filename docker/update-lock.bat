@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%pyproject.toml" (
  set "ROOT_DIR=%SCRIPT_DIR:~0,-1%"
) else (
  for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"
)

if "%UV_IMAGE%"=="" (
  set "UV_IMAGE=ghcr.io/astral-sh/uv:debian"
)

set "TARGET=%~1"

if /i "%TARGET%"=="main" (
  call :parse_args %*
  call :lock_main
  exit /b %ERRORLEVEL%
)

if /i "%TARGET%"=="unified" (
  call :parse_args %*
  call :lock_main
  exit /b %ERRORLEVEL%
)

if /i "%TARGET%"=="v1" (
  call :parse_args %*
  call :lock_main
  exit /b %ERRORLEVEL%
)

if /i "%TARGET%"=="hebtts" (
  call :parse_args %*
  call :lock_hebtts
  exit /b %ERRORLEVEL%
)

if /i "%TARGET%"=="zonos" (
  call :parse_args %*
  call :lock_zonos
  exit /b %ERRORLEVEL%
)

if /i "%TARGET%"=="all" (
  call :parse_args %*
) else (
  set "EXTRA_ARGS=%*"
)

call :lock_main
if errorlevel 1 exit /b %ERRORLEVEL%

call :lock_hebtts
if errorlevel 1 exit /b %ERRORLEVEL%

call :lock_zonos
if errorlevel 1 exit /b %ERRORLEVEL%

echo.
echo Lockfiles updated successfully!
exit /b 0


:: ==================================================
:: Methods
:: ==================================================

:lock_main
  echo ==================================================
  echo Updating lockfile for Main [root]...
  echo ==================================================
  docker run --rm --entrypoint uv -v "%ROOT_DIR%:/app" -w /app %UV_IMAGE% lock %EXTRA_ARGS%
  exit /b %ERRORLEVEL%

:lock_hebtts
  echo ==================================================
  echo Updating lockfile for HebTTS [docker/hebtts]...
  echo ==================================================
  docker run --rm --entrypoint uv -v "%ROOT_DIR%:/app" -w /app/docker/hebtts %UV_IMAGE% lock %EXTRA_ARGS%
  exit /b %ERRORLEVEL%

:lock_zonos
  echo ==================================================
  echo Updating lockfile for Zonos [docker/zonos]...
  echo ==================================================
  docker run --rm --entrypoint uv -v "%ROOT_DIR%:/app" -w /app/docker/zonos %UV_IMAGE% lock %EXTRA_ARGS%
  exit /b %ERRORLEVEL%

:parse_args
  shift
  set "EXTRA_ARGS="
:args_loop
  if "%~1"=="" exit /b
  set "EXTRA_ARGS=%EXTRA_ARGS% %1"
  shift
  goto args_loop

