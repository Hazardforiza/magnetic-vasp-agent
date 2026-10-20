@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo uv was not found. Install uv from https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

uv sync --extra agent --extra web --frozen
if errorlevel 1 (
    echo Dependency setup failed.
    pause
    exit /b 1
)

uv run --extra agent --extra web --frozen magvasp web
if errorlevel 1 pause
