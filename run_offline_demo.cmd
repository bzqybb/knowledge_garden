@echo off
setlocal
cd /d "%~dp0"

set GARDEN_DISABLE_NETWORK=1
set GARDEN_DISABLE_SAVED_API_KEY=1
set GARDEN_API_KEY=
set GARDEN_UNDERSTANDING_API_KEY=

if not exist ".venv\Scripts\python.exe" (
  echo [offline-demo] Missing .venv. Run bootstrap.cmd -SkipTests first.
  exit /b 1
)

echo [offline-demo] Network and saved model keys are disabled for this process.
echo [offline-demo] Open http://127.0.0.1:8765 after the server starts.
".venv\Scripts\python.exe" app.py
