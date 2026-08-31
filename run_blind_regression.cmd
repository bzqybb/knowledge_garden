@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [blind-regression] Missing .venv. Run bootstrap.cmd -Rebuild first.
  exit /b 1
)

set GARDEN_DISABLE_NETWORK=1
set GARDEN_EVAL_SURFACE_TIMEOUT_SECONDS=180
set GARDEN_CLOSED_LOOP_TIMEOUT_SECONDS=90

echo [blind-regression] Re-running the original 20-case blind pack after router fixes.
".venv\Scripts\python.exe" -m evals.dual_surface_capability_eval --dataset "evals\datasets\zhili_blind_20_v1.jsonl"
if errorlevel 1 exit /b %errorlevel%

echo [blind-regression] Complete. Reports are under evals\reports.
