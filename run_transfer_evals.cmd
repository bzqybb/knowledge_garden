@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [transfer-eval] Missing .venv. Run bootstrap.cmd -Rebuild first.
  exit /b 1
)

set GARDEN_DISABLE_NETWORK=1
set GARDEN_EVAL_SURFACE_TIMEOUT_SECONDS=180
set GARDEN_CLOSED_LOOP_TIMEOUT_SECONDS=90

echo [transfer-eval] Running the independent 15-case transfer pack on both surfaces.
".venv\Scripts\python.exe" -m evals.dual_surface_capability_eval --dataset "evals\datasets\zhili_transfer_15_v1.jsonl" --symbolic-dataset "evals\datasets\zhili_transfer_symbolic_checks_v1.json"
if errorlevel 1 exit /b %errorlevel%

echo [transfer-eval] Complete. Reports are under evals\reports.
