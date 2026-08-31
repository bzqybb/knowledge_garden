@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [final-verification] Missing .venv. Run bootstrap.cmd -Rebuild first.
  exit /b 1
)

set GARDEN_DISABLE_NETWORK=1
set GARDEN_EVAL_SURFACE_TIMEOUT_SECONDS=180
set GARDEN_CLOSED_LOOP_TIMEOUT_SECONDS=90

echo [final-verification] Phase 1/3: running all unit tests with live output.
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 (
  echo [final-verification] STOPPED in phase 1: unit tests failed.
  exit /b 1
)

echo [final-verification] Phase 2/3: re-running only the previously timed-out TCS-P1-03 case.
".venv\Scripts\python.exe" -m evals.dual_surface_capability_eval --dataset "evals\datasets\zhili_blind_20_v1.jsonl" --ids "TCS-P1-03"
if errorlevel 1 (
  echo [final-verification] STOPPED in phase 2: targeted timeout regression failed.
  exit /b 1
)

echo [final-verification] Phase 3/3: running the 10 transfer cases with independent symbolic checks.
".venv\Scripts\python.exe" -m evals.dual_surface_capability_eval --dataset "evals\datasets\zhili_transfer_15_v1.jsonl" --symbolic-dataset "evals\datasets\zhili_transfer_symbolic_checks_v1.json" --ids "TR-MATH-02,TR-MATH-03,TR-PHYS-02,TR-PHYS-03,TR-CHEM-01,TR-CHEM-02,TR-CHEM-03,TR-BIO-01,TR-BIO-02,TR-BIO-03" --skip-judge
if errorlevel 1 (
  echo [final-verification] STOPPED in phase 3: symbolic transfer run failed.
  exit /b 1
)

echo [final-verification] COMPLETE: unit tests, timeout regression, and symbolic transfer checks finished.
echo [final-verification] Reports are under evals\reports.
