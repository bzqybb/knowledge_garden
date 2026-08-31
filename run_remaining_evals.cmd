@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [remaining-evals] Missing .venv. Run bootstrap.cmd -Rebuild first.
  exit /b 1
)

echo [remaining-evals] Phase 1/3: running the complete unit-test suite with live output.
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 (
  echo [remaining-evals] STOPPED in phase 1: unit tests failed.
  exit /b 1
)

echo [remaining-evals] Phase 2/3: re-running the original 20-case blind pack.
call "%~dp0run_blind_regression.cmd"
if errorlevel 1 (
  echo [remaining-evals] STOPPED in phase 2: blind regression failed.
  exit /b 1
)

echo [remaining-evals] Phase 3/3: running the independent 15-case transfer pack.
call "%~dp0run_transfer_evals.cmd"
if errorlevel 1 (
  echo [remaining-evals] STOPPED in phase 3: transfer evaluation failed.
  exit /b 1
)

echo [remaining-evals] COMPLETE: tests, blind regression, and transfer evaluation all finished.
echo [remaining-evals] Reports are under evals\reports.
