@echo off
setlocal
cd /d "%~dp0"
echo [continuous-improvement] Judging opted-in development cases. The sealed holdout is not read.
".venv\Scripts\python.exe" -m evals.continuous_improvement %*
endlocal
