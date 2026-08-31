@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_science_100_evals.ps1" %*
exit /b %ERRORLEVEL%
