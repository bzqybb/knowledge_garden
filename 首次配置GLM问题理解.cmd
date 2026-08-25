@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Knowledge Garden - GLM Understanding Setup
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0configure_glm_understanding.ps1"
set setup_exit=%errorlevel%
echo.
if not "%setup_exit%"=="0" (
  echo GLM setup failed. Please keep this window open and send the error to Codex.
) else (
  echo GLM setup finished. You may return to Knowledge Garden.
)
pause
exit /b %setup_exit%
