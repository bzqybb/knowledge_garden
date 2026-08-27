@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_tencent_tokenhub_evals.ps1" -Model "deepseek-v4-flash-0731"
pause
