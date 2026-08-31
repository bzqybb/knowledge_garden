@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "GARDEN_UNDERSTANDING_PROVIDER=glm"
set "GARDEN_UNDERSTANDING_BASE_URL=https://llmapi.paratera.com"
set "GARDEN_UNDERSTANDING_MODEL=GLM-4.5-AirX"
set "GARDEN_UNDERSTANDING_TIMEOUT_SECONDS=20"
set "GARDEN_CLOSED_LOOP_TIMEOUT_SECONDS=240"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" -BaseUrl "https://llmapi.paratera.com" -Model "GLM-5.2"
if errorlevel 1 pause
