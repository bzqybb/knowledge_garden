@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%迁移Codex存储到D盘.ps1"
if errorlevel 1 (
  echo.
  echo 迁移没有完成，请查看 D:\CodexData\migration.log
  pause
  exit /b 1
)
echo.
echo 迁移完成，详细结果见 D:\CodexData\迁移完成.txt
pause
