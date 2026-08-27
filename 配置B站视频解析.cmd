@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 未找到项目 Python 环境：.venv\Scripts\python.exe
  pause
  exit /b 1
)
echo B站凭证只会在这个本地窗口中输入，并保存到 D 盘项目 vendor\bilibili-home。
echo 请勿把 SESSDATA、bili_jct 或 DedeUserID 发到聊天中。
echo.
".venv\Scripts\python.exe" -m core.bilibili_mcp setup
echo.
echo 配置流程已结束。请回到知识花园刷新 Bilibili MCP 状态。
pause
