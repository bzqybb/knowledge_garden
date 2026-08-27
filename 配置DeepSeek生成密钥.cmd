@echo off
".venv\Scripts\python.exe" -X utf8 -m core.credentials save "data\runtime\garden-api-key.dpapi" --prompt "Paste official DeepSeek API Key (hidden): " --saved-label "Official DeepSeek generator API key" --show-fingerprint
pause
