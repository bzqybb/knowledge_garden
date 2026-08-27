@echo off
".venv\Scripts\python.exe" -X utf8 -m core.credentials save "data\runtime\glm-eval-api-key.dpapi" --prompt "Paste school GLM API Key (hidden): " --saved-label "GLM evaluator API key" --show-fingerprint
pause
