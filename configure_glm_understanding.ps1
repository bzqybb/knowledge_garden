$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Knowledge Garden - GLM Understanding Setup"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$credential = Join-Path $projectRoot "data\runtime\understanding-api-key.dpapi"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found: $python"
}

Write-Host "Knowledge Garden - GLM question understanding" -ForegroundColor Cyan
Write-Host "The key is hidden while you type and is stored with Windows DPAPI." -ForegroundColor DarkGray
& $python -m core.credentials save $credential `
    --prompt "Paste the Zhipu GLM API Key (input is hidden): " `
    --saved-label "GLM understanding API key" `
    --show-fingerprint
if ($LASTEXITCODE -ne 0) {
    throw "Credential helper exited with code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $credential)) {
    throw "Encrypted GLM credential was not created."
}

$item = Get-Item -LiteralPath $credential
Write-Host "GLM credential saved successfully ($($item.Length) encrypted bytes)." -ForegroundColor Green
