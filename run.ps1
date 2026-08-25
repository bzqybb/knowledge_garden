param(
    [string]$Python = "python",
    [int]$Port = 8765,
    [switch]$AskForApiKey,
    [switch]$SaveApiKey,
    [switch]$ForgetSavedApiKey,
    [switch]$TestDpapi
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
if ($Python -eq "python") {
    $projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $projectPython) { $Python = $projectPython }
}

$runtimeDir = Join-Path $projectRoot "data\runtime"
$credentialPath = Join-Path $runtimeDir "garden-api-key.dpapi"
$keyInjected = $false

function Set-GardenProcessApiKey([Security.SecureString]$SecureKey) {
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureKey)
    try {
        $env:GARDEN_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
}

if ($TestDpapi) {
    & $Python "-m" "core.credentials" "test"
    if ($LASTEXITCODE -ne 0) { throw "Windows DPAPI self-test failed." }
    return
}

if ($ForgetSavedApiKey) {
    if (Test-Path -LiteralPath $credentialPath) {
        Remove-Item -LiteralPath $credentialPath -Force
        Write-Host "Saved Garden API key removed." -ForegroundColor Yellow
    }
    else {
        Write-Host "No saved Garden API key was found." -ForegroundColor Yellow
    }
    return
}

if ($SaveApiKey) {
    & $Python "-m" "core.credentials" "save" $credentialPath
    if ($LASTEXITCODE -ne 0) { throw "The API key could not be saved." }
    try {
        $recheck = Invoke-RestMethod -Method Post -Uri ("http://127.0.0.1:" + $Port + "/api/llm/recheck") -ContentType "application/json" -Body "{}" -TimeoutSec 90
        if ($recheck.ok) {
            Write-Host $recheck.settings.llm_message -ForegroundColor Green
            Write-Host "The running Knowledge Garden has reloaded the saved API key." -ForegroundColor Green
            return
        }
    }
    catch {
        # No compatible Garden service is running yet; continue with normal startup.
    }
}
elseif ($AskForApiKey) {
    $secureKey = Read-Host "Paste the DeepSeek API Key (hidden; cleared when the service stops)" -AsSecureString
    try {
        Set-GardenProcessApiKey $secureKey
    }
    finally {
        $secureKey.Dispose()
    }
    $keyInjected = $true
}
if (-not $env:GARDEN_BASE_URL) { $env:GARDEN_BASE_URL = "https://api.deepseek.com" }
if (-not $env:GARDEN_MODEL) { $env:GARDEN_MODEL = "deepseek-v4-flash" }
if (-not $env:DEEPDIAGRAM_BASE_URL) { $env:DEEPDIAGRAM_BASE_URL = "http://127.0.0.1:8000" }
if (-not $env:DEEPDIAGRAM_TIMEOUT_SECONDS) { $env:DEEPDIAGRAM_TIMEOUT_SECONDS = "45" }

# Start the installed DeepDiagram backend only when its local health check is
# unavailable. It uses an isolated environment and SQLite; no knowledge-vault
# content is scanned during startup.
$deepDiagramBackend = Join-Path $projectRoot "vendor\DeepDiagram\backend"
$deepDiagramPython = Join-Path $deepDiagramBackend ".venv\Scripts\python.exe"
$deepDiagramReady = $false
try {
    $deepDiagramHealth = Invoke-RestMethod -Uri ($env:DEEPDIAGRAM_BASE_URL.TrimEnd('/') + "/") -TimeoutSec 2
    $deepDiagramReady = $deepDiagramHealth.message -match "DeepDiagram"
}
catch {
    $deepDiagramReady = $false
}
if (-not $deepDiagramReady -and (Test-Path -LiteralPath $deepDiagramPython)) {
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    Start-Process -FilePath $deepDiagramPython `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $deepDiagramBackend -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $runtimeDir "deepdiagram.log") `
        -RedirectStandardError (Join-Path $runtimeDir "deepdiagram-error.log")
    $deepDiagramDeadline = (Get-Date).AddSeconds(35)
    do {
        Start-Sleep -Milliseconds 500
        try {
            $deepDiagramHealth = Invoke-RestMethod -Uri ($env:DEEPDIAGRAM_BASE_URL.TrimEnd('/') + "/") -TimeoutSec 2
            $deepDiagramReady = $deepDiagramHealth.message -match "DeepDiagram"
        }
        catch {
            $deepDiagramReady = $false
        }
    } while (-not $deepDiagramReady -and (Get-Date) -lt $deepDiagramDeadline)
}
if ($deepDiagramReady) {
    Write-Host "DeepDiagram full service connected." -ForegroundColor Green
}
else {
    Write-Host "DeepDiagram full service unavailable; Garden will use its safe local diagram fallback." -ForegroundColor Yellow
}
Write-Host "Starting Knowledge Garden..." -ForegroundColor Green
try {
    & $Python "app.py" --port $Port
}
finally {
    if ($keyInjected) { Remove-Item Env:GARDEN_API_KEY -ErrorAction SilentlyContinue }
}
