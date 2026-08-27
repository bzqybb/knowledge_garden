param(
    [switch]$Run,
    [switch]$AllowNetwork,
    [int]$Limit = 0,
    [string]$Ids = ""
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到项目虚拟环境：$python"
}

$arguments = @("-m", "evals.general_reasoning_benchmark")
if ($Run) { $arguments += "--run" }
if ($AllowNetwork) { $arguments += "--allow-network" }
if ($Limit -gt 0) { $arguments += @("--limit", "$Limit") }
if ($Ids.Trim()) { $arguments += @("--ids", $Ids.Trim()) }

Push-Location -LiteralPath $projectRoot
try {
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
