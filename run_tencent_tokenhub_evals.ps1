param(
    [int]$Limit = 5,
    [string]$InputReport = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $projectRoot "run_evals.ps1") `
    -Limit $Limit `
    -JudgeBaseUrl "https://tokenhub.tencentmaas.com/v1" `
    -JudgeModel "kimi-k3" `
    -InputReport $InputReport
