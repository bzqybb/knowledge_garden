param(
    [int]$Limit = 5,
    [string]$Model = "kimi-k2.5",
    [string]$InputReport = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $projectRoot "run_evals.ps1") `
    -Limit $Limit `
    -JudgeBaseUrl "https://api.lkeap.cloud.tencent.com/coding/v3" `
    -JudgeModel $Model `
    -InputReport $InputReport
