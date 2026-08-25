param(
    [int]$Limit = 5,
    [string]$Model = "kimi-k2.5",
    [string]$Dataset = "",
    [string]$InputReport = "",
    [switch]$CheckJudge,
    [switch]$ResumeScores
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $projectRoot "run_evals.ps1") `
    -Limit $Limit `
    -JudgeBaseUrl "https://api.lkeap.cloud.tencent.com/coding/v3" `
    -JudgeModel $Model `
    -Dataset $Dataset `
    -InputReport $InputReport `
    -CheckJudge:$CheckJudge `
    -ResumeScores:$ResumeScores
