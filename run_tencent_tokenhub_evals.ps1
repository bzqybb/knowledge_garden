param(
    [int]$Limit = 5,
    [string]$Model = "kimi-k2.6",
    [string]$Dataset = "",
    [string]$InputReport = "",
    [switch]$CheckJudge,
    [switch]$ResumeScores
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $projectRoot "run_evals.ps1") `
    -Limit $Limit `
    -JudgeBaseUrl "https://tokenhub.tencentmaas.com/v1" `
    -JudgeModel $Model `
    -Dataset $Dataset `
    -InputReport $InputReport `
    -CheckJudge:$CheckJudge `
    -ResumeScores:$ResumeScores
