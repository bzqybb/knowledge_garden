param(
    [string]$ResumeCheckpoint = "",
    [string]$InputReport = "",
    [int]$Limit = 0,
    [switch]$SkipJudge,
    [switch]$RejudgeCompleted
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
. (Join-Path $ProjectRoot "project_storage.ps1")

$env:GARDEN_DISABLE_NETWORK = "1"
$env:GARDEN_EVAL_SURFACE_TIMEOUT_SECONDS = "300"
$env:GARDEN_CLOSED_LOOP_TIMEOUT_SECONDS = "120"
$env:GARDEN_EVAL_JUDGE_TIMEOUT_SECONDS = "240"
$env:JUDGE_MODEL = "deepseek-v4-pro"
$env:JUDGE_BASE_URL = "https://api.deepseek.com"
$env:JUDGE_USE_GENERATOR_CREDENTIAL = "false"
$env:PYTHONUNBUFFERED = "1"
$env:GARDEN_MODEL = "glm-5.2"
$env:GARDEN_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
$env:EVAL_EXPECTED_GENERATOR_MODEL = "glm-5.2"
$env:EVAL_EXPECTED_GENERATOR_HOST = "open.bigmodel.cn"
$env:EVAL_EXPECTED_JUDGE_MODEL = "deepseek-v4-pro"
$env:EVAL_EXPECTED_JUDGE_HOST = "api.deepseek.com"

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python virtual environment is missing: $Python"
}

$EvalArgs = @(
    "-m", "evals.dual_surface_capability_eval",
    "--dataset", "evals\datasets\science_exploration_100_v1.jsonl"
)
if ($ResumeCheckpoint) {
    $EvalArgs += @("--resume-checkpoint", $ResumeCheckpoint)
}
if ($InputReport) {
    $EvalArgs += @("--input-report", $InputReport)
}
if ($Limit -gt 0) {
    $EvalArgs += @("--limit", [string]$Limit)
}
if ($SkipJudge) {
    $EvalArgs += "--skip-judge"
}
if ($RejudgeCompleted) {
    $EvalArgs += "--rejudge-completed"
}

Write-Host "[science-100] tested_model=glm-5.2"
Write-Host "[science-100] model_judge=deepseek-v4-pro"
Write-Host "[science-100] independent_audit=Codex child agent (reported separately)"
Write-Host "[science-100] reports=$ProjectRoot\evals\reports"
& $Python @EvalArgs
exit $LASTEXITCODE
