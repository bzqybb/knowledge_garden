param(
    [string]$Python = "python",
    [switch]$SaveKimiKey,
    [switch]$RetrievalOnly,
    [switch]$SkipJudge,
    [Alias("KimiBaseUrl")]
    [string]$JudgeBaseUrl = "",
    [string]$JudgeModel = "",
    [string]$InputReport = "",
    [switch]$CheckJudge,
    [switch]$ResumeScores,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
if ($Python -eq "python") {
    $projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $projectPython) { $Python = $projectPython }
}

$env:RAGAS_DO_NOT_TRACK = "true"
$env:GARDEN_DISABLE_NETWORK = "1"
if ($JudgeBaseUrl) { $env:KIMI_BASE_URL = $JudgeBaseUrl }
if ($JudgeModel) { $env:KIMI_EVAL_MODEL = $JudgeModel }
$credentialPath = Join-Path $projectRoot "data\runtime\kimi-eval-api-key.dpapi"

if ($SaveKimiKey) {
    & $Python "-m" "core.credentials" "save" $credentialPath `
        "--prompt" "Paste the evaluation Judge API Key (input is hidden): " `
        "--saved-label" "evaluation Judge API key" `
        "--show-fingerprint"
    if ($LASTEXITCODE -ne 0) { throw "The Kimi Judge API key could not be saved." }
    return
}

$arguments = @("-m", "evals.run_eval")
if ($RetrievalOnly) { $arguments += "--retrieval-only" }
if ($SkipJudge) { $arguments += "--skip-judge" }
if ($Limit -gt 0) { $arguments += @("--limit", [string]$Limit) }
if ($InputReport) { $arguments += @("--input-report", $InputReport) }
if ($CheckJudge) { $arguments += "--check-judge" }
if ($ResumeScores) { $arguments += "--resume-scores" }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Knowledge Garden evaluation failed." }
