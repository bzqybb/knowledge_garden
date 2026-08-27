param(
    [string]$Python = "python",
    [Alias("SaveKimiKey")]
    [switch]$SaveJudgeKey,
    [switch]$RetrievalOnly,
    [switch]$SkipJudge,
    [Alias("KimiBaseUrl")]
    [string]$JudgeBaseUrl = "",
    [string]$JudgeModel = "",
    [string]$Dataset = "",
    [string]$InputReport = "",
    [switch]$CheckJudge,
    [switch]$ResumeScores,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$storageRoot = Join-Path $projectRoot "data"
$env:GARDEN_TEMP_DIR = Join-Path $storageRoot "tmp"
$env:GARDEN_CACHE_DIR = Join-Path $storageRoot "cache"
$env:GARDEN_MODEL_CACHE_DIR = Join-Path $storageRoot "models"
foreach ($directory in @($env:GARDEN_TEMP_DIR, $env:GARDEN_CACHE_DIR, $env:GARDEN_MODEL_CACHE_DIR)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
$env:TEMP = $env:GARDEN_TEMP_DIR
$env:TMP = $env:GARDEN_TEMP_DIR
$env:TMPDIR = $env:GARDEN_TEMP_DIR
$env:XDG_CACHE_HOME = $env:GARDEN_CACHE_DIR
$env:HF_HOME = Join-Path $env:GARDEN_MODEL_CACHE_DIR "huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME "transformers"
$env:SENTENCE_TRANSFORMERS_HOME = Join-Path $env:GARDEN_MODEL_CACHE_DIR "sentence-transformers"
$env:TORCH_HOME = Join-Path $env:GARDEN_MODEL_CACHE_DIR "torch"
$env:PIP_CACHE_DIR = Join-Path $env:GARDEN_CACHE_DIR "pip"
if ($Python -eq "python") {
    $projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $projectPython) { $Python = $projectPython }
}

$env:RAGAS_DO_NOT_TRACK = "true"
$env:GARDEN_DISABLE_NETWORK = "1"
$env:GARDEN_UNDERSTANDING_PROVIDER = "primary"
if ($JudgeBaseUrl) { $env:JUDGE_BASE_URL = $JudgeBaseUrl }
if ($JudgeModel) { $env:JUDGE_MODEL = $JudgeModel }
$credentialPath = Join-Path $projectRoot "data\runtime\kimi-eval-api-key.dpapi"

if ($SaveJudgeKey) {
    & $Python "-m" "core.credentials" "save" $credentialPath `
        "--prompt" "Paste the evaluation Judge API Key (input is hidden): " `
        "--saved-label" "evaluation Judge API key" `
        "--show-fingerprint"
    if ($LASTEXITCODE -ne 0) { throw "The independent Judge API key could not be saved." }
    return
}

$arguments = @("-m", "evals.run_eval")
if ($RetrievalOnly) { $arguments += "--retrieval-only" }
if ($SkipJudge) { $arguments += "--skip-judge" }
if ($Limit -gt 0) { $arguments += @("--limit", [string]$Limit) }
if ($Dataset) { $arguments += @("--dataset", $Dataset) }
if ($InputReport) { $arguments += @("--input-report", $InputReport) }
if ($CheckJudge) { $arguments += "--check-judge" }
if ($ResumeScores) { $arguments += "--resume-scores" }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Knowledge Garden evaluation failed." }
