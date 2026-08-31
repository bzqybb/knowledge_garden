param([string]$ProjectRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path))

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$projectDrive = [IO.Path]::GetPathRoot($ProjectRoot)
if ($env:GARDEN_DATA_DIR) {
    $storageRoot = [IO.Path]::GetFullPath($env:GARDEN_DATA_DIR)
} elseif ($projectDrive -and $projectDrive.StartsWith('D:', [StringComparison]::OrdinalIgnoreCase)) {
    $storageRoot = Join-Path $ProjectRoot 'data'
} elseif (Test-Path -LiteralPath 'D:\') {
    $storageRoot = 'D:\KnowledgeGarden\data'
} else {
    $storageRoot = Join-Path $ProjectRoot 'data'
}

$tempRoot = Join-Path $storageRoot 'tmp'
$cacheRoot = Join-Path $storageRoot 'cache'
$modelRoot = Join-Path $storageRoot 'models'
$buildRoot = Join-Path $storageRoot 'build'
foreach ($directory in @($storageRoot, $tempRoot, $cacheRoot, $modelRoot, $buildRoot)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

# Process-local only: do not change the user's global Windows TEMP or package settings.
$env:GARDEN_DATA_DIR = $storageRoot
$env:GARDEN_TEMP_DIR = $tempRoot
$env:GARDEN_CACHE_DIR = $cacheRoot
$env:GARDEN_MODEL_CACHE_DIR = $modelRoot
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:TMPDIR = $tempRoot
$env:XDG_CACHE_HOME = $cacheRoot
$env:PIP_CACHE_DIR = Join-Path $cacheRoot 'pip'
$env:HF_HOME = Join-Path $modelRoot 'huggingface'
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME 'transformers'
$env:SENTENCE_TRANSFORMERS_HOME = Join-Path $modelRoot 'sentence-transformers'
$env:TORCH_HOME = Join-Path $modelRoot 'torch'
$env:MPLCONFIGDIR = Join-Path $cacheRoot 'matplotlib'
$env:NLTK_DATA = Join-Path $modelRoot 'nltk'
$env:JOBLIB_TEMP_FOLDER = Join-Path $tempRoot 'joblib'
$env:PYINSTALLER_CONFIG_DIR = Join-Path $cacheRoot 'pyinstaller'
$env:npm_config_cache = Join-Path $cacheRoot 'npm'
$env:PNPM_HOME = Join-Path $cacheRoot 'pnpm-home'
$env:PNPM_STORE_DIR = Join-Path $cacheRoot 'pnpm-store'
$env:CARGO_HOME = Join-Path $cacheRoot 'cargo-home'
$env:RUSTUP_HOME = Join-Path $cacheRoot 'rustup-home'
$env:CARGO_TARGET_DIR = Join-Path $buildRoot 'cargo-target'

foreach ($directory in @(
    $env:PIP_CACHE_DIR, $env:HF_HOME, $env:SENTENCE_TRANSFORMERS_HOME,
    $env:TORCH_HOME, $env:MPLCONFIGDIR, $env:NLTK_DATA,
    $env:JOBLIB_TEMP_FOLDER, $env:PYINSTALLER_CONFIG_DIR,
    $env:npm_config_cache, $env:PNPM_HOME, $env:PNPM_STORE_DIR,
    $env:CARGO_HOME, $env:RUSTUP_HOME, $env:CARGO_TARGET_DIR
)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

$script:GardenStorageRoot = $storageRoot

$cargoBin = Join-Path $env:CARGO_HOME 'bin'
if (Test-Path -LiteralPath $cargoBin) {
    $env:Path = $cargoBin + [IO.Path]::PathSeparator + $env:Path
}
