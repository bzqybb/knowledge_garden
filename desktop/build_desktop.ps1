param(
    [string]$Config = 'src-tauri/tauri.local.conf.json',
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$desktopRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $desktopRoot
. (Join-Path $projectRoot 'project_storage.ps1') -ProjectRoot $projectRoot

Push-Location -LiteralPath $desktopRoot
try {
    if (-not $SkipInstall) {
        & pnpm install --frozen-lockfile
        if ($LASTEXITCODE -ne 0) { throw 'pnpm install failed' }
    }

    & pnpm exec tauri build --config $Config
    if ($LASTEXITCODE -ne 0) { throw 'Tauri desktop build failed' }
} finally {
    Pop-Location
}
