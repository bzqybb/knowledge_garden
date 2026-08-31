$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$traceMemoPortable = Join-Path $PSScriptRoot "src-tauri\resources\tracememo\TraceMemo.exe"
$traceMemoInstaller = Join-Path $PSScriptRoot "src-tauri\resources\tracememo\TraceMemo-setup.exe"
if (-not (Test-Path -LiteralPath $traceMemoPortable -PathType Leaf) -and -not (Test-Path -LiteralPath $traceMemoInstaller -PathType Leaf)) {
    throw "Release would be incomplete. Missing bundled TraceMemo executable or installer."
}
$requiredFiles = @(
    (Join-Path $PSScriptRoot "src-tauri\resources\tracememo\THIRD_PARTY_NOTICE.txt"),
    (Join-Path $PSScriptRoot "src-tauri\resources\bilibili\node\node.exe"),
    (Join-Path $PSScriptRoot "src-tauri\resources\bilibili\node\LICENSE"),
    (Join-Path $PSScriptRoot "src-tauri\resources\bilibili\runtime\dist\index.js"),
    (Join-Path $PSScriptRoot "src-tauri\resources\bilibili\runtime\LICENSE")
)
$missing = @($requiredFiles | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missing.Count -gt 0) {
    throw "Release would be incomplete. Missing bundled files:`n$($missing -join "`n")"
}
Write-Host "Verified: the single installer contains TraceMemo, Node.js, Bilibili MCP, and notices."
