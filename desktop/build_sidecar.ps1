$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $DesktopRoot
. (Join-Path $RepoRoot 'project_storage.ps1') -ProjectRoot $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$BinaryDir = Join-Path $DesktopRoot "src-tauri\binaries"
New-Item -ItemType Directory -Force -Path $BinaryDir | Out-Null

& $Python -m pip install "pyinstaller>=6.10,<7"
& $Python -m PyInstaller --noconfirm --clean --distpath (Join-Path $DesktopRoot "sidecar-dist") (Join-Path $DesktopRoot "knowledge_garden_sidecar.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller sidecar build failed" }

if (Get-Command rustc -ErrorAction SilentlyContinue) {
    $TargetTriple = (& rustc --print host-tuple).Trim()
} elseif ($env:PROCESSOR_ARCHITECTURE -eq 'AMD64') {
    # The Python sidecar can be prepared before the Rust/Tauri toolchain is
    # installed. Windows release builds in this repository target x64 MSVC.
    $TargetTriple = 'x86_64-pc-windows-msvc'
} else {
    throw 'rustc is unavailable and the target triple cannot be inferred safely.'
}
if (-not $TargetTriple) { throw "Unable to determine Rust target triple" }
$Source = Join-Path $DesktopRoot "sidecar-dist\knowledge-garden-sidecar.exe"
$Target = Join-Path $BinaryDir "knowledge-garden-sidecar-$TargetTriple.exe"
Copy-Item -LiteralPath $Source -Destination $Target -Force
Write-Host "Sidecar ready: $Target"
