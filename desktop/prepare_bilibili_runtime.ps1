$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $repoRoot 'project_storage.ps1') -ProjectRoot $repoRoot
Set-StrictMode -Version Latest

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$vendorRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot "vendor"))
$runtimeRoot = [IO.Path]::GetFullPath((Join-Path $vendorRoot "bilibili-runtime"))
$nodeRoot = [IO.Path]::GetFullPath((Join-Path $vendorRoot "node"))
$desktopResourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "src-tauri\resources\bilibili"))

if (-not $runtimeRoot.StartsWith($vendorRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to prepare a runtime outside the project vendor directory."
}

$pnpm = Get-Command pnpm -ErrorAction Stop
$node = Get-Command node -ErrorAction Stop
if (Test-Path -LiteralPath $runtimeRoot) {
    Remove-Item -LiteralPath $runtimeRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
New-Item -ItemType Directory -Path $nodeRoot -Force | Out-Null

& $pnpm.Source --dir $runtimeRoot add --prod --save-exact --ignore-scripts "@xzxzzx/bilibili-mcp@1.13.1"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to prepare the pinned Bilibili MCP runtime."
}

Copy-Item -LiteralPath $node.Source -Destination (Join-Path $nodeRoot "node.exe") -Force
$nodeVersion = (& $node.Source --version).Trim()
$nodeLicense = Get-ChildItem -LiteralPath (Split-Path $node.Source) -File -Filter "LICENSE*" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($nodeLicense) {
    Copy-Item -LiteralPath $nodeLicense.FullName -Destination (Join-Path $nodeRoot "LICENSE") -Force
} else {
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nodejs/node/$nodeVersion/LICENSE" -OutFile (Join-Path $nodeRoot "LICENSE")
}
$entry = Join-Path $runtimeRoot "node_modules\@xzxzzx\bilibili-mcp\dist\index.js"
$license = Join-Path $runtimeRoot "node_modules\@xzxzzx\bilibili-mcp\LICENSE"
if (-not (Test-Path -LiteralPath $entry) -or -not (Test-Path -LiteralPath $license)) {
    throw "Bilibili MCP runtime or its GPL license is incomplete."
}
if (Test-Path -LiteralPath $desktopResourceRoot) {
    Remove-Item -LiteralPath $desktopResourceRoot -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $desktopResourceRoot "node"), (Join-Path $desktopResourceRoot "runtime\dist") -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $nodeRoot "node.exe") -Destination (Join-Path $desktopResourceRoot "node\node.exe")
Copy-Item -LiteralPath (Join-Path $nodeRoot "LICENSE") -Destination (Join-Path $desktopResourceRoot "node\LICENSE")
$packageRoot = Join-Path $runtimeRoot "node_modules\@xzxzzx\bilibili-mcp"
$bundleScript = Join-Path $PSScriptRoot "scripts\bundle_bilibili.mjs"
& $node.Source $bundleScript (Join-Path $packageRoot "dist\index.js") (Join-Path $desktopResourceRoot "runtime\dist\index.js")
if ($LASTEXITCODE -ne 0) { throw "Failed to bundle the Bilibili MCP server." }
& $node.Source $bundleScript (Join-Path $packageRoot "dist\cli.js") (Join-Path $desktopResourceRoot "runtime\dist\cli.js")
if ($LASTEXITCODE -ne 0) { throw "Failed to bundle the Bilibili MCP CLI." }
Copy-Item -LiteralPath (Join-Path $packageRoot "LICENSE") -Destination (Join-Path $desktopResourceRoot "runtime\LICENSE")
Copy-Item -LiteralPath (Join-Path $packageRoot "README.md") -Destination (Join-Path $desktopResourceRoot "runtime\README.md")
$runtimePackageJson = @{
    name = "@xzxzzx/bilibili-mcp"
    version = "1.13.1"
    type = "commonjs"
    license = "GPL-3.0"
    source = "https://github.com/XZXZZX-Ai/bilibili-mcp"
} | ConvertTo-Json
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText(
    (Join-Path $desktopResourceRoot "runtime\package.json"),
    $runtimePackageJson,
    $utf8WithoutBom
)
Write-Host "Prepared bundled Bilibili runtime 1.13.1 with Node $nodeVersion."
