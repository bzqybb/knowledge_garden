param(
    [string]$ArchivePath = $env:TRACEMEMO_BUNDLE_PATH,
    [string]$DownloadUrl = $env:TRACEMEMO_BUNDLE_URL,
    [string]$ExpectedSha256 = $env:TRACEMEMO_SHA256
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $repoRoot 'project_storage.ps1') -ProjectRoot $repoRoot

if ($env:TRACEMEMO_REDISTRIBUTION_APPROVED -ne "true") {
    throw "TraceMemo redistribution is not approved. Set TRACEMEMO_REDISTRIBUTION_APPROVED=true only after written permission or a documented non-commercial redistribution review."
}
if ([string]::IsNullOrWhiteSpace($ExpectedSha256)) {
    throw "TRACEMEMO_SHA256 is required; unverified third-party binaries will not be bundled."
}
if ([string]::IsNullOrWhiteSpace($ArchivePath) -and [string]::IsNullOrWhiteSpace($DownloadUrl)) {
    throw "Provide TRACEMEMO_BUNDLE_PATH or TRACEMEMO_BUNDLE_URL for an approved portable archive."
}

$resourcesRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "src-tauri\resources"))
$componentRoot = [IO.Path]::GetFullPath((Join-Path $resourcesRoot "tracememo"))
if (-not $componentRoot.StartsWith($resourcesRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to prepare a component outside desktop resources."
}

$temporaryRoot = Join-Path $env:GARDEN_TEMP_DIR ("knowledge-garden-tracememo-" + [Guid]::NewGuid().ToString("N"))
$payload = Join-Path $temporaryRoot "tracememo-component"
$expanded = Join-Path $temporaryRoot "expanded"
New-Item -ItemType Directory -Path $temporaryRoot, $expanded -Force | Out-Null
try {
    if (-not [string]::IsNullOrWhiteSpace($ArchivePath)) {
        $sourceArchive = [IO.Path]::GetFullPath($ArchivePath)
        if (-not (Test-Path -LiteralPath $sourceArchive -PathType Leaf)) {
            throw "TraceMemo archive not found: $sourceArchive"
        }
        Copy-Item -LiteralPath $sourceArchive -Destination $payload
        $payloadKind = [IO.Path]::GetExtension($sourceArchive).ToLowerInvariant()
    } else {
        if (-not $DownloadUrl.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase)) {
            throw "TraceMemo downloads must use HTTPS."
        }
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $payload
        $payloadKind = [IO.Path]::GetExtension(([Uri]$DownloadUrl).AbsolutePath).ToLowerInvariant()
    }

    $actualSha256 = (Get-FileHash -LiteralPath $payload -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $ExpectedSha256.Trim().ToLowerInvariant()) {
        throw "TraceMemo SHA-256 mismatch. Expected $ExpectedSha256 but received $actualSha256."
    }
    if (Test-Path -LiteralPath $componentRoot) {
        Remove-Item -LiteralPath $componentRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $componentRoot -Force | Out-Null
    if ($payloadKind -eq ".zip") {
        Expand-Archive -LiteralPath $payload -DestinationPath $expanded -Force
        $executable = Get-ChildItem -LiteralPath $expanded -Recurse -File -Filter "TraceMemo.exe" | Select-Object -First 1
        if (-not $executable) {
            throw "The approved portable archive does not contain TraceMemo.exe."
        }
        Get-ChildItem -LiteralPath $executable.Directory.FullName -Force | Copy-Item -Destination $componentRoot -Recurse -Force
    } elseif ($payloadKind -eq ".exe") {
        Copy-Item -LiteralPath $payload -Destination (Join-Path $componentRoot "TraceMemo-setup.exe") -Force
    } else {
        throw "TraceMemo component must be an official .exe installer or an approved portable .zip archive."
    }
    @"
TraceMemo is a third-party component by Wxw-Gu and its contributors.
Source: https://github.com/Wxw-Gu/TraceMemo
Bundled archive SHA-256: $actualSha256
Redistribution basis: non-commercial competition, testing, and demonstration use.
The project owner explicitly confirmed this release scope before the build.
See the source project and the release compliance record for applicable terms.
"@ | Set-Content -LiteralPath (Join-Path $componentRoot "THIRD_PARTY_NOTICE.txt") -Encoding utf8
    Write-Host "Prepared hash-verified bundled TraceMemo component."
} finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
