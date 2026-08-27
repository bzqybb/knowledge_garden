param(
    [switch]$WaitForCodexExit
)

$ErrorActionPreference = 'Stop'
$migrationRoot = 'D:\CodexData'
$logPath = Join-Path $migrationRoot 'migration.log'
$donePath = Join-Path $migrationRoot 'migration-complete.txt'
$profileRoot = [Environment]::GetFolderPath('UserProfile')
$codexSource = Join-Path $profileRoot '.codex'
$runtimeSource = Join-Path $profileRoot '.cache\codex-runtimes'

function Write-MigrationLog([string]$message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $message"
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function Get-CodexProcesses {
    @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -match '^codex($|[-_])'
    })
}

function Assert-ExactMigrationPath([string]$path, [string]$expected) {
    $resolved = [IO.Path]::GetFullPath($path).TrimEnd('\')
    $required = [IO.Path]::GetFullPath($expected).TrimEnd('\')
    if (-not [StringComparer]::OrdinalIgnoreCase.Equals($resolved, $required)) {
        throw "Migration path validation failed: $resolved"
    }
}

function Migrate-DirectoryToJunction([string]$source, [string]$target) {
    Assert-ExactMigrationPath $source $source
    Assert-ExactMigrationPath $target $target
    if (-not (Test-Path -LiteralPath $source)) {
        Write-MigrationLog "Skipped missing directory: $source"
        return
    }
    $sourceItem = Get-Item -LiteralPath $source -Force
    if ($sourceItem.LinkType -eq 'Junction') {
        $currentTarget = [IO.Path]::GetFullPath([string]$sourceItem.Target).TrimEnd('\')
        $expectedTarget = [IO.Path]::GetFullPath($target).TrimEnd('\')
        if ([StringComparer]::OrdinalIgnoreCase.Equals($currentTarget, $expectedTarget)) {
            Write-MigrationLog "Junction already points to D: $source -> $target"
            return
        }
        throw "Unexpected junction target: $source -> $currentTarget"
    }

    New-Item -ItemType Directory -Path $target -Force | Out-Null
    Write-MigrationLog "Starting verified copy: $source -> $target"
    & robocopy.exe $source $target /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 /XJ /NP /NFL /NDL /NJH /NJS | Out-Null
    $robocopyExit = $LASTEXITCODE
    if ($robocopyExit -gt 7) {
        throw "Robocopy failed with exit code $robocopyExit"
    }

    $missing = 0
    $mismatched = 0
    $sourceFiles = @(Get-ChildItem -LiteralPath $source -File -Recurse -Force -ErrorAction Stop)
    foreach ($file in $sourceFiles) {
        $relative = $file.FullName.Substring($source.Length).TrimStart('\')
        $copyPath = Join-Path $target $relative
        if (-not (Test-Path -LiteralPath $copyPath -PathType Leaf)) {
            $missing++
            continue
        }
        if ((Get-Item -LiteralPath $copyPath -Force).Length -ne $file.Length) {
            $mismatched++
        }
    }
    if ($missing -ne 0 -or $mismatched -ne 0) {
        throw "Copy verification failed: missing=$missing size_mismatch=$mismatched"
    }

    $backup = "$source.migration-backup-$(Get-Date -Format 'yyyyMMddHHmmss')"
    if (Test-Path -LiteralPath $backup) {
        throw "Backup path already exists: $backup"
    }
    Move-Item -LiteralPath $source -Destination $backup
    try {
        New-Item -ItemType Junction -Path $source -Target $target | Out-Null
        $junction = Get-Item -LiteralPath $source -Force
        if ($junction.LinkType -ne 'Junction') {
            throw "Failed to create junction: $source"
        }
        $junctionTarget = [IO.Path]::GetFullPath([string]$junction.Target).TrimEnd('\')
        $expectedTarget = [IO.Path]::GetFullPath($target).TrimEnd('\')
        if (-not [StringComparer]::OrdinalIgnoreCase.Equals($junctionTarget, $expectedTarget)) {
            throw "Junction target validation failed: $junctionTarget"
        }
        # The exact backup was created above inside the user's profile and has
        # already been byte-size verified against D. Removing it fulfills the
        # user's request to clear the old C-drive copy.
        Remove-Item -LiteralPath $backup -Recurse -Force
        Write-MigrationLog "Migration complete; removed verified C-drive copy: $source -> $target; files=$($sourceFiles.Count)"
    }
    catch {
        if (Test-Path -LiteralPath $source) {
            $failedJunction = Get-Item -LiteralPath $source -Force
            if ($failedJunction.LinkType -eq 'Junction') {
                Remove-Item -LiteralPath $source -Force
            }
        }
        if ((-not (Test-Path -LiteralPath $source)) -and (Test-Path -LiteralPath $backup)) {
            Move-Item -LiteralPath $backup -Destination $source
        }
        throw
    }
}

New-Item -ItemType Directory -Path $migrationRoot -Force | Out-Null
Remove-Item -LiteralPath $donePath -Force -ErrorAction SilentlyContinue
if ($WaitForCodexExit) {
    Write-MigrationLog 'Waiting for Codex to exit before migration.'
    while ((Get-CodexProcesses).Count -gt 0) {
        Start-Sleep -Seconds 3
    }
    Start-Sleep -Seconds 5
}
elseif ((Get-CodexProcesses).Count -gt 0) {
    throw 'Codex is still running. Close Codex before running this script.'
}

try {
    Assert-ExactMigrationPath $codexSource $codexSource
    Assert-ExactMigrationPath 'D:\CodexData\codex-home' 'D:\CodexData\codex-home'
    Assert-ExactMigrationPath $runtimeSource $runtimeSource
    Assert-ExactMigrationPath 'D:\CodexData\codex-runtimes' 'D:\CodexData\codex-runtimes'
    Migrate-DirectoryToJunction $codexSource 'D:\CodexData\codex-home'
    Migrate-DirectoryToJunction $runtimeSource 'D:\CodexData\codex-runtimes'
    $message = "Migration succeeded. Original C-drive paths are junctions to D.`r`nLog: $logPath"
    Set-Content -LiteralPath $donePath -Value $message -Encoding UTF8
    Write-MigrationLog 'All migration tasks completed.'
}
catch {
    Write-MigrationLog "Migration failed: $($_.Exception.Message)"
    throw
}
