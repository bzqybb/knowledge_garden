param(
    [string]$PythonLauncher = "",
    [switch]$Rebuild,
    [switch]$WithVector,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $projectRoot 'project_storage.ps1') -ProjectRoot $projectRoot
$venvDir = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
Set-Location -LiteralPath $projectRoot

function Resolve-PythonLauncher {
    param([string]$ExplicitLauncher)

    $candidates = [System.Collections.Generic.List[object]]::new()
    if ($ExplicitLauncher) {
        $candidates.Add([pscustomobject]@{
            File = $ExplicitLauncher
            Args = @()
            Label = "explicit -PythonLauncher"
        })
    }

    $pyCommand = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyCommand) {
        $candidates.Add([pscustomobject]@{
            File = $pyCommand.Source
            Args = @("-3.13")
            Label = "Windows py launcher (Python 3.13)"
        })
        $candidates.Add([pscustomobject]@{
            File = $pyCommand.Source
            Args = @("-3")
            Label = "Windows py launcher (latest Python 3)"
        })
    }

    foreach ($commandName in @("python3.exe", "python.exe")) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($command) {
            $candidates.Add([pscustomobject]@{
                File = $command.Source
                Args = @()
                Label = $commandName
            })
        }
    }

    $localPrograms = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path -LiteralPath $localPrograms) {
        Get-ChildItem -LiteralPath $localPrograms -Directory -Filter "Python3*" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object {
                $executable = Join-Path $_.FullName "python.exe"
                if (Test-Path -LiteralPath $executable) {
                    $candidates.Add([pscustomobject]@{
                        File = $executable
                        Args = @()
                        Label = "local installation $($_.Name)"
                    })
                }
            }
    }

    foreach ($candidate in $candidates) {
        try {
            $candidateArgs = @($candidate.Args)
            $version = & $candidate.File @candidateArgs --version 2>&1
            if ($LASTEXITCODE -eq 0 -and "$version" -match "Python 3\.(1[0-9]|[2-9][0-9])") {
                Write-Host "Using $($candidate.Label): $version"
                return $candidate
            }
        } catch {
            # Continue probing; the final error lists the supported remedies.
        }
    }

    throw @"
No usable Python 3.10+ interpreter was found.
Install Python 3.13 from https://www.python.org/downloads/windows/ and enable the Python Launcher,
then run bootstrap.cmd again. If Python is already installed in a custom location, run:
  .\bootstrap.cmd -PythonLauncher "C:\full\path\to\python.exe"
"@
}

if ($Rebuild -and (Test-Path -LiteralPath $venvDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backup = Join-Path $projectRoot ".venv.broken-$stamp"
    Move-Item -LiteralPath $venvDir -Destination $backup
    Write-Host "Previous virtual environment preserved at $backup"
}

$venvHealthy = $false
if (Test-Path -LiteralPath $venvPython) {
    try {
        & $venvPython --version | Out-Host
        $venvHealthy = ($LASTEXITCODE -eq 0)
    } catch {
        $venvHealthy = $false
    }
}

if (-not $venvHealthy) {
    if (Test-Path -LiteralPath $venvDir) {
        throw "Existing .venv cannot start. Run bootstrap.cmd -Rebuild; the old environment will be preserved."
    }
    $launcher = Resolve-PythonLauncher -ExplicitLauncher $PythonLauncher
    $launcherArgs = @($launcher.Args)
    & $launcher.File @launcherArgs -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv" }
}

& $venvPython -m pip install -r requirements-eval.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install core/evaluation dependencies" }
if ($WithVector) {
    & $venvPython -m pip install -r requirements-vector.txt
    if ($LASTEXITCODE -ne 0) { throw "Failed to install vector dependencies" }
}

& $venvPython -c "import langgraph, sympy; print('smoke imports: langgraph=ok sympy=' + sympy.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Critical dependency import smoke test failed" }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "pip check failed" }

if (-not $SkipTests) {
    $testLog = Join-Path $projectRoot "data\runtime\bootstrap-tests.log"
    $testStdoutLog = Join-Path $projectRoot "data\runtime\bootstrap-tests.stdout.log"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $testLog) | Out-Null
    $testProcess = Start-Process -FilePath $venvPython `
        -ArgumentList @("-m", "unittest", "discover", "-s", "tests", "-v") `
        -WorkingDirectory $projectRoot `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $testStdoutLog `
        -RedirectStandardError $testLog
    if (Test-Path -LiteralPath $testLog) {
        Get-Content -LiteralPath $testLog
    }
    if (Test-Path -LiteralPath $testStdoutLog) {
        Get-Content -LiteralPath $testStdoutLog
    }
    if ($testProcess.ExitCode -ne 0) {
        throw "Full test suite failed (exit $($testProcess.ExitCode)); see $testLog"
    }
}

Write-Host "Knowledge Garden environment verification completed."
