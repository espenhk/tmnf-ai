<#
.SYNOPSIS
    Setup and launch script for tmnf-ai on Windows.

.DESCRIPTION
    Two modes:
      Fresh machine  — installs Python 3.11+, Poetry, Git, TMInterface 1.4, and
                       Python dependencies, then launches everything.
      Existing setup — detects what is already present, skips install steps, and
                       goes straight to launching TMInterface + the requested command.

    Idempotent: safe to re-run at any time.

.PARAMETER Command
    The Python command to run after setup, e.g.:
        "python main.py my_experiment"
        "python grid_search.py config/gs_cmaes.yaml"

.PARAMETER DryRun
    Print what would be done without executing anything.

.EXAMPLE
    .\setup_and_run.ps1 "python main.py my_experiment"
    .\setup_and_run.ps1 "python grid_search.py config/gs_cmaes.yaml" -DryRun

.NOTES
    Prerequisites this script does NOT handle:
        - Trackmania Nations Forever must already be installed.

    TMInterface 1.4:
        Official installer: https://donadigo.com/tminterface
        Only version 1.4.x is compatible with the tminterface Python package
        used by this project.  Later versions changed the Python API.

    Source packages (not on PyPI):
        tminterface  https://github.com/donadigo/tminterface  (any recent commit on main)
        pygbx        https://github.com/donadigo/pygbx        (any recent commit on main)

    These are installed into the Poetry virtual-env with pip before
    `poetry install` resolves the remaining dependencies.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "",

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Step([string]$msg) {
    Write-Host "[SETUP] $msg" -ForegroundColor Cyan
}

function Write-Skip([string]$msg) {
    Write-Host "[SKIP]  $msg" -ForegroundColor DarkGray
}

function Write-Ok([string]$msg) {
    Write-Host "[OK]    $msg" -ForegroundColor Green
}

function Write-Err([string]$msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
}

function Invoke-Step([string]$description, [scriptblock]$action) {
    Write-Step $description
    if ($DryRun) {
        Write-Host "        (dry-run — skipped)" -ForegroundColor Yellow
        return
    }
    & $action
}

function Test-CommandExists([string]$name) {
    return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

function Assert-Success([string]$context) {
    if ($LASTEXITCODE -ne 0) {
        Write-Err "$context failed (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
}

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

$ScriptDir      = $PSScriptRoot
$TMIfaceDir     = Join-Path $env:USERPROFILE "TMInterface"
$TMIfaceExe     = Join-Path $TMIfaceDir "TMInterface.exe"
# Source repos — pinned to HEAD of the branches known to work with TMInterface 1.4
$TminterfaceUrl = "https://github.com/donadigo/tminterface/archive/refs/heads/master.zip"
$PygbxUrl       = "https://github.com/donadigo/pygbx/archive/refs/heads/master.zip"
# TMInterface 1.4 installer (official release page; update hash/URL when a new 1.4.x drops)
$TMIfaceInstallerUrl = "https://github.com/donadigo/tminterface/releases/download/1.4.3/TMInterface_1.4.3_Setup.exe"

# ---------------------------------------------------------------------------
# 1. Python 3.11+
# ---------------------------------------------------------------------------

function Install-Python {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) {
        $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver -and [version]$ver -ge [version]"3.11") {
            Write-Skip "Python $ver already in PATH."
            return
        }
    }

    Invoke-Step "Installing Python 3.11 via winget" {
        winget install --id Python.Python.3.11 --source winget --silent --accept-package-agreements --accept-source-agreements
        Assert-Success "Python install"
        # Refresh PATH for the rest of this process
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path","User")
    }
}

# ---------------------------------------------------------------------------
# 2. Git
# ---------------------------------------------------------------------------

function Install-Git {
    if (Test-CommandExists "git") {
        Write-Skip "Git already in PATH."
        return
    }

    Invoke-Step "Installing Git via winget" {
        winget install --id Git.Git --source winget --silent --accept-package-agreements --accept-source-agreements
        Assert-Success "Git install"
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path","User")
    }
}

# ---------------------------------------------------------------------------
# 3. Poetry
# ---------------------------------------------------------------------------

function Install-Poetry {
    if (Test-CommandExists "poetry") {
        Write-Skip "Poetry already in PATH."
        return
    }

    Invoke-Step "Installing Poetry" {
        $installer = Join-Path $env:TEMP "install-poetry.py"
        Invoke-WebRequest -Uri "https://install.python-poetry.org" -OutFile $installer -UseBasicParsing
        python $installer
        Assert-Success "Poetry install"
        # Add Poetry bin dir to PATH for this session
        $poetryBin = Join-Path $env:APPDATA "Python\Scripts"
        if (Test-Path $poetryBin) {
            $env:Path = "$poetryBin;$env:Path"
        }
    }
}

# ---------------------------------------------------------------------------
# 4. TMInterface 1.4
# ---------------------------------------------------------------------------

function Install-TMInterface {
    if (Test-Path $TMIfaceExe) {
        Write-Skip "TMInterface already found at $TMIfaceExe."
        return
    }

    Invoke-Step "Downloading TMInterface 1.4 installer" {
        $installer = Join-Path $env:TEMP "TMInterface_Setup.exe"
        Write-Step "  Fetching $TMIfaceInstallerUrl"
        Invoke-WebRequest -Uri $TMIfaceInstallerUrl -OutFile $installer -UseBasicParsing
        Assert-Success "TMInterface download"

        Write-Step "  Running TMInterface installer (silent)"
        # /SILENT — run without wizard UI; /DIR — install location
        Start-Process -FilePath $installer -ArgumentList "/SILENT", "/DIR=`"$TMIfaceDir`"" -Wait
        Assert-Success "TMInterface installation"

        if (-not (Test-Path $TMIfaceExe)) {
            Write-Err "TMInterface executable not found after install at $TMIfaceExe."
            Write-Err "Please install TMInterface 1.4 manually from: https://donadigo.com/tminterface"
            exit 1
        }
    }
}

# ---------------------------------------------------------------------------
# 5. Python source packages (tminterface, pygbx)
# ---------------------------------------------------------------------------

function Install-SourcePackage([string]$name, [string]$zipUrl) {
    # Check whether the package is already importable inside the Poetry venv
    $already = & poetry run python -c "import $name" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Skip "$name already importable in Poetry venv."
        return
    }

    Invoke-Step "Installing $name from source ($zipUrl)" {
        $zip  = Join-Path $env:TEMP "$name-master.zip"
        $dest = Join-Path $env:TEMP "$name-src"

        Invoke-WebRequest -Uri $zipUrl -OutFile $zip -UseBasicParsing
        Assert-Success "$name download"

        if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
        Expand-Archive -Path $zip -DestinationPath $dest -Force

        # The archive unpacks to a single subdirectory
        $srcDir = Get-ChildItem -Path $dest -Directory | Select-Object -First 1
        Push-Location $srcDir.FullName
        try {
            poetry run pip install .
            Assert-Success "$name pip install"
        } finally {
            Pop-Location
        }
    }
}

# ---------------------------------------------------------------------------
# 6. Poetry dependencies
# ---------------------------------------------------------------------------

function Install-PoetryDeps {
    Invoke-Step "Running 'poetry install' (skips if lock is satisfied)" {
        Push-Location $ScriptDir
        try {
            poetry install --with tmnf
            Assert-Success "poetry install"
        } finally {
            Pop-Location
        }
    }
}

# ---------------------------------------------------------------------------
# 7. Launch TMInterface
# ---------------------------------------------------------------------------

function Start-TMInterface {
    $procs = Get-Process -Name "TMInterface" -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Skip "TMInterface process already running (PID $($procs[0].Id))."
        return
    }

    if (-not (Test-Path $TMIfaceExe)) {
        Write-Err "Cannot launch TMInterface — executable not found at $TMIfaceExe."
        exit 1
    }

    Invoke-Step "Launching TMInterface" {
        Start-Process -FilePath $TMIfaceExe
        Write-Step "Waiting for TMInterface to initialise (up to 30 s)…"
        $deadline = (Get-Date).AddSeconds(30)
        $ready    = $false
        while ((Get-Date) -lt $deadline) {
            $proc = Get-Process -Name "TMInterface" -ErrorAction SilentlyContinue
            if ($proc) { $ready = $true; break }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ready) {
            Write-Err "TMInterface did not start within 30 seconds."
            exit 1
        }
        # Extra settle time for the plugin to register its TCP listener
        Start-Sleep -Seconds 3
        Write-Ok "TMInterface is running."
    }
}

# ---------------------------------------------------------------------------
# 8. Run user command
# ---------------------------------------------------------------------------

function Invoke-UserCommand {
    if (-not $Command) {
        Write-Step "No command supplied — setup complete.  Pass a command as the first argument to launch training."
        return
    }

    Write-Step "Running: $Command"
    if ($DryRun) {
        Write-Host "        (dry-run — skipped)" -ForegroundColor Yellow
        return
    }

    Push-Location $ScriptDir
    try {
        # Split the command string so that Invoke-Expression is not required
        $parts = $Command -split "\s+", 2
        $exe   = $parts[0]
        $args  = if ($parts.Length -gt 1) { $parts[1] } else { "" }

        if ($exe -eq "python") {
            # Run inside the Poetry venv
            if ($args) {
                poetry run python $args
            } else {
                poetry run python
            }
        } else {
            # Arbitrary command — forward as-is
            Invoke-Expression $Command
        }
        Assert-Success "User command"
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if ($DryRun) {
    Write-Host "=== DRY RUN — no changes will be made ===" -ForegroundColor Yellow
}

Install-Python
Install-Git
Install-Poetry
Install-TMInterface
# Source packages must come before full poetry install so the lock can resolve them
Install-SourcePackage "tminterface" $TminterfaceUrl
Install-SourcePackage "pygbx"        $PygbxUrl
Install-PoetryDeps
Start-TMInterface
Invoke-UserCommand

Write-Ok "Done."
