param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [ValidateRange(1, 16)]
    [int]$TermuxProcesses = 1
)

$ErrorActionPreference = "Stop"
$repoDir = Join-Path $HOME "Devorar"
$tempRoot = Join-Path $env:TEMP ("devorar-install-" + [Guid]::NewGuid().ToString("N"))
$zipPath = Join-Path $tempRoot "devorar.zip"
$extractDir = Join-Path $tempRoot "extract"

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    return $null
}

function Install-PythonIfNeeded {
    $python = Find-Python
    if ($python) {
        return
    }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Python nao esta instalado e o winget nao foi encontrado. Instale o App Installer do Windows e execute novamente."
    }
    Write-Host "Python nao encontrado. Instalando Python 3.12..."
    & winget install --exact --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao instalar Python pelo winget."
    }
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312"
    if (Test-Path $candidate) {
        $env:Path = "$candidate;$candidate\Scripts;$env:Path"
    }
    if (-not (Find-Python)) {
        throw "Python foi instalado, mas ainda nao foi localizado neste PowerShell. Feche e abra o PowerShell e execute o bloco novamente."
    }
}

try {
    Install-PythonIfNeeded
    New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $extractDir -Force | Out-Null
    Write-Host "Baixando a versao atual do DragonBRX/Devorar..."
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/DragonBRX/Devorar/archive/refs/heads/main.zip" -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
    $sourceDir = Join-Path $extractDir "Devorar-main"
    if (-not (Test-Path $sourceDir)) {
        throw "O pacote baixado nao contem Devorar-main."
    }
    New-Item -ItemType Directory -Path $repoDir -Force | Out-Null
    Get-ChildItem -Path $sourceDir -Force | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $repoDir -Recurse -Force
    }
    Write-Host "Projeto instalado/atualizado em $repoDir"
    & (Join-Path $repoDir "windows_start.ps1") -Port $Port -TermuxProcesses $TermuxProcesses
} finally {
    if (Test-Path $tempRoot) {
        Remove-Item -Path $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
