param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [ValidateRange(1, 16)]
    [int]$TermuxProcesses = 1
)

$ErrorActionPreference = "Stop"
$repoDir = Join-Path $HOME "Devorar"

Set-Location $HOME

Write-Host "Parando instancias antigas do servidor Devorar, se existirem..."
try {
    Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and $_.CommandLine -and $_.CommandLine -match "windows_server\.py"
    } | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
} catch {
    Write-Host "Aviso: nao foi possivel inspecionar todos os processos antigos."
}

Start-Sleep -Milliseconds 500

if (Test-Path $repoDir) {
    Write-Host "Removendo instalacao antiga: $repoDir"
    Remove-Item -Path $repoDir -Recurse -Force
}

Write-Host "Baixando instalador limpo do DragonBRX/Devorar..."
$installerText = Invoke-RestMethod -Uri "https://raw.githubusercontent.com/DragonBRX/Devorar/main/windows_install.ps1"
$installer = [ScriptBlock]::Create([string]$installerText)
& $installer -Port $Port -TermuxProcesses $TermuxProcesses
