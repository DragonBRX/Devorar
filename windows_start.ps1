param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [ValidateRange(1, 16)]
    [int]$TermuxProcesses = 1,
    [string]$AdvertiseHost = ""
)

$ErrorActionPreference = "Stop"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Enable-DevorarFirewall {
    param([int]$LocalPort)
    $ruleName = "Devorar Coordinator TCP $LocalPort"
    if (Test-Administrator) {
        $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
        if (-not $existing) {
            New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $LocalPort -Profile Any -RemoteAddress LocalSubnet | Out-Null
        }
        Write-Host "Firewall: porta TCP $LocalPort liberada para a rede local."
        return
    }

    $escapedRule = $ruleName.Replace("'", "''")
    $command = "`$rule = Get-NetFirewallRule -DisplayName '$escapedRule' -ErrorAction SilentlyContinue; if (-not `$rule) { New-NetFirewallRule -DisplayName '$escapedRule' -Direction Inbound -Action Allow -Protocol TCP -LocalPort $LocalPort -Profile Any -RemoteAddress LocalSubnet | Out-Null }"
    Write-Host "O Windows vai pedir permissao de administrador para liberar a porta $LocalPort no Firewall."
    $process = Start-Process -FilePath "powershell.exe" -Verb RunAs -Wait -PassThru -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", $command
    )
    if ($process.ExitCode -ne 0) {
        throw "Nao foi possivel liberar a porta $LocalPort no Firewall do Windows."
    }
    Write-Host "Firewall: porta TCP $LocalPort liberada para a rede local."
}

function Invoke-PythonServer {
    param([string[]]$ServerArguments)
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 @ServerArguments
        return $LASTEXITCODE
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python @ServerArguments
        return $LASTEXITCODE
    }
    throw "Python nao encontrado. Execute primeiro windows_install.ps1."
}

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo
Enable-DevorarFirewall -LocalPort $Port

$serverArgs = @(
    "windows_server.py",
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--termux-processes", "$TermuxProcesses"
)
if ($AdvertiseHost.Trim()) {
    $serverArgs += @("--advertise-host", $AdvertiseHost.Trim())
}

Write-Host "Iniciando Devorar no PowerShell..."
$exitCode = Invoke-PythonServer -ServerArguments $serverArgs
exit $exitCode
