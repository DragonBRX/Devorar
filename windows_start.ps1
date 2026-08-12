param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [ValidateRange(1, 65535)]
    [int]$DiscoveryPort = 8764,
    [ValidateRange(1, 16)]
    [int]$TermuxProcesses = 1
)

$ErrorActionPreference = "Stop"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Enable-DevorarFirewall {
    param(
        [int]$TcpPort,
        [int]$UdpPort
    )

    $tcpRule = "Devorar Coordinator TCP $TcpPort"
    $udpRule = "Devorar Discovery UDP $UdpPort"

    if (Test-Administrator) {
        if (-not (Get-NetFirewallRule -DisplayName $tcpRule -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName $tcpRule -Direction Inbound -Action Allow -Protocol TCP -LocalPort $TcpPort -Profile Any -RemoteAddress LocalSubnet | Out-Null
        }
        if (-not (Get-NetFirewallRule -DisplayName $udpRule -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName $udpRule -Direction Inbound -Action Allow -Protocol UDP -LocalPort $UdpPort -Profile Any -RemoteAddress LocalSubnet | Out-Null
        }
        Write-Host "Firewall: conexao TCP e descoberta UDP liberadas somente para a rede local."
        return
    }

    $escapedTcpRule = $tcpRule.Replace("'", "''")
    $escapedUdpRule = $udpRule.Replace("'", "''")
    $command = "`$tcp = Get-NetFirewallRule -DisplayName '$escapedTcpRule' -ErrorAction SilentlyContinue; if (-not `$tcp) { New-NetFirewallRule -DisplayName '$escapedTcpRule' -Direction Inbound -Action Allow -Protocol TCP -LocalPort $TcpPort -Profile Any -RemoteAddress LocalSubnet | Out-Null }; `$udp = Get-NetFirewallRule -DisplayName '$escapedUdpRule' -ErrorAction SilentlyContinue; if (-not `$udp) { New-NetFirewallRule -DisplayName '$escapedUdpRule' -Direction Inbound -Action Allow -Protocol UDP -LocalPort $UdpPort -Profile Any -RemoteAddress LocalSubnet | Out-Null }"
    Write-Host "O Windows vai pedir permissao de administrador para habilitar a descoberta local do Devorar."
    $process = Start-Process -FilePath "powershell.exe" -Verb RunAs -Wait -PassThru -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", $command
    )
    if ($process.ExitCode -ne 0) {
        throw "Nao foi possivel configurar o Firewall do Windows para o Devorar."
    }
    Write-Host "Firewall: conexao TCP e descoberta UDP liberadas somente para a rede local."
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
Enable-DevorarFirewall -TcpPort $Port -UdpPort $DiscoveryPort

$serverArgs = @(
    "windows_server.py",
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--discovery-port", "$DiscoveryPort",
    "--termux-processes", "$TermuxProcesses"
)

Write-Host "Iniciando Devorar no PowerShell com descoberta automatica na rede local..."
$exitCode = Invoke-PythonServer -ServerArguments $serverArgs
exit $exitCode
