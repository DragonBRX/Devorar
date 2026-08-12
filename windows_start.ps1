param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [ValidateRange(1, 65535)]
    [int]$DiscoveryPort = 8764,
    [ValidateRange(1, 16)]
    [int]$TermuxProcesses = 1,
    [ValidateRange(5, 120)]
    [int]$StartupTimeoutSeconds = 20
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
        Write-Host "[OK] Firewall liberado somente para a rede local."
        return
    }

    $escapedTcpRule = $tcpRule.Replace("'", "''")
    $escapedUdpRule = $udpRule.Replace("'", "''")
    $command = "`$tcp = Get-NetFirewallRule -DisplayName '$escapedTcpRule' -ErrorAction SilentlyContinue; if (-not `$tcp) { New-NetFirewallRule -DisplayName '$escapedTcpRule' -Direction Inbound -Action Allow -Protocol TCP -LocalPort $TcpPort -Profile Any -RemoteAddress LocalSubnet | Out-Null }; `$udp = Get-NetFirewallRule -DisplayName '$escapedUdpRule' -ErrorAction SilentlyContinue; if (-not `$udp) { New-NetFirewallRule -DisplayName '$escapedUdpRule' -Direction Inbound -Action Allow -Protocol UDP -LocalPort $UdpPort -Profile Any -RemoteAddress LocalSubnet | Out-Null }"
    Write-Host "[1/4] O Windows vai pedir permissao de administrador para a rede local do Devorar."
    $process = Start-Process -FilePath "powershell.exe" -Verb RunAs -Wait -PassThru -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", $command
    )
    if ($process.ExitCode -ne 0) {
        throw "Nao foi possivel configurar o Firewall do Windows para o Devorar."
    }
    Write-Host "[OK] Firewall liberado somente para a rede local."
}

function Test-PythonCandidate {
    param(
        [string]$FilePath,
        [string[]]$Prefix
    )

    $probeId = [Guid]::NewGuid().ToString("N")
    $probeOut = Join-Path $env:TEMP "devorar-python-$probeId.out"
    $probeErr = Join-Path $env:TEMP "devorar-python-$probeId.err"
    $arguments = @()
    $arguments += $Prefix
    $arguments += @("-c", "__import__('sys').stdout.write(__import__('sys').executable)")

    try {
        $params = @{
            FilePath = $FilePath
            ArgumentList = $arguments
            NoNewWindow = $true
            RedirectStandardOutput = $probeOut
            RedirectStandardError = $probeErr
            PassThru = $true
        }
        $process = Start-Process @params
        if (-not $process.WaitForExit(5000)) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            return $false
        }
        if ($process.ExitCode -ne 0) {
            return $false
        }
        $result = ""
        if (Test-Path $probeOut) {
            $result = (Get-Content -Raw -Path $probeOut -ErrorAction SilentlyContinue).Trim()
        }
        return -not [string]::IsNullOrWhiteSpace($result)
    } catch {
        return $false
    } finally {
        Remove-Item $probeOut, $probeErr -Force -ErrorAction SilentlyContinue
    }
}

function Resolve-PythonCommand {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher -and $launcher.Source) {
        if (Test-PythonCandidate -FilePath $launcher.Source -Prefix @("-3")) {
            return @{
                FilePath = $launcher.Source
                Prefix = @("-3")
                Display = "$($launcher.Source) -3"
            }
        }
        Write-Host "[AVISO] O launcher 'py -3' existe, mas nao respondeu ao teste."
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python -and $python.Source) {
        if (Test-PythonCandidate -FilePath $python.Source -Prefix @()) {
            return @{
                FilePath = $python.Source
                Prefix = @()
                Display = $python.Source
            }
        }
        Write-Host "[AVISO] O comando 'python' existe, mas nao respondeu ao teste."
    }

    throw "Nenhum Python funcional respondeu em ate 5 segundos. Execute windows_install.ps1 novamente."
}

function Show-LogTail {
    param(
        [string]$StdoutPath,
        [string]$StderrPath,
        [int]$Lines = 80
    )

    if (Test-Path $StdoutPath) {
        $content = @(Get-Content -Path $StdoutPath -Tail $Lines -ErrorAction SilentlyContinue)
        if ($content.Count -gt 0) {
            Write-Host ""
            Write-Host "----- SAIDA DO SERVIDOR -----"
            $content | ForEach-Object { Write-Host $_ }
        }
    }

    if (Test-Path $StderrPath) {
        $content = @(Get-Content -Path $StderrPath -Tail $Lines -ErrorAction SilentlyContinue)
        if ($content.Count -gt 0) {
            Write-Host ""
            Write-Host "----- ERRO DO SERVIDOR -----"
            $content | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        }
    }
}

function Write-NewLogLines {
    param(
        [string]$Path,
        [ref]$Index,
        [switch]$ErrorStream
    )

    if (-not (Test-Path $Path)) {
        return
    }

    $lines = @(Get-Content -Path $Path -ErrorAction SilentlyContinue)
    if ($lines.Count -le $Index.Value) {
        return
    }

    for ($i = $Index.Value; $i -lt $lines.Count; $i++) {
        if ($ErrorStream) {
            Write-Host $lines[$i] -ForegroundColor Yellow
        } else {
            Write-Host $lines[$i]
        }
    }
    $Index.Value = $lines.Count
}

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

Write-Host ""
Write-Host "=============================================="
Write-Host "           DEVORAR - INICIALIZACAO"
Write-Host "=============================================="
Enable-DevorarFirewall -TcpPort $Port -UdpPort $DiscoveryPort

Write-Host "[2/4] Verificando Python..."
$pythonCommand = Resolve-PythonCommand
Write-Host "[OK] Python respondeu: $($pythonCommand.Display)"

$runtimeDir = Join-Path $repo "cluster-state\runtime"
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$stdoutPath = Join-Path $runtimeDir "windows-server.stdout.log"
$stderrPath = Join-Path $runtimeDir "windows-server.stderr.log"
$pidPath = Join-Path $runtimeDir "windows-server.pid"
Remove-Item $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

$serverArguments = @()
$serverArguments += $pythonCommand.Prefix
$serverArguments += @(
    "-u",
    "windows_server.py",
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--discovery-port", "$DiscoveryPort",
    "--termux-processes", "$TermuxProcesses"
)

Write-Host "[3/4] Iniciando o processo do servidor..."
$startParams = @{
    FilePath = $pythonCommand.FilePath
    ArgumentList = $serverArguments
    WorkingDirectory = $repo
    NoNewWindow = $true
    RedirectStandardOutput = $stdoutPath
    RedirectStandardError = $stderrPath
    PassThru = $true
}
$serverProcess = Start-Process @startParams

Set-Content -Path $pidPath -Value $serverProcess.Id -Encoding ascii
Write-Host "[OK] Processo iniciado. PID: $($serverProcess.Id)"
Write-Host "[4/4] Confirmando que o coordenador respondeu..."

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
$online = $false
while ((Get-Date) -lt $deadline) {
    $serverProcess.Refresh()
    if ($serverProcess.HasExited) {
        break
    }

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 1
        if ($response.StatusCode -eq 200) {
            $online = $true
            break
        }
    } catch {
    }

    Start-Sleep -Milliseconds 500
}

if (-not $online) {
    $serverProcess.Refresh()
    $exitedEarly = $serverProcess.HasExited
    if (-not $exitedEarly) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Red
    Write-Host "       DEVORAR - FALHA AO INICIAR" -ForegroundColor Red
    Write-Host "==============================================" -ForegroundColor Red
    if ($exitedEarly) {
        Write-Host "O processo Python encerrou antes do servidor ficar pronto."
    } else {
        Write-Host "O servidor nao respondeu em $StartupTimeoutSeconds segundos e foi encerrado."
    }
    Show-LogTail -StdoutPath $stdoutPath -StderrPath $stderrPath
    Write-Host ""
    Write-Host "Logs completos:"
    Write-Host "  $stdoutPath"
    Write-Host "  $stderrPath"
    throw "Falha na inicializacao do Devorar. O erro acima mostra onde parou."
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "                 DEVORAR ONLINE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Servidor do PC: PRONTO"
Write-Host "Descoberta automatica na Wi-Fi: ATIVA"
Write-Host "Estado: AGUARDANDO DISPOSITIVOS TERMUX"
Write-Host "Teste de saude: OK"
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

$outIndex = 0
$errIndex = 0
try {
    while (-not $serverProcess.HasExited) {
        Write-NewLogLines -Path $stdoutPath -Index ([ref]$outIndex)
        Write-NewLogLines -Path $stderrPath -Index ([ref]$errIndex) -ErrorStream
        Start-Sleep -Seconds 1
        $serverProcess.Refresh()
    }

    Write-NewLogLines -Path $stdoutPath -Index ([ref]$outIndex)
    Write-NewLogLines -Path $stderrPath -Index ([ref]$errIndex) -ErrorStream
    Write-Host ""
    Write-Host "Servidor Devorar encerrou com codigo $($serverProcess.ExitCode)."
} finally {
    Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
