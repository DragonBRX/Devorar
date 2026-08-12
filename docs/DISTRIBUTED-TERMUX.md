# Devorar distribuído: Windows PowerShell + Termux

## Objetivo

O PC Windows funciona como coordenador persistente. Os celulares Android com Termux funcionam como workers. Cada worker recebe um job, lê somente os intervalos necessários do checkpoint remoto via HTTP Range, calcula localmente e devolve o resultado ao PC.

O cluster atual não baixa o DeepSeek inteiro em cada aparelho e não apresenta a fatia executada como uma inferência completa do modelo.

## Inicialização do PC

### Bloco automático

No PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; irm https://raw.githubusercontent.com/DragonBRX/Devorar/main/windows_install.ps1 | iex
```

O script instala Python pelo `winget` quando necessário, baixa o repositório, cria a regra de Firewall para a porta 8765 limitada a `LocalSubnet` e executa `windows_start.ps1`.

A criação da regra pode gerar uma solicitação UAC do Windows. O script não tenta contornar essa proteção.

### Execução depois da instalação

```powershell
cd $HOME\Devorar
.\windows_start.ps1
```

Parâmetros:

```powershell
.\windows_start.ps1 -Port 8765 -TermuxProcesses 2 -AdvertiseHost 192.168.1.10
```

`AdvertiseHost` só é necessário quando a detecção automática escolher um IP que os celulares não conseguem alcançar.

## Instalação dos celulares

O coordenador imprime um bloco completo com:

- endereço HTTP do PC;
- porta;
- token do cluster;
- quantidade de processos por aparelho.

Cole esse bloco diretamente em um Termux recém-instalado. Ele executa `pkg update`, instala Python/Git/tmux, baixa o Devorar, configura o aparelho e inicia o worker.

O worker fica em segundo plano dentro de uma sessão tmux e continua em polling depois de concluir os jobs disponíveis.

## Identidade de dispositivo

Na primeira instalação o Termux gera `DEVORAR_DEVICE_ID`, salvo em:

```text
~/.config/devorar/worker.env
```

Todos os processos do mesmo celular usam esse ID. Assim o painel do PC pode agrupar vários slots/processos como um único aparelho físico.

O nome padrão tenta usar fabricante e modelo Android e acrescenta os primeiros caracteres do ID persistente.

## Telemetria de hardware

A telemetria é coletada sem `psutil`:

- Windows: `GlobalMemoryStatusEx` para RAM;
- Linux/Android: `/proc/meminfo`;
- fallback Unix: `os.sysconf`;
- CPU: `platform`, variáveis do Windows e `/proc/cpuinfo`;
- Android: `getprop` para fabricante, modelo e versão quando disponível.

Cada registro/heartbeat pode conter:

```text
device_id
hostname
device_label
system
release
machine
python
cpu_model
cpu_logical_cores
ram_total_bytes
ram_available_bytes
ram_used_bytes
ram_usage_percent
android_manufacturer
android_model
android_version
slot
pid
```

O heartbeat atualiza a RAM disponível enquanto o worker está funcionando.

## Painel do PC

O servidor imprime o hardware do PC na inicialização e, por padrão, atualiza o painel a cada 15 segundos.

```powershell
python windows_server.py --status-seconds 15
```

O painel agrega os registros por `device_id`, evitando contar cada processo como um telefone diferente.

`--status-seconds 0` desliga apenas a impressão periódica.

## Informações no próprio celular

Depois da instalação:

```bash
devorar-worker status
```

mostra se o worker está ativo e imprime CPU/núcleos/RAM do aparelho.

Também existe:

```bash
devorar-worker hardware
```

O log do worker mostra a mesma telemetria local na conexão:

```bash
devorar-worker logs
```

## Protocolo

Os requests autenticados usam HMAC-SHA256 sobre método, caminho, timestamp, nonce e SHA-256 do corpo. O token não precisa trafegar como header em texto puro em cada request.

Fluxo principal:

```text
POST /v1/register
POST /v1/heartbeat
POST /v1/claim
POST /v1/result
GET  /v1/status
```

`/v1/heartbeat` também atualiza a telemetria do worker.

Jobs usam lease. Se um aparelho cair, o job pode voltar para a fila depois do vencimento, até o limite de tentativas.

## Porta e rede

A configuração padrão usa TCP 8765 e bind `0.0.0.0` no PC. O `windows_start.ps1` cria uma regra de entrada do Windows Firewall para `LocalSubnet`.

PC e celulares devem conseguir alcançar um ao outro na rede local. Algumas redes Wi-Fi de convidados usam isolamento entre clientes; nesse caso os celulares não conseguirão conectar mesmo com a porta aberta.

Não faça port-forward da porta do Devorar para a Internet pública.

## Estado científico

O worker atual executa `remote_bf16_head_teacher` sobre linhas completas BF16 de `head.weight` e ativações sintéticas determinísticas. Isso produz sinais numéricos verificáveis vindos de pesos reais da revisão fixada, mas ainda não produz a resposta linguística completa do DeepSeek.

Para chegar a um professor completo por pesos remotos ainda seriam necessários, entre outros componentes:

- embeddings;
- normalizações;
- atenção;
- KV cache;
- roteadores MoE;
- experts selecionados;
- formatos FP4/FP8 do corpo;
- execução camada a camada;
- validação contra um runtime de referência para os mesmos tokens.

## Exportação e treino

No PC:

```powershell
python distributed_export.py
python distributed_train_head.py --rank 8 --epochs 1000
```

O primeiro `student-head` é um surrogate experimental, não um LLM completo.
