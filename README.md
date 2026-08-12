# Devorar

Devorar é um protótipo de pesquisa do DragonBRX para experimentos com assimilação paramétrica e computação distribuída sobre partes de checkpoints remotos. A configuração principal atual é:

```text
DeepSeek no Hugging Face
        │
        ├── HTTP Range ──> Android / Termux 1 ─┐
        ├── HTTP Range ──> Android / Termux 2 ─┤
        ├── HTTP Range ──> Android / Termux N ─┤
        │                                       │
        └───────────────────────────────────────┘
                                                ▼
                                      PC Windows / PowerShell
                                      coordenador + resultados
```

O checkpoint completo do DeepSeek não precisa ser baixado em cada celular. Os workers recebem jobs do PC, buscam apenas intervalos delimitados dos pesos remotos, fazem a operação local e devolvem os resultados ao coordenador.

> Estado científico: o estágio distribuído atual executa fatias BF16 de `head.weight` com ativações sintéticas. Isso já é computação real sobre pesos remotos, mas ainda não é o forward completo do DeepSeek nem uma destilação completa de linguagem. O transformer inteiro, o roteamento MoE e os formatos quantizados do corpo ainda exigem um runtime paginado próprio.

## PC Windows: PowerShell

### Instalação automática em um bloco

Abra o **PowerShell** e cole:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; irm https://raw.githubusercontent.com/DragonBRX/Devorar/main/windows_install.ps1 | iex
```

O instalador:

- verifica se Python está disponível;
- se necessário, instala Python 3.12 pelo `winget`;
- baixa/atualiza `DragonBRX/Devorar` em `~/Devorar`;
- prepara a porta TCP `8765` para a rede local;
- inicia o coordenador em `0.0.0.0:8765`;
- detecta CPU, arquitetura, núcleos lógicos, RAM total e RAM disponível do PC;
- imprime o bloco único que deve ser colado em cada Termux.

Para liberar a porta no Firewall, o Windows pode mostrar o UAC. Basta confirmar a solicitação de administrador. A regra criada aceita conexões TCP na porta do Devorar somente a partir da sub-rede local.

### Iniciar novamente depois de instalado

Dentro de `~/Devorar`:

```powershell
.\windows_start.ps1
```

Porta diferente:

```powershell
.\windows_start.ps1 -Port 9000
```

Dois processos por celular no bloco Termux gerado:

```powershell
.\windows_start.ps1 -TermuxProcesses 2
```

Se o IP automático não for o IP Wi-Fi correto do PC:

```powershell
.\windows_start.ps1 -AdvertiseHost 192.168.1.10
```

O PowerShell mantém o servidor ativo enquanto a janela permanecer aberta.

## Termux: instalação automática desde zero

Não é necessário instalar Python, Git ou tmux manualmente antes.

Quando o servidor do PC iniciar, ele imprime:

```text
=== TERMUX: COLE ESTE BLOCO INTEIRO EM CADA CELULAR ===
...
=== FIM DO BLOCO TERMUX ===
```

Copie **o bloco inteiro mostrado pelo PC** e cole no Termux recém-instalado. O bloco já contém o IP do PC, a porta, o token do cluster e a quantidade de processos.

O instalador do celular:

- atualiza os pacotes do Termux;
- instala `python`, `git` e `tmux`;
- clona ou atualiza `~/Devorar`;
- cria um ID persistente para identificar o aparelho;
- detecta fabricante/modelo Android quando disponível;
- detecta CPU, arquitetura, núcleos e RAM;
- salva a configuração privada em `~/.config/devorar/worker.env`;
- inicia o worker em uma sessão `tmux` em segundo plano;
- reconecta automaticamente se o PC ou a rede ficarem temporariamente indisponíveis;
- continua esperando novos jobs depois de terminar o lote atual.

Depois da instalação:

```bash
devorar-worker status
devorar-worker hardware
devorar-worker logs
devorar-worker restart
devorar-worker stop
devorar-worker start
```

`devorar-worker status` e `devorar-worker hardware` mostram no próprio celular a CPU, os núcleos, a RAM total e a RAM disponível.

## Painel automático de hardware

O coordenador atualiza o painel no PowerShell a cada 15 segundos por padrão. Ele agrupa os processos pelo aparelho físico e mostra algo deste tipo:

```text
=== DEVORAR HARDWARE / atualização automática ===
PC: NOTEBOOK | CPU: ... | núcleos lógicos: 8 | RAM: 8.00 GiB total / 4.21 GiB disponível
Jobs: fila=180 ativos=4 concluídos=72 falhos=0 | dispositivos=3
- realme RMX3830 [ONLINE] | CPU: ... | núcleos: 8 | RAM: 3.76 GiB total / 1.42 GiB disponível
- iphone-worker ...
- outro-android [ONLINE] | ...
=== FIM HARDWARE ===
```

Cada worker envia telemetria de RAM atualizada por heartbeat. No celular, ao conectar, o log também registra o hardware local.

Para mudar o intervalo do painel ao executar diretamente o Python:

```powershell
python windows_server.py --status-seconds 30
```

Use `--status-seconds 0` para desligar somente a impressão periódica; a telemetria e o endpoint de status continuam ativos.

## Execução direta sem os scripts PowerShell

No Windows, Linux ou outro PC com Python:

```powershell
python windows_server.py --host 0.0.0.0 --port 8765
```

O servidor cria:

```text
cluster-state/
├── cluster-token.txt
├── cluster.sqlite3
└── plan.json
```

`cluster-state/` é ignorado pelo Git para não publicar token, banco e resultados locais.

## Resultados e primeiro treino experimental

No PC:

```powershell
python distributed_export.py
```

Para treinar o primeiro surrogate low-rank da cabeça remota:

```powershell
python distributed_train_head.py --rank 8 --epochs 1000
```

O treino do `student-head` precisa das dependências de treinamento no PC. Os workers Termux usados para buscar/calcular as fatias remotas utilizam somente a biblioteca padrão do Python e o código do próprio projeto.

## Ferramentas Frontier locais

Todas estas ferramentas são executadas diretamente no dispositivo, sem sintaxe de notebook:

```powershell
python frontier_scan.py
python frontier_tensor_probe.py
python frontier_head_parity.py
```

O scanner e as sondas trabalham com HTTP Range para evitar materializar um shard inteiro quando a operação só precisa de uma região específica.

Na revisão fixada atualmente, o mapa remoto do `DeepSeek-V4-Flash` possui dezenas de milhares de tensores distribuídos em dezenas de shards e aproximadamente 148 GiB de payload. O projeto não interpreta um peso isolado como texto ou pensamento: um parâmetro só ganha efeito dentro do restante do grafo.

## Protótipo de assimilação homóloga

A parte original do projeto também contém o experimento com modelos de mesma anatomia. A ideia aplicada aos tensores flutuantes compatíveis é:

```text
delta      = doador - base
delta_DARE = drop_aleatório(delta) / probabilidade_de_manter
candidato  = base + força_de_assimilação * delta_DARE
```

Isso é próximo de **Task Arithmetic + DARE**. Não é leitura de pensamentos dos pesos e não converte arbitrariamente qualquer arquitetura em qualquer outra.

## Segurança e integridade

- O token do cluster autentica os workers por HMAC-SHA256.
- Nonces e janela de tempo reduzem replay de requisições.
- Jobs possuem lease e voltam para a fila quando um worker cai.
- O Firewall do Windows é configurado para a sub-rede local, não para exposição pública intencional.
- Não exponha a porta do coordenador diretamente à Internet.
- Assimilar ou transformar pesos pode também carregar vieses, falhas ou backdoors dos modelos de origem.
- Preserve licenças, revisões e atribuições dos checkpoints utilizados.

## Testes locais

```powershell
python -m pip install -e ".[test]"
python -m pytest
```

Os testes locais usam fixtures pequenas e não precisam baixar o checkpoint completo do DeepSeek.

Veja também `docs/DISTRIBUTED-TERMUX.md` para a arquitetura distribuída e os detalhes do protocolo.
