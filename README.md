# Devorar

Devorar é um protótipo de pesquisa do DragonBRX para assimilação paramétrica e computação distribuída sobre partes de checkpoints remotos. A configuração principal atual é:

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

## PC Windows: instalação automática

### Instalar ou atualizar

Abra o PowerShell e cole somente este bloco:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; irm https://raw.githubusercontent.com/DragonBRX/Devorar/main/windows_install.ps1 | iex
```

O instalador verifica Python, instala Python 3.12 pelo `winget` quando necessário, baixa o projeto para `~/Devorar`, configura o Firewall para a rede local e inicia a porta TCP `8765`.

### Reinstalação limpa do PC

Use este bloco quando quiser apagar a instalação local do Devorar, baixar tudo novamente do GitHub e iniciar o servidor já configurado:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; irm https://raw.githubusercontent.com/DragonBRX/Devorar/main/windows_reinstall.ps1 | iex
```

A reinstalação:

- para instâncias antigas de `windows_server.py` quando encontradas;
- muda para a pasta do usuário antes da remoção;
- apaga `~/Devorar` completamente;
- baixa novamente o instalador atual do GitHub;
- reinstala o projeto;
- reutiliza/cria a regra do Firewall;
- abre a porta `8765` para `LocalSubnet`;
- inicia o coordenador automaticamente.

Para usar outra porta depois da instalação:

```powershell
cd $HOME\Devorar
.\windows_start.ps1 -Port 9000
```

Para fazer o bloco Termux usar dois processos por celular:

```powershell
cd $HOME\Devorar
.\windows_start.ps1 -TermuxProcesses 2
```

Se o IP automático não for o IP Wi-Fi correto do PC:

```powershell
cd $HOME\Devorar
.\windows_start.ps1 -AdvertiseHost 192.168.1.10
```

O Windows pode mostrar o UAC ao configurar o Firewall. A regra criada aceita conexões na porta do Devorar somente da sub-rede local.

## O bloco do Termux aparece imediatamente

O servidor não espera mais a preparação do plano remoto do DeepSeek para mostrar os comandos dos celulares. A ordem agora é:

```text
abre a porta
↓
mostra hardware do PC
↓
mostra bloco de instalação do Termux
↓
mostra bloco de reinstalação do Termux
↓
começa a preparar os jobs do DeepSeek em segundo plano
```

Assim, mesmo que o Hugging Face esteja lento ou temporariamente indisponível, os celulares já podem instalar, conectar e ficar aguardando trabalho.

Quando o PC inicia, aparecem dois blocos completos:

```text
=== TERMUX: INSTALAÇÃO DO ZERO / CONECTAR ===
...
=== FIM INSTALAÇÃO TERMUX ===

=== TERMUX: REINSTALAÇÃO LIMPA ===
...
=== FIM REINSTALAÇÃO TERMUX ===
```

Os blocos gerados já contêm automaticamente o IP do PC, a porta, o token do cluster e a quantidade de processos. Não é necessário editar nada.

## Termux: instalação do zero

Para um Termux recém-instalado, a opção recomendada é copiar o bloco `TERMUX: INSTALAÇÃO DO ZERO / CONECTAR` mostrado pelo PC. Ele já faz tudo e conecta o aparelho.

Se você quiser apenas preparar um Termux do zero antes de ter o PC disponível, pode usar este bloco genérico:

```bash
pkg update -y && pkg install -y python git tmux && \
rm -rf "$HOME/Devorar" && \
git clone --depth 1 https://github.com/DragonBRX/Devorar.git "$HOME/Devorar" && \
cd "$HOME/Devorar" && chmod +x termux_install.sh termux_device_install.sh && \
./termux_install.sh
```

Esse bloco instala a base. Para conectar ao PC, depois use o bloco completo que o coordenador imprime, pois somente o PC conhece o token atual do cluster.

O bloco completo gerado pelo PC:

- atualiza os pacotes do Termux;
- instala `python`, `git` e `tmux`;
- clona ou atualiza `~/Devorar`;
- recebe IP, porta e token do PC;
- cria ou preserva o ID persistente do aparelho;
- detecta fabricante/modelo Android quando disponível;
- detecta CPU, arquitetura, núcleos e RAM;
- salva a configuração privada em `~/.config/devorar/worker.env`;
- inicia o worker em `tmux`;
- reconecta automaticamente se o PC ou a rede caírem;
- continua aguardando novos jobs quando a fila estiver vazia.

## Termux: reinstalação limpa

O próprio PC também imprime um bloco `TERMUX: REINSTALAÇÃO LIMPA`. Esse bloco para o worker antigo, apaga `~/Devorar`, clona o repositório novamente e reconecta usando a configuração atual do PC.

Depois que um celular já estiver configurado, também existe o comando curto:

```bash
devorar-worker reinstall
```

Ele para o worker, apaga o clone local, clona `DragonBRX/Devorar` novamente e inicia o worker com o mesmo servidor, token, nome e ID físico do aparelho.

Outros comandos úteis:

```bash
devorar-worker status
devorar-worker hardware
devorar-worker logs
devorar-worker restart
devorar-worker stop
devorar-worker start
```

`devorar-worker status` e `devorar-worker hardware` mostram CPU, núcleos, RAM total e RAM disponível no próprio celular.

## Painel automático de hardware

O coordenador atualiza o painel no PowerShell a cada 15 segundos por padrão e agrupa vários processos do mesmo aparelho físico:

```text
=== DEVORAR HARDWARE / atualização ===
PC: NOTEBOOK | CPU: ... | núcleos lógicos: 8 | RAM: 8.00 GiB total / 4.21 GiB disponível
Jobs: fila=180 ativos=4 concluídos=72 falhos=0 | dispositivos=3
- realme RMX3830 [ONLINE] | CPU: ... | núcleos: 8 | RAM: 3.76 GiB total / 1.42 GiB disponível
- outro-android [ONLINE] | CPU: ... | núcleos: 8 | RAM: ...
=== FIM HARDWARE ===
```

Cada worker envia telemetria atualizada de RAM por heartbeat.

Para mudar o intervalo do painel:

```powershell
python windows_server.py --status-seconds 30
```

Use `--status-seconds 0` para desligar somente a impressão periódica.

## Execução direta

No Windows ou outro PC com Python:

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

`cluster-state/` é ignorado pelo Git para não publicar token, banco ou resultados locais.

## Resultados e primeiro treino experimental

No PC:

```powershell
python distributed_export.py
python distributed_train_head.py --rank 8 --epochs 1000
```

O treino do `student-head` precisa das dependências de treinamento no PC. Os workers Termux usados para buscar/calcular as fatias remotas utilizam somente Python e o código do próprio projeto.

## Ferramentas Frontier locais

```powershell
python frontier_scan.py
python frontier_tensor_probe.py
python frontier_head_parity.py
```

O scanner e as sondas trabalham com HTTP Range para evitar materializar um shard inteiro quando a operação precisa somente de uma região específica.

## Segurança

- O token do cluster autentica workers por HMAC-SHA256.
- Nonces e janela de tempo reduzem replay.
- Jobs possuem lease e voltam à fila quando um worker cai.
- O Firewall do Windows é limitado à rede local.
- Não exponha a porta do coordenador diretamente à Internet.
- Preserve licenças, revisões e atribuições dos checkpoints utilizados.

## Testes locais

```powershell
python -m pip install -e ".[test]"
python -m pytest
```

Veja também `docs/DISTRIBUTED-TERMUX.md`.
