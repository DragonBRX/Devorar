# Devorar

Devorar é um protótipo de pesquisa do DragonBRX para assimilação paramétrica e computação distribuída sobre partes de checkpoints remotos.

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

> Estado científico: o estágio distribuído atual executa fatias BF16 de `head.weight` com ativações sintéticas. Isso já é computação real sobre pesos remotos, mas ainda não é o forward completo do DeepSeek nem uma destilação completa de linguagem.

## Modo principal: mesma Wi-Fi, sem digitar IP, porta ou token

A versão atual possui descoberta automática na rede local. No Termux, o usuário não precisa informar nenhum endereço.

Internamente o PC usa TCP `8765` para o coordenador e UDP `8764` para a descoberta automática. Esses detalhes ficam escondidos do fluxo normal. O Termux envia uma busca na rede local, o PC responde diretamente e o celular usa o IP de origem da resposta para conectar sozinho.

```text
Termux
  │
  ├── procura um PC Devorar na Wi-Fi ───────────────>
  │                                                  PC
  │                                                   │
  <── resposta direta com configuração do cluster ────┘
  │
  └── conecta automaticamente
```

## Windows: instalar ou atualizar

Abra o PowerShell e cole:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; irm https://raw.githubusercontent.com/DragonBRX/Devorar/main/windows_install.ps1 | iex
```

O instalador verifica Python, instala Python 3.12 pelo `winget` quando necessário, baixa o projeto, configura o Firewall somente para a sub-rede local e abre o serviço Devorar.

## Windows: reinstalação limpa

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; irm https://raw.githubusercontent.com/DragonBRX/Devorar/main/windows_reinstall.ps1 | iex
```

A reinstalação para instâncias antigas, apaga `~/Devorar`, baixa o projeto novamente, configura o Firewall e inicia tudo automaticamente.

## Como saber que o PC terminou de iniciar

O terminal não usa mais uma linha indefinida de “iniciando” como estado final. Quando o serviço estiver pronto aparece claramente:

```text
============================================================
                 DEVORAR ONLINE
============================================================
Servidor do PC: PRONTO
Descoberta automática na Wi-Fi: ATIVA
Estado: AGUARDANDO DISPOSITIVOS TERMUX
Você pode deixar esta janela aberta. O servidor já iniciou.
============================================================
```

Isso significa que o PC já terminou a inicialização e está esperando celulares.

A preparação dos jobs do DeepSeek ocorre em segundo plano. Ela não bloqueia a conexão dos aparelhos. Quando termina aparece `SEGUNDO PLANO CONCLUÍDO`; se houver falha temporária, a mensagem deixa explícito que o servidor continua `ONLINE`.

O painel também mostra:

```text
=== DEVORAR / ATUALIZAÇÃO ===
PC: ...
Jobs: fila=... ativos=... concluídos=... falhos=... | dispositivos=0
STATUS: AGUARDANDO DISPOSITIVOS TERMUX NA MESMA WI-FI...
=== FIM STATUS ===
```

Quando um Termux encontra o PC:

```text
[NOVO DISPOSITIVO] Termux encontrado na rede: 192.168.x.x
[NOVO DISPOSITIVO] Configuração enviada. Aguardando o worker registrar o hardware...
```

## Termux: instalação do zero

Com o PC Devorar ligado na mesma Wi-Fi, cole este bloco em qualquer Termux recém-instalado:

```bash
pkg update -y && pkg install -y python git tmux && \
rm -rf "$HOME/Devorar" && \
git clone --depth 1 https://github.com/DragonBRX/Devorar.git "$HOME/Devorar" && \
cd "$HOME/Devorar" && chmod +x termux_auto_install.sh && \
./termux_auto_install.sh
```

Não existe IP, porta ou token nesse bloco.

O celular procura o PC automaticamente. Ao terminar, aparece:

```text
========================================
       DEVORAR TERMUX PRONTO
========================================
PC: NOME-DO-PC
Estado: WORKER INICIADO
Nenhum IP, porta ou token precisou ser digitado.
O celular agora aguarda trabalhos do PC.
========================================
```

## Termux: reinstalação

Depois que o celular já estiver configurado:

```bash
devorar-worker reinstall
```

O comando para o worker, apaga o clone, baixa o projeto novamente, procura o PC pela Wi-Fi e reconecta automaticamente.

Outros comandos:

```bash
devorar-worker status
devorar-worker hardware
devorar-worker logs
devorar-worker restart
devorar-worker stop
devorar-worker start
```

`status` e `hardware` mostram CPU, núcleos, RAM total e RAM disponível no próprio celular.

## Painel de hardware no PC

O PowerShell atualiza o painel periodicamente e agrupa múltiplos processos do mesmo aparelho físico:

```text
=== DEVORAR / ATUALIZAÇÃO ===
PC: NOTEBOOK | CPU: ... | núcleos lógicos: 8 | RAM: 8.00 GiB total / 4.21 GiB disponível
Jobs: fila=180 ativos=4 concluídos=72 falhos=0 | dispositivos=2
- realme RMX3830 [ONLINE] | CPU: ... | núcleos: 8 | RAM: 3.76 GiB total / 1.42 GiB disponível
- outro-android [ONLINE] | CPU: ... | RAM: ...
=== FIM STATUS ===
```

## Limitação de rede importante

A descoberta automática exige que PC e celular estejam na mesma rede IPv4 e que o roteador permita comunicação entre clientes Wi-Fi. Redes com `AP isolation`, `client isolation` ou redes de convidados podem bloquear a descoberta mesmo quando os dois aparelhos mostram o mesmo nome de Wi-Fi.

O modo zero-config deve ser usado em rede local confiável. Qualquer dispositivo na mesma sub-rede que implemente o protocolo de descoberta pode solicitar a configuração do cluster, então não use esse modo em Wi-Fi público.

## Resultados

No PC:

```powershell
python distributed_export.py
python distributed_train_head.py --rank 8 --epochs 1000
```

## Testes locais

```powershell
python -m pip install -e ".[test]"
python -m pytest
```

Os testes usam fixtures pequenas e não precisam baixar o checkpoint completo do DeepSeek.
