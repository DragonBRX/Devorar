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

## Modo principal: mesma Wi-Fi, zero configuração no celular

A versão atual possui descoberta automática na rede local. O usuário do Termux não precisa digitar IP, porta ou token.

Internamente o PC usa:

- TCP `8765` para o coordenador;
- UDP `8764` somente para descoberta automática.

Esses valores são configurados automaticamente. O celular envia um pedido de descoberta por broadcast na rede local. O PC responde diretamente ao celular. O Termux usa o IP de origem dessa resposta para descobrir o endereço correto do PC, recebe a configuração do cluster e conecta sozinho.

```text
Termux
  │
  ├── "Existe um Devorar nesta Wi-Fi?" ── broadcast UDP ──>
  │                                                    PC
  │                                                     │
  <── resposta direta: coordenador + configuração ──────┘
  │
  └── conecta automaticamente ao PC
```

Isso evita depender da detecção de `192.168.x.x` pelo PowerShell.

## Windows: instalar ou atualizar

Abra o PowerShell e cole:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; irm https://raw.githubusercontent.com/DragonBRX/Devorar/main/windows_install.ps1 | iex
```

O instalador verifica Python, instala Python 3.12 pelo `winget` quando necessário, baixa o projeto, configura o Firewall para a sub-rede local e inicia o coordenador e a descoberta automática.

O Windows pode mostrar o UAC. A regra criada permite somente a sub-rede local para as portas do Devorar.

## Windows: reinstalação limpa

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; irm https://raw.githubusercontent.com/DragonBRX/Devorar/main/windows_reinstall.ps1 | iex
```

A reinstalação para instâncias antigas do servidor, apaga `~/Devorar`, baixa o projeto novamente, configura o Firewall e inicia tudo automaticamente.

## O que aparece no PowerShell

A inicialização deve mostrar algo semelhante a:

```text
Firewall: conexao TCP e descoberta UDP liberadas somente para a rede local.
Iniciando Devorar no PowerShell com descoberta automatica na rede local...
Coordenador Devorar ativo na rede local (TCP 8765).
Descoberta automática ativa (UDP 8764). No Termux não é necessário digitar IP, porta ou token.

=== DEVORAR HARDWARE / inicialização ===
PC: ...
- nenhum celular conectado ainda
=== FIM HARDWARE ===

=== TERMUX: INSTALAÇÃO AUTOMÁTICA NA MESMA WI-FI ===
...
=== FIM INSTALAÇÃO TERMUX ===
```

A preparação dos jobs do DeepSeek começa depois, em segundo plano. Os celulares podem conectar antes dela terminar.

## Termux: instalação do zero sem IP, porta ou token

Com o PC Devorar ligado na mesma Wi-Fi, cole este mesmo bloco em qualquer Termux recém-instalado:

```bash
pkg update -y && pkg install -y python git tmux && \
rm -rf "$HOME/Devorar" && \
git clone --depth 1 https://github.com/DragonBRX/Devorar.git "$HOME/Devorar" && \
cd "$HOME/Devorar" && chmod +x termux_auto_install.sh && \
./termux_auto_install.sh
```

Não há IP, porta nem token nesse bloco.

O celular faz sozinho:

1. instala Python, Git e tmux;
2. procura um PC Devorar na mesma rede Wi-Fi;
3. recebe a resposta do PC;
4. identifica automaticamente o endereço do coordenador;
5. recebe a configuração do cluster;
6. detecta CPU, núcleos e RAM do celular;
7. inicia o worker em `tmux`;
8. fica aguardando os jobs do PC.

Enquanto procura, o Termux mostra:

```text
Procurando um PC Devorar na mesma rede Wi-Fi...
PC encontrado: NOME-DO-PC em 192.168.x.x. Conectando automaticamente...
```

O IP é apenas mostrado como informação; você não precisa digitá-lo.

## Termux: reinstalação limpa automática

O PowerShell também imprime um bloco de reinstalação que não contém IP, porta ou token. Depois que o celular já estiver instalado, também é possível usar:

```bash
devorar-worker reinstall
```

Esse comando para o worker, apaga o clone, baixa o projeto novamente, procura o PC outra vez pela Wi-Fi e reconecta automaticamente.

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
=== DEVORAR HARDWARE / atualização ===
PC: NOTEBOOK | CPU: ... | núcleos lógicos: 8 | RAM: 8.00 GiB total / 4.21 GiB disponível
Jobs: fila=180 ativos=4 concluídos=72 falhos=0 | dispositivos=2
- realme RMX3830 [ONLINE] | CPU: ... | núcleos: 8 | RAM: 3.76 GiB total / 1.42 GiB disponível
- outro-android [ONLINE] | CPU: ... | RAM: ...
=== FIM HARDWARE ===
```

## Limitação de rede importante

A descoberta automática exige que PC e celular estejam na mesma rede IPv4 e que o roteador permita comunicação entre clientes Wi-Fi. Redes com `AP isolation`, `client isolation` ou redes de convidados podem bloquear a descoberta mesmo quando os dois aparelhos mostram o mesmo nome de Wi-Fi.

O pareamento zero-config também reduz a barreira de autenticação: qualquer dispositivo na mesma sub-rede que execute o protocolo de descoberta pode solicitar a configuração do cluster. Use esse modo em uma rede local confiável, como sua rede doméstica, e não em Wi-Fi público.

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
