# Cluster distribuído PC + Termux

Esta etapa adiciona um coordenador no PC e workers leves para Android/Termux ou outros PCs.
O objetivo é usar vários dispositivos para executar trabalho numérico sobre fatias reais dos pesos remotos do DeepSeek sem baixar o checkpoint completo em cada aparelho.

## Arquitetura

```text
Hugging Face / DeepSeek-V4-Flash
          │
          │ HTTP Range
          ├──────────────► celular Termux 1 ─┐
          ├──────────────► celular Termux 2 ─┤
          ├──────────────► celular Termux N ─┤
          │                                  │ resultados numéricos
          └──────────────► worker no PC ─────┤
                                             ▼
                                      PC coordenador
                                      SQLite persistente
                                             │
                                             ├─ exporta dataset numérico
                                             └─ treina student-head experimental
```

O PC cria jobs a partir do índice e do cabeçalho Safetensors. Cada job contém somente identidade imutável da fonte, posição do tensor, algumas linhas e seeds determinísticas. O worker lê somente essas linhas diretamente do Hugging Face, decodifica BF16, executa projeções locais e devolve logits parciais ao PC.

O cluster não envia os pesos completos para o PC. O coordenador persiste fila, leases, tentativas, workers e resultados em `cluster-state/cluster.sqlite3`. Se um celular cair, o lease expira e o job volta para a fila.

## 1. Iniciar o coordenador no PC

No diretório do projeto:

```bash
python distributed_server.py --host 0.0.0.0 --port 8765
```

Na primeira execução são criados:

```text
cluster-state/cluster-token.txt
cluster-state/cluster.sqlite3
cluster-state/plan.json
```

O coordenador também imprime um bloco completo de instalação do Termux já preenchido com o endereço do PC e o token. O segredo continua sendo usado em HMAC-SHA256 nas requisições; o bloco o grava em `~/.config/devorar/worker.env` com permissão `600`.

Se o endereço detectado automaticamente não for acessível pelos celulares, force o IP da interface LAN:

```bash
python distributed_server.py --host 0.0.0.0 --port 8765 --advertise-host 192.168.1.10
```

Para gerar o bloco configurando mais processos por aparelho:

```bash
python distributed_server.py --termux-processes 2
```

O padrão cria 256 jobs, com 4 linhas remotas e 8 ativações sintéticas por job:

```bash
python distributed_server.py --job-count 256 --rows-per-job 4 --samples-per-job 8
```

Aumentar `--samples-per-job` aumenta principalmente o cálculo local sem aumentar proporcionalmente os bytes de pesos baixados, porque as mesmas linhas são reutilizadas para várias ativações.

A fonte também é configurável, mas a revisão deve ser um SHA imutável de 40 caracteres:

```bash
python distributed_server.py --source-model deepseek-ai/DeepSeek-V4-Flash --source-revision SHA_IMUTAVEL_DE_40_CARACTERES
```

O padrão continua usando a revisão já fixada pelo projeto para que execuções e manifestos sejam reproduzíveis.

## 2. Preparar cada Android com Termux

Depois de instalar o aplicativo Termux, **não instale nada manualmente**. Pegue o bloco que o PC imprimiu e cole inteiro no terminal do celular.

A forma genérica é:

```bash
pkg update -y && pkg install -y python git tmux && \
if [ -d "$HOME/Devorar/.git" ]; then git -C "$HOME/Devorar" pull --ff-only; else git clone --depth 1 https://github.com/DragonBRX/Devorar.git "$HOME/Devorar"; fi && \
cd "$HOME/Devorar" && chmod +x termux_install.sh && \
DEVORAR_SERVER=http://IP_DO_PC:8765 DEVORAR_CLUSTER_TOKEN=TOKEN_GERADO_PELO_PC DEVORAR_PROCESSES=1 ./termux_install.sh
```

O bloco real gerado pelo PC já substitui os placeholders. `termux_install.sh` instala o ambiente, registra uma identidade persistente para o aparelho e inicia `devorar-worker` em uma sessão `tmux` desacoplada do terminal.

O worker agora permanece ativo quando a fila fica vazia. Assim, concluir um lote não encerra o celular: ele continua consultando o PC e pega automaticamente jobs criados depois. Um supervisor reinicia a conexão após falhas de rede ou reinício do coordenador.

Comandos locais:

```bash
devorar-worker status
devorar-worker logs
devorar-worker restart
devorar-worker stop
devorar-worker start
```

Em aparelhos com pouca RAM ou que esquentam muito, prefira `DEVORAR_PROCESSES=1`. Outros computadores podem executar `distributed_worker.py` diretamente ou usar uma configuração equivalente.

## 3. Acompanhar o cluster

O coordenador mostra jobs concluídos no terminal. O estado também pode ser consultado por um cliente assinado através de `/v1/status`.

A base SQLite torna a execução retomável: fechar e abrir `distributed_server.py` novamente com o mesmo `--state-dir` preserva resultados e jobs já concluídos.

## 4. Exportar o resultado dos celulares

No PC:

```bash
python distributed_export.py
```

Isso cria:

```text
cluster-state/teacher-head-samples.jsonl
```

Cada linha contém a fonte, seed, hash da ativação e logits das linhas da cabeça de saída processadas por um worker.

## 5. Treinamento experimental no PC

Depois que os jobs necessários terminarem:

```bash
python distributed_train_head.py --rank 8 --epochs 1000
```

O comando cria:

```text
cluster-state/student-head.safetensors
cluster-state/student-head-manifest.json
```

Esse estágio faz treinamento real: um pequeno operador low-rank aprende a reproduzir os alvos que os celulares calcularam a partir das linhas BF16 remotas de `head.weight`.

Ele é deliberadamente chamado de `student-head`, e não de LLM. Não possui tokenizer, transformer completo, roteamento MoE nem geração de texto.

## O que já funciona

- vários celulares e PCs podem trabalhar ao mesmo tempo;
- cada worker baixa somente intervalos necessários dos pesos remotos;
- o DeepSeek completo não é materializado nos celulares;
- processamento numérico ocorre no worker;
- resultados vão para o PC;
- fila é persistente e retomável;
- jobs com worker desconectado voltam à fila;
- autenticação HMAC evita transmitir o segredo do cluster;
- existe exportação dos sinais do professor;
- existe um primeiro treinamento real de um surrogate de cabeça de saída.

## O que ainda falta para o objetivo completo

O `DeepSeek-V4-Flash` é MoE e usa formatos quantizados no corpo do checkpoint. A versão atual do Devorar ainda não executa o transformer inteiro por streaming. Portanto o cluster ainda não pode afirmar que obteve respostas, logits completos ou comportamento linguístico do DeepSeek apenas lendo os pesos por Range.

Para transformar esta infraestrutura em treinamento de um LLM estudante completo, os gates seguintes são:

1. decodificar de forma validada todos os formatos usados nas camadas relevantes, inclusive os pesos quantizados;
2. executar embedding, atenção, normalizações, mHC, roteadores e experts na ordem correta;
3. distribuir esse grafo por jobs sem baixar o checkpoint inteiro em um único dispositivo;
4. comparar os logits completos com um runtime oficial para os mesmos tokens;
5. somente depois usar esses logits/ativações reais como supervisão de distilação de um estudante;
6. distribuir também gradientes ou lotes do estudante entre os workers, caso o custo de comunicação compense.

A infraestrutura nova foi feita para esses próximos tipos de job serem adicionados sem trocar o protocolo PC ↔ Termux.
