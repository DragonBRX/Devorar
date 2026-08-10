# Devorar

Experimento do DragonBRX para **assimilação paramétrica direta**: transformar
o delta de um modelo doador homólogo em um novo checkpoint autônomo, sem usar
respostas, logits ou cadeias de raciocínio do doador como professor.

> Estado: protótipo de pesquisa Classe A. Ele funciona somente quando base e
> doador têm a mesma anatomia de tensores. Não existe aqui uma técnica capaz de
> ler pensamentos dos pesos ou converter sem perdas qualquer arquitetura em
> qualquer outra.

## Executar no Google Colab

Clone o repositório e execute um único arquivo:

```python
!git clone https://github.com/DragonBRX/Devorar.git
!python Devorar/start_colab.py
```

`start_colab.py` instala as dependências, baixa as revisões fixadas, mede a base,
assimila os parâmetros, libera o doador, recarrega e testa o checkpoint final,
grava os manifestos e cria `/content/devorar-output.zip`. Não é preciso copiar
um notebook nem montar uma célula grande. Ao repetir o comando, a saída anterior
só é substituída se contiver o marcador de propriedade criado pelo Devorar; a
troca completa ocorre atomicamente depois que a nova execução termina.
As dependências da V2 fixam `transformers 5.14.1` e uma versão compatível do
Hugging Face Hub, substituindo a combinação antiga que gerou o aviso de conflito
com o Gradio no primeiro piloto.

O Colab fornece recursos dinamicamente: RAM, GPU, duração e limites não são
garantidos. O experimento usa modelos de 360 milhões de parâmetros para caber
com folga razoável em uma sessão comum e também possui fallback para CPU.

## Rodar o modelo criado

Quando o primeiro comando terminar, execute o checkpoint assimilado em uma
segunda célula:

```python
!python Devorar/run_model.py --prompt "Quem é você e como foi criado?"
```

Antes de gerar texto, `run_model.py` confere o inventário e o SHA-256 dos
artefatos, carrega somente arquivos locais com `trust_remote_code=False` e
recalcula o hash lógico do `state_dict`. O modelo doador não é carregado nessa
execução. Sem `--prompt`, o arquivo roda uma demonstração curta. Ao concluir o
build, o primeiro comando também imprime o SHA-256 do manifesto e uma versão do
segundo comando com `--expected-manifest-sha256`; prefira essa linha quando
quiser detectar alterações ocorridas entre construção e inferência.

Opções úteis:

```python
# Testar somente checkpoint/tokenizer, sem a identidade de apresentação
!python Devorar/run_model.py --raw --prompt "Complete: 2 + 3 ="

# Comparar as mesmas entradas com a base pinada (nunca com o doador)
!python Devorar/run_model.py --compare-host --prompt "Responda somente SIM: a água contém oxigênio?"

# Emitir também o relatório verificável em JSON
!python Devorar/run_model.py --prompt "Explique adaptação em uma frase." --json-output /content/devorar-inference.json
```

“DragonBRX Assimilated” é uma identidade declarada pela camada de apresentação.
Os pesos assimilados são realmente diferentes e produzem suas próprias
ativações a cada inferência, mas o programa não extrai, copia nem verifica uma
cadeia de pensamento privada do doador.

## V3 Frontier: mapear sem baixar o modelo inteiro

A V3 possui um scanner independente para checkpoints que não cabem no Colab.
Ele está fixado no `DeepSeek-V4-Flash` oficial e lê apenas o índice, os configs,
os primeiros 8 bytes de cada shard e seus cabeçalhos Safetensors por HTTP Range:

```python
!python Devorar/start_frontier_colab.py
```

O comando cria `/content/devorar-frontier-plan.zip`. Ele **não** baixa shards
inteiros, não instancia o doador, não faz inferência e não cria um checkpoint.
Seu objetivo é responder primeiro se uma receita cabe no armazenamento, RAM e
runtime, sem gastar dezenas de minutos transferindo pesos inutilmente.
Cada intervalo aceito fica ligado a um ETag forte e recebe SHA-256; URLs
assinadas são registradas sem a query, e o token do Hub não entra no manifesto.

Na revisão imutável atualmente fixada, o mapa remoto contém 69.187 tensores em
46 shards e aproximadamente 148,65 GiB de payload. Cerca de 140,25 GiB são os
256 experts roteados. O tronco, embeddings, cabeça, roteador e expert
compartilhado já somam aproximadamente 8,40 GiB no formato misto armazenado.
Manter ou fundir o equivalente a apenas um expert elevaria o piso estimado para
8,95 GiB, mas isso não prova execução na T4 nem preservação de conhecimento.

O gate bloqueia materialização porque o checkpoint usa FP4/FP8, enquanto a T4
exige uma conversão/runtime compatível, e porque um colapso de 256 experts para
um ou poucos experts está muito além da redução conservadora validada pela
pesquisa atual. O manifesto registra essa reprovação em vez de chamar uma
estimativa de “modelo pronto”.

### Ler micropedaços reais dos pesos

Para conferir a hipótese do “pedaço do parâmetro” sem baixar um shard inteiro:

```python
!python Devorar/frontier_tensor_probe.py
```

A sonda localiza seis tensores de papéis diferentes e lê pequenas janelas do
início, meio e fim por HTTP Range. Ela grava hashes, offsets e estatísticas em
`/content/dragonbrx-frontier-tensor-probe.lira.json`; não grava os bytes crus,
não instancia o DeepSeek e não executa inferência. Isso confirma que o trecho é
real e permite comparar armazenamento, dispersão e papel estrutural. Não
transforma o trecho em texto: sem tokenizer, ativações, camadas vizinhas e
cabeça de saída, um parâmetro ou micropedaço não consegue responder a um prompt.

### Executar uma fatia real dos pesos

O gate seguinte transforma uma quantidade delimitada desses pesos em
computação. Em um Colab com GPU, execute:

```python
!python Devorar/frontier_head_parity.py --require-cuda
```

O teste lê por HTTP Range 16 linhas **completas** de `head.weight`, distribuídas
pelo vocabulário do checkpoint fixado. São 131.072 bytes de payload BF16, em vez
do shard de aproximadamente 1 GiB. As linhas são decodificadas, multiplicadas na
GPU por uma entrada oculta sintética e determinística e comparadas com uma
referência CPU que acumula os mesmos produtos com `math.fsum`. O comando falha se
o erro de cada logit parcial ultrapassar `atol + rtol × |referência|` e grava
`/content/dragonbrx-frontier-head-parity.lira.json` com hashes, intervalos,
dispositivo, tolerâncias e erros.

Esse resultado é uma primeira prova de **pesos remotos → operação numérica**. A
entrada não é uma ativação produzida pelo DeepSeek; somente algumas linhas da
cabeça são executadas; o grafo transformer, o roteamento MoE e o vocabulário
completo não são avaliados. Portanto, mesmo quando o gate passa, ele não é
paridade dos logits do modelo completo e não autoriza atribuir uma frase ao
DeepSeek. Esse rótulo exige executar o caminho completo para os mesmos tokens e
comparar todos os logits do próximo token com um runtime de referência.


## Cluster distribuído: PC coordenador + celulares Termux

A etapa distribuída permite juntar vários Android/Termux e PCs. O PC mantém uma fila persistente, enquanto cada worker lê somente os intervalos necessários dos pesos remotos do DeepSeek por HTTP Range, executa a operação localmente e devolve o resultado numérico ao coordenador.

### Instalação automática do Termux em um único bloco

No PC, dentro do repositório:

```bash
python distributed_server.py --host 0.0.0.0 --port 8765
```

Ao iniciar, o coordenador imprime entre `TERMUX: COLE ESTE BLOCO INTEIRO` e `FIM DO BLOCO TERMUX` um comando completo já contendo o IP local detectado, a porta e o token do cluster. Em **cada celular com o aplicativo Termux recém-instalado**, cole o bloco inteiro e pressione Enter. Não é necessário instalar Python, Git ou tmux manualmente antes.

O bloco gerado tem esta estrutura; os valores reais de `IP_DO_PC` e `TOKEN_GERADO_PELO_PC` saem preenchidos automaticamente pelo coordenador:

```bash
pkg update -y && pkg install -y python git tmux && \
if [ -d "$HOME/Devorar/.git" ]; then git -C "$HOME/Devorar" pull --ff-only; else git clone --depth 1 https://github.com/DragonBRX/Devorar.git "$HOME/Devorar"; fi && \
cd "$HOME/Devorar" && chmod +x termux_install.sh && \
DEVORAR_SERVER=http://IP_DO_PC:8765 DEVORAR_CLUSTER_TOKEN=TOKEN_GERADO_PELO_PC DEVORAR_PROCESSES=1 ./termux_install.sh
```

O instalador:

- atualiza o índice de pacotes do Termux;
- instala `python`, `git` e `tmux`;
- usa o repositório em `~/Devorar`;
- grava IP, token, nome do worker e quantidade de processos com permissão `600`;
- instala o comando `devorar-worker` no próprio `$PREFIX/bin`;
- inicia o worker dentro de uma sessão `tmux` em segundo plano;
- mantém um supervisor que reconecta automaticamente se o PC ou a rede ficarem indisponíveis;
- deixa o worker em polling contínuo mesmo depois de concluir os jobs atuais, pronto para novos trabalhos enviados pelo PC.

Se a detecção automática do IP escolher a interface errada, inicie o PC especificando o endereço que os celulares enxergam:

```bash
python distributed_server.py --host 0.0.0.0 --port 8765 --advertise-host 192.168.1.10
```

Para já configurar dois processos em cada celular no bloco gerado:

```bash
python distributed_server.py --termux-processes 2
```

Depois da instalação, os comandos úteis no celular são:

```bash
devorar-worker status
devorar-worker logs
devorar-worker restart
devorar-worker stop
devorar-worker start
```

É possível exportar os resultados no PC com `python distributed_export.py` e treinar o primeiro surrogate low-rank da cabeça remota com `python distributed_train_head.py`. O worker atual usa somente a biblioteca padrão do Python e o código Frontier existente; PyTorch é necessário somente no PC para o treinamento do `student-head`.

Este estágio já é computação distribuída real sobre pesos remotos, mas **ainda não é uma distilação completa do DeepSeek**. Ele trabalha com linhas BF16 de `head.weight` e ativações sintéticas. O transformer inteiro, o roteamento MoE e os formatos quantizados do corpo ainda precisam de um runtime paginado validado antes que o DeepSeek possa ser tratado como professor de linguagem via pesos remotos. Veja [docs/DISTRIBUTED-TERMUX.md](docs/DISTRIBUTED-TERMUX.md).

## Microscópio de parâmetros e intervenção local

Depois de construir o checkpoint pequeno da V2, é possível medir quais grupos
de pesos mais influenciaram uma resposta específica:

```python
!python Devorar/probe_model.py --prompt "Quem é você e como foi criado?"
```

O microscópio usa uma única retropropagação `gradiente × peso` para ranquear
tensores, camadas e papéis estruturais, encontra alguns pesos locais de maior
influência e mede os grupos principais com atenuações pequenas e reversíveis.
Ao final, confere que o hash do checkpoint voltou exatamente ao valor anterior
e grava `/content/dragonbrx-parameter-probe.lira.json`.

Para intervir em até oito tensores escolhidos, em vez de aceitar o ranking
automático, repita o nome canônico exato:

```python
!python Devorar/probe_model.py --prompt "Quem é você?" --target-tensor model.layers.10.self_attn.q_proj.weight
```

Esse relatório descreve influência **condicionada ao prompt e à resposta**. Um
parâmetro isolado não possui tokenizer, contexto ou capacidade de responder, e
não recebe um significado permanente como “este peso contém esta frase”. Para
testar causalidade são necessários o restante do grafo e uma intervenção. Em
um modelo frontier, uma inferência paginada poderia carregar uma camada e os
experts escolhidos pelo roteador de cada vez; isso reduz o pico de RAM, mas
continua lendo grande parte do caminho ativo para cada token e exige um runtime
específico.

Os pesos individuais listados permanecem `proxy_only`: são coordenadas úteis
para orientar testes, não conceitos decodificados. Uma afirmação causal válida
tem a forma “reduzir este tensor em 5% mudou a perda desta resposta neste
prompt”, e não “este tensor significa identidade”. A resposta repetitiva do
primeiro piloto também não foi sorteada: a geração era *greedy* determinística;
o texto ruim revelou degeneração do checkpoint, não aleatoriedade do teste.

## Primeiro teste

| Papel | Checkpoint fixado | Licença |
| --- | --- | --- |
| Hospedeiro/base | `HuggingFaceTB/SmolLM2-360M@f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | Apache-2.0 |
| Doador | `HuggingFaceTB/SmolLM2-360M-Instruct@a10cc1512eabd3dde888204e902eca88bddb4951` | Apache-2.0 |

O cartão oficial informa que a variante Instruct foi criada por SFT e depois
DPO a partir da família SmolLM2. O código ainda confere nomes, formas, tipos,
configuração essencial e vocabulário antes de tocar em qualquer tensor.

O primeiro piloto real passou todos os gates: 176.212.442 valores mudaram,
o escore das quatro regras subiu `+0,50` e a razão de perplexidade foi
`1,157764`. A V2 preserva a receita observada, mas reduziu o limite de retenção
para `1,25` e adicionou um piso absoluto de acerto. Veja o
[relatório auditável do primeiro piloto](docs/FIRST-PILOT.md).

## O que “devorar” significa neste protótipo

Para cada tensor flutuante compatível:

```text
delta     = doador - base
delta_DARE = drop_aleatório(delta) / probabilidade_de_manter
candidato = base + força_de_assimilação * delta_DARE
```

O descarte é determinístico pela seed. Pesos compartilhados (*tied weights*)
são alterados uma única vez. Tensores incompatíveis, valores `NaN`/`Inf`,
configurações divergentes e tokenizers incompatíveis interrompem o processo.

Durante a construção:

- `donor_forward_calls_build == 0`;
- não há consulta a respostas ou logits do doador;
- não há treinamento professor–aluno;
- `trust_remote_code=False` e somente checkpoints `safetensors` são aceitos;
- o doador é descarregado antes da inferência do candidato;
- o resultado precisa funcionar como checkpoint independente.

Isso é próximo de **Task Arithmetic + DARE**, não uma destilação. A hipótese
vem do artigo *Language Models are Super Mario: Absorbing Abilities from
Homologous Models as a Free Lunch*.

## O que o teste mede

- fração real do delta mantida;
- norma, cosseno e projeção do delta assimilado sobre o delta do doador;
- diferença do candidato em relação à base e ao doador;
- perplexidade em um corpus fixo de retenção;
- regras determinísticas simples de seguir instruções;
- ausência de `NaN`/`Inf`;
- hashes dos artefatos e independência do checkpoint final.

Resultados qualitativos não provam que o modelo adquiriu raciocínio geral. Um
ganho só conta como evidência se vier acompanhado de retenção e repetição com
outras seeds, receitas e avaliações externas.

## Saída

O diretório escolhido por `--output-dir` contém:

```text
devorar-output/
├── .devorar-output-v1              # marcador para overwrite seguro
├── assimilated-model/                 # checkpoint Hugging Face autônomo
├── assimilation-manifest.lira.json    # receita, linhagem, métricas e hashes
├── evaluation.json
├── THIRD_PARTY_MODELS.md
└── run-summary.json

devorar-output.zip                     # pacote criado automaticamente
```

`assimilation-manifest.lira.json` é um **manifesto Lira experimental**. Ele não
é apresentado como o contêiner binário Lira canônico do DragonBRX. A promoção
para a memória estável exigirá um compilador e gates próprios no projeto
principal.

## Testes locais

Os testes não baixam modelos grandes:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

Eles usam redes minúsculas e transportes em memória para verificar determinismo,
incompatibilidades, pesos compartilhados, proibição de `forward` do doador,
manifestos, runner, launcher e paridade da execução delimitada sem acessar o
checkpoint frontier durante a suíte local.

## Limites e segurança

- A V2 aceita apenas modelos homólogos; arquiteturas diferentes precisam de
  alinhamento, *stitching* ou adaptadores e ficam fora deste teste.
- Assimilar pesos também pode assimilar vieses, falhas ou *backdoors*.
- O checkpoint resultante é derivado dos modelos de origem; preserve licença,
  atribuição, revisões e hashes antes de redistribuí-lo.
- A licença do código deste repositório ainda deve ser definida pelo mantenedor;
  ela é uma decisão separada das licenças Apache-2.0 dos modelos de terceiros.
- Não envie automaticamente o candidato para a Lira estável nem substitua um
  modelo de produção sem avaliação, canário e rollback.
- Execute a verificação em um diretório local que não esteja sendo alterado por
  outro processo; hashes detectam alterações, mas não eliminam uma troca de
  arquivos concorrente entre verificação e carregamento.

Veja [o mapa científico e as próximas classes de compatibilidade](docs/RESEARCH.md).
