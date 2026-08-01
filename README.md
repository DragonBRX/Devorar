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

O Colab fornece recursos dinamicamente: RAM, GPU, duração e limites não são
garantidos. O experimento usa modelos de 360 milhões de parâmetros para caber
com folga razoável em uma sessão comum e também possui fallback para CPU.

## Primeiro teste

| Papel | Checkpoint fixado | Licença |
| --- | --- | --- |
| Hospedeiro/base | `HuggingFaceTB/SmolLM2-360M@f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | Apache-2.0 |
| Doador | `HuggingFaceTB/SmolLM2-360M-Instruct@a10cc1512eabd3dde888204e902eca88bddb4951` | Apache-2.0 |

O cartão oficial informa que a variante Instruct foi criada por SFT e depois
DPO a partir da família SmolLM2. O código ainda confere nomes, formas, tipos,
configuração essencial e vocabulário antes de tocar em qualquer tensor.

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

Eles usam redes minúsculas para verificar determinismo, incompatibilidades,
pesos compartilhados, proibição de `forward` do doador, manifesto, runner e
launcher automático.

## Limites e segurança

- A V1 aceita apenas modelos homólogos; arquiteturas diferentes precisam de
  alinhamento, *stitching* ou adaptadores e ficam fora deste teste.
- Assimilar pesos também pode assimilar vieses, falhas ou *backdoors*.
- O checkpoint resultante é derivado dos modelos de origem; preserve licença,
  atribuição, revisões e hashes antes de redistribuí-lo.
- A licença do código deste repositório ainda deve ser definida pelo mantenedor;
  ela é uma decisão separada das licenças Apache-2.0 dos modelos de terceiros.
- Não envie automaticamente o candidato para a Lira estável nem substitua um
  modelo de produção sem avaliação, canário e rollback.

Veja [o mapa científico e as próximas classes de compatibilidade](docs/RESEARCH.md).
