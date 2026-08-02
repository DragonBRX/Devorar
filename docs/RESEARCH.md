# Mapa científico — assimilação paramétrica Lira

## Definição operacional

Assimilação paramétrica é transformar um checkpoint autorizado em módulos
reversíveis ou em um novo checkpoint autônomo, sem usar saídas do doador como
professor, medindo transferência, retenção, segurança e independência.

Pesos são regras estáticas distribuídas. Ativações semelhantes a pensamentos
só surgem durante uma execução; portanto, o projeto não afirma traduzir cada
parâmetro em uma palavra, resposta ou cadeia de raciocínio.

## Classes

| Classe | Compatibilidade | Caminho experimental |
| --- | --- | --- |
| A | Mesma base e topologia | Task Arithmetic, TIES, DARE, fatores SVD |
| B | Mesma topologia, inicialização diferente | casamento de pesos, Git Re-Basin, transporte ótimo |
| C | Mesma família, largura/profundidade/tokenizer diferentes | projeções, crescimento funcional, transplante lexical |
| D | Arquiteturas heterogêneas | *stitching*, adaptador ou órgão MoE; não fusão nativa |
| E | Somente API, sem checkpoint autorizado | impossível no modo estrito sem professor |

A V2 materializa somente a Classe A. A V3 Frontier inspeciona a anatomia de um
MoE enorme sem materializá-lo, mas ainda não afirma resolver as Classes C/D.

## Pipeline Lira proposto

```text
checkpoint externo
  -> quarentena + hash + origem + licença
  -> anatomia de tensores/config/tokenizer
  -> classificação de compatibilidade
  -> delta/alinhamento/fatoração
  -> manifesto Lira experimental imutável
  -> avaliação isolada
  -> canário
  -> promoção ou rollback
```

Gates futuros devem incluir ganho de transferência, retenção nas capacidades
centrais, perplexidade, segurança, reprodutibilidade, funcionamento standalone
e restauração exata do hospedeiro.

## V3 Frontier e inferência paginada

O scanner V3 implementa a parte verificável de um fluxo *out-of-core*:

```text
commit imutável + index.json + config.json
  -> Range 0-7 de cada shard
  -> Range do cabeçalho Safetensors
  -> validação de nomes, formas, dtype, offsets e total
  -> mapa de trunk/router/shared/routed experts
  -> estimativas de receitas
  -> autorizar ou bloquear materialização
```

O servidor precisa responder `206 Partial Content` e devolver exatamente o
intervalo solicitado; uma resposta `200` é rejeitada antes da leitura do corpo.
Assim, um shard de vários GiB pode ter sua anatomia inspecionada transferindo
apenas seu cabeçalho. Um hash do inventário prova os metadados observados, não o
conteúdo dos intervalos de pesos que ainda não foram lidos.

Cada segunda leitura do mesmo shard usa `If-Range` e precisa manter ETag forte e
tamanho. O Hub pode alternar legitimamente entre endpoints CDN/Xet; cada endpoint
ainda precisa permanecer no limite HTTPS confiável e é registrado sanitizado no
recibo, mas não define sozinho a identidade do objeto. O manifesto conserva o
SHA-256 dos bytes exatos de cada resposta aceita, além dos hashes canônicos do
JSON; credenciais e queries assinadas não são persistidas.

A sonda de micropedaços acrescenta uma evidência intermediária: busca pequenas
janelas do payload de tensores selecionados, registra o hash de cada intervalo e
calcula estatísticas sem publicar bytes crus. Isso prova quais bytes foram
amostrados, mas não prova o hash do tensor completo, significado semântico ou
capacidade de inferência. A consulta funcional continua exigindo o grafo.

Uma eventual inferência paginada é diferente de compressão. Ela mantém apenas
o estado oculto e uma camada na memória, usa o roteador para descobrir os
experts ativos, busca esses tensores por Range, calcula e descarta. O pico de RAM
cai, mas o tráfego e a latência permanecem altos e os formatos quantizados
precisam de kernels compatíveis. Cada resposta continua dependendo de um
caminho completo pelo grafo; um micropedaço isolado não gera tokens.

## Microscópio de parâmetros

O microscópio do checkpoint pequeno combina três níveis de evidência:

1. papel estrutural pelo nome, forma e posição do tensor;
2. proxy local `|peso × gradiente|` sobre a perda dos tokens da resposta;
3. pequena atenuação reversível dos grupos de maior influência, comparando a
   mudança da perda, a previsão de Taylor e restaurando o hash do checkpoint.

| Unidade observada | Responde sozinha? | Teste útil | Conclusão permitida |
| --- | --- | --- | --- |
| byte ou peso escalar | não | hash, distribuição e `gradiente × peso` | coordenada candidata |
| tensor | não | atenuação/ablação reversível | efeito local naquela métrica |
| cabeça, canal ou expert | não | *activation patching* no grafo | apoio/supressão contextual |
| prefixo de camadas + cabeça de saída | parcialmente | *logit lens* | tendência de tokens, não resposta independente |
| caminho completo paginado | sim | inferência token a token | resposta real do modelo |

Uma única retropropagação mede todos os grupos, evitando uma execução para cada
parâmetro. Ainda assim, o resultado é específico ao prompt, resposta e ponto de
operação. Simetrias, superposição e representações distribuídas impedem uma
ontologia confiável do tipo “parâmetro 123 significa adaptação”.

`gradiente × peso` é uma triagem local; tensores grandes tendem a dominar a soma
bruta e, por isso, o relatório também preserva a média por valor. Apenas a
intervenção reversível é causal para a métrica, prompt e intensidade escolhidos.
Uma versão posterior pode usar pares de prompts, *attribution patching* e
ablação exata de camadas, cabeças e canais, com separação entre descoberta e
confirmação. Mesmo assim, o rótulo correto será “apoiou esta métrica sob esta
intervenção”, nunca “contém este pensamento”.

## Compressão MoE sem professor

Métodos parameter-only como Joint Rank-k e MoLAE podem fatorar matrizes ou
compartilhar bases entre experts sem consultar logits do doador. Eles precisam
salvar fatores executáveis e fornecer um runtime que calcule esses fatores; um
*sketch* sozinho não é um modelo. Os resultados publicados são conservadores,
tipicamente dezenas de pontos percentuais de redução, não o colapso superior a
90% necessário para converter um frontier MoE em um modelo minúsculo de T4.

O caminho experimental aceito é validar primeiro um backend fatorado em um MoE
menor, medir erro de reconstrução e comportamento e somente então ampliar. Para
conhecimento verificável e atual, o DragonBRX pode combinar seu núcleo próprio
com memória Lira, navegador, ferramentas, planejamento, crítica e testes. Isso
pode superar um modelo maior em tarefas delimitadas, mas não constitui prova de
superioridade geral nem extração de cadeia de pensamento.

## Fontes primárias

- DARE: <https://arxiv.org/abs/2311.03099>
- Task Arithmetic: <https://openreview.net/forum?id=6t0Kwf8-jrj>
- TIES-Merging: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html>
- Git Re-Basin: <https://arxiv.org/abs/2209.04836>
- Model Stitching: <https://proceedings.neurips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html>
- DeepSeek-V4-Flash: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
- DeepSeek-V4 no Transformers: <https://huggingface.co/docs/transformers/model_doc/deepseek_v4>
- Formato Safetensors: <https://github.com/huggingface/safetensors/blob/main/README.md>
- Parsing de metadados por Range: <https://huggingface.co/docs/safetensors/metadata_parsing>
- Joint Rank-k Approximation: <https://arxiv.org/abs/2402.16319>
- MoLAE: <https://arxiv.org/abs/2503.23100>
- Randomized SVD: <https://arxiv.org/abs/0909.4061>
- SmolLM2 base: <https://huggingface.co/HuggingFaceTB/SmolLM2-360M>
- SmolLM2 Instruct: <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>
- SNIP / saliência `gradiente × peso`: <https://arxiv.org/abs/1810.02340>
- Attribution Patching: <https://arxiv.org/abs/2310.10348>
- AtP*: <https://arxiv.org/abs/2403.00745>
- Boas práticas de activation patching: <https://arxiv.org/abs/2309.16042>
- Limites do Colab: <https://research.google.com/colaboratory/faq.html>
