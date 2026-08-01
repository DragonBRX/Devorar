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

A implementação atual (V2) cobre somente a Classe A.

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

## Fontes primárias

- DARE: <https://arxiv.org/abs/2311.03099>
- Task Arithmetic: <https://openreview.net/forum?id=6t0Kwf8-jrj>
- TIES-Merging: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html>
- Git Re-Basin: <https://arxiv.org/abs/2209.04836>
- Model Stitching: <https://proceedings.neurips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html>
- SmolLM2 base: <https://huggingface.co/HuggingFaceTB/SmolLM2-360M>
- SmolLM2 Instruct: <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>
- Limites do Colab: <https://research.google.com/colaboratory/faq.html>
