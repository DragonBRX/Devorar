# Primeiro piloto real no Colab

Este registro resume o primeiro artefato completo produzido pelo Devorar em
uma sessão limpa do Google Colab com uma Tesla T4. Os números abaixo vieram do
manifesto, do `audit.json` e dos arquivos baixados ao fim da execução. Eles
descrevem somente esse build; não são uma alegação geral de capacidade.

## Identidade do build

| Campo | Valor |
| --- | --- |
| Build ID | `db84f05d-5442-4650-a884-7d47d77c505e` |
| Hospedeiro | `HuggingFaceTB/SmolLM2-360M@f8027fd0eaeea54caa13c31d31b9fdc459c38b49` |
| Doador | `HuggingFaceTB/SmolLM2-360M-Instruct@a10cc1512eabd3dde888204e902eca88bddb4951` |
| Receita | `alpha=0.75`, `drop_rate=0.50`, `seed=24051996` |
| Estado da base | `e7a223e6635550f4a67d909bed83db3ba8aa2c345b7af429f6ad833615dc762f` |
| Estado do doador | `4a91be354f32d6b555d9649bcdfbec4e3603bcfec42b85570c1ff543e40c19cc` |
| Estado assimilado | `9c0a98b386749631ccb9957ec17ecf1b5ab77dc09e6a92143fb9dd9bcfed1a97` |
| Arquivo `model.safetensors` | `39c210db01412682f22b1607e140ce461128fe26463d90cf467aff2ab9ccd1f3` |

O estado final tem hash diferente tanto da base quanto do doador e foi
recarregado como checkpoint autônomo. O doador registrou zero chamadas de
`forward` durante a construção.

## O que mudou

- 290 tensores físicos e 291 nomes lógicos foram auditados;
- 361.821.120 parâmetros físicos foram processados;
- 176.212.442 valores mudaram, aproximadamente 48,70% do total;
- a fração DARE ponderada mantida foi `0.4999950775`;
- não houve `NaN` nem infinito;
- o grupo amarrado `lm_head.weight` / `model.embed_tokens.weight` permaneceu
  válido;
- todos os tensores salvos estavam em FP16.

O modelo é, portanto, um checkpoint parametricamente novo e derivado. Isso não
significa que cada valor seja novo, nem que uma cadeia de pensamento privada do
doador tenha sido copiada ou decodificada.

## Avaliação observada

| Métrica | Resultado |
| --- | ---: |
| Delta das quatro regras determinísticas | `+0.50` |
| Delta de NLL, candidato menos base | `+0.1464908123` |
| Razão de perplexidade, candidato/base | `1.1577642936` |
| Degradação relativa nesse corpus pequeno | aproximadamente `15,78%` |
| Gates do piloto | aprovados |

O ganho nas regras é evidência inicial de transferência de comportamento de
instrução. O corpus de retenção piorou, embora tenha permanecido dentro do
limite original de `1.50`. Quatro regras e um corpus pequeno não demonstram
raciocínio geral.

## Mudanças aplicadas na V2

- limite padrão de perplexidade reduzido de `1.50` para `1.25`;
- piso absoluto de `0.50` para o escore determinístico do candidato;
- escores absolutos da base e do candidato registrados no manifesto;
- mesma mensagem de sistema neutra usada ao avaliar base e candidato;
- dtype do `config.json` alinhado ao dtype real dos pesos;
- caminho interno do cache removido da identidade persistida;
- identidade DragonBRX declarada como camada de apresentação, separada da
  transformação paramétrica;
- segundo executor verifica artefatos e o hash do estado antes da inferência.

Esses ajustes preservam a receita que funcionou no piloto e tornam o próximo
teste mais rigoroso e mais fácil de reproduzir.
