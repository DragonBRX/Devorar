# Modelos de terceiros

Este repositório não redistribui os checkpoints de terceiros. O script baixa
os modelos diretamente de suas páginas oficiais com revisões fixadas.

## SmolLM2-360M

- Autor/publicador: Hugging FaceTB
- Repositório: <https://huggingface.co/HuggingFaceTB/SmolLM2-360M>
- Revisão usada: `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`
- Licença declarada no cartão: Apache License 2.0

## SmolLM2-360M-Instruct

- Autor/publicador: Hugging FaceTB
- Repositório: <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>
- Revisão usada: `a10cc1512eabd3dde888204e902eca88bddb4951`
- Licença declarada no cartão: Apache License 2.0

## DeepSeek-V4-Flash — inspeção Frontier

- Autor/publicador: DeepSeek AI
- Repositório: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
- Revisão inspecionada: `60d8d70770c6776ff598c94bb586a859a38244f1`
- Licença declarada no cartão: MIT
- Uso nesta versão: `config.json`, índice, cabeçalhos, pequenas janelas e até 64
  linhas completas BF16 de tensores selecionados por HTTP Range; nenhum shard
  completo, byte bruto amostrado ou checkpoint derivado é redistribuído pelo
  relatório. O gate padrão executa 16 linhas de `head.weight` e persiste somente
  hashes, recibos e resultados numéricos.

Qualquer checkpoint produzido por assimilação permanece sujeito aos termos e
atribuições aplicáveis aos modelos de origem. Verifique novamente os cartões e
as licenças antes de distribuir um artefato gerado.
