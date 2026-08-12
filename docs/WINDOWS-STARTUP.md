# Inicialização supervisionada no Windows

O `windows_start.ps1` não chama mais o servidor Python de forma cega e indefinida.

Fluxo atual:

1. configura o Firewall para a rede local;
2. testa `py -3` e `python` com timeout de 5 segundos e usa somente um executável que realmente responde;
3. inicia `windows_server.py` como processo supervisionado com saída e erro gravados em `cluster-state/runtime/`;
4. testa `http://127.0.0.1:8765/health` durante até 20 segundos;
5. se responder, mostra `DEVORAR ONLINE` e `Teste de saúde: OK`;
6. se não responder, encerra o processo preso, mostra o fim dos logs e informa os caminhos dos arquivos completos.

Assim, a tela não pode ficar indefinidamente em uma mensagem genérica de inicialização sem explicar o estado real.

Logs:

```text
cluster-state/runtime/windows-server.stdout.log
cluster-state/runtime/windows-server.stderr.log
```

O servidor continua exibindo os eventos do cluster no PowerShell depois que o health check confirma que ele está online.
