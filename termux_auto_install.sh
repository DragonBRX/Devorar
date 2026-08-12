#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

if ! command -v pkg >/dev/null 2>&1; then
    echo "ERROR: execute este instalador dentro do Termux." >&2
    exit 2
fi

printf '\n========================================\n'
printf '        DEVORAR TERMUX\n'
printf '========================================\n'
printf 'Preparando o celular...\n\n'

pkg update -y
pkg install -y python git tmux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$HOME/.cache/devorar"
DISCOVERY_JSON="$CACHE_DIR/discovery.json"
mkdir -p "$CACHE_DIR"
chmod 700 "$CACHE_DIR"

cd "$SCRIPT_DIR"
printf '\n[1/3] Procurando um PC Devorar na mesma Wi-Fi...\n'
python lan_discover.py --json-output "$DISCOVERY_JSON"

SERVER="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["server_url"])' "$DISCOVERY_JSON")"
TOKEN="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["cluster_token"])' "$DISCOVERY_JSON")"
PROCESSES="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["worker_processes"])' "$DISCOVERY_JSON")"
PC_NAME="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["pc_name"])' "$DISCOVERY_JSON")"

printf '[2/3] PC encontrado: %s\n' "$PC_NAME"
printf '[2/3] Configuração recebida automaticamente.\n'
printf '[3/3] Iniciando worker...\n'

chmod +x termux_device_install.sh
DEVORAR_SERVER="$SERVER" \
DEVORAR_CLUSTER_TOKEN="$TOKEN" \
DEVORAR_PROCESSES="$PROCESSES" \
./termux_device_install.sh

printf '\n========================================\n'
printf '       DEVORAR TERMUX PRONTO\n'
printf '========================================\n'
printf 'PC: %s\n' "$PC_NAME"
printf 'Estado: WORKER INICIADO\n'
printf 'Nenhum IP, porta ou token precisou ser digitado.\n'
printf 'O celular agora aguarda trabalhos do PC.\n'
printf '========================================\n\n'
