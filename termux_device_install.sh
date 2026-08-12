#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$HOME/.config/devorar/worker.env"
PREVIOUS_DEVICE_ID=""

if [ -f "$CONFIG_FILE" ]; then
    PREVIOUS_DEVICE_ID="$(bash -c '. "$1"; printf "%s" "${DEVORAR_DEVICE_ID:-}"' _ "$CONFIG_FILE")"
fi

: "${DEVORAR_SERVER:?DEVORAR_SERVER ausente}"
: "${DEVORAR_CLUSTER_TOKEN:?DEVORAR_CLUSTER_TOKEN ausente}"
DEVORAR_PROCESSES="${DEVORAR_PROCESSES:-1}"

cd "$SCRIPT_DIR"
chmod +x termux_install.sh
./termux_install.sh

BIN_FILE="$PREFIX/bin/devorar-worker"
LOG_DIR="$HOME/.local/state/devorar"
DEVICE_ID=""
if [ -f "$CONFIG_FILE" ]; then
    DEVICE_ID="$(bash -c '. "$1"; printf "%s" "${DEVORAR_DEVICE_ID:-}"' _ "$CONFIG_FILE")"
fi
if [ -z "$DEVICE_ID" ] && [ -n "$PREVIOUS_DEVICE_ID" ]; then
    DEVICE_ID="$PREVIOUS_DEVICE_ID"
fi
if [ -z "$DEVICE_ID" ]; then
    DEVICE_ID="$(python - <<'PY'
import uuid
print(uuid.uuid4().hex)
PY
)"
fi
printf 'DEVORAR_DEVICE_ID=%q\n' "$DEVICE_ID" >> "$CONFIG_FILE"
chmod 600 "$CONFIG_FILE"

cat > "$BIN_FILE" <<'WORKER'
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
CONFIG_FILE="$HOME/.config/devorar/worker.env"
SESSION="devorar-worker"
LOG_DIR="$HOME/.local/state/devorar"
LOG_FILE="$LOG_DIR/worker.log"
ACTION="${1:-status}"
mkdir -p "$LOG_DIR"
[ -f "$CONFIG_FILE" ] || { echo "Configuração ausente: $CONFIG_FILE" >&2; exit 2; }
set -a
. "$CONFIG_FILE"
set +a
hardware() {
    cd "$DEVORAR_HOME"
    python -m src.system_info | sed -n '1p'
}
case "$ACTION" in
    start)
        tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" "$PREFIX/bin/devorar-worker foreground"
        echo "Worker ativo: $DEVORAR_WORKER_NAME -> PC Devorar descoberto automaticamente"
        hardware
        ;;
    stop)
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        echo "Worker parado."
        hardware
        ;;
    restart)
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        tmux new-session -d -s "$SESSION" "$PREFIX/bin/devorar-worker foreground"
        echo "Worker reiniciado: $DEVORAR_WORKER_NAME"
        hardware
        ;;
    reinstall)
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        cd "$HOME"
        rm -rf "$HOME/Devorar"
        git clone --depth 1 https://github.com/DragonBRX/Devorar.git "$HOME/Devorar"
        cd "$HOME/Devorar"
        chmod +x termux_auto_install.sh
        DEVORAR_WORKER_NAME="$DEVORAR_WORKER_NAME" ./termux_auto_install.sh
        ;;
    status)
        if tmux has-session -t "$SESSION" 2>/dev/null; then STATE="ativo"; else STATE="parado"; fi
        echo "Worker $STATE: $DEVORAR_WORKER_NAME"
        hardware
        ;;
    hardware|info)
        hardware
        ;;
    logs)
        touch "$LOG_FILE"
        tail -n 100 -f "$LOG_FILE"
        ;;
    foreground)
        cd "$DEVORAR_HOME"
        trap 'exit 0' INT TERM
        while true; do
            printf '[%s] procurando/conectando ao PC Devorar\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
            set +e
            python device_worker.py --server "$DEVORAR_SERVER" --name "$DEVORAR_WORKER_NAME" --processes "$DEVORAR_PROCESSES" >> "$LOG_FILE" 2>&1
            CODE=$?
            set -e
            [ "$CODE" -eq 130 ] && exit 0
            printf '[%s] worker encerrou com código %s; nova tentativa em 5s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CODE" >> "$LOG_FILE"
            sleep 5
        done
        ;;
    *)
        echo "Uso: devorar-worker {start|stop|restart|reinstall|status|hardware|logs|foreground}" >&2
        exit 2
        ;;
esac
WORKER
chmod 700 "$BIN_FILE"
"$BIN_FILE" restart
"$BIN_FILE" status
