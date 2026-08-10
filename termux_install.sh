#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

if ! command -v pkg >/dev/null 2>&1; then
    echo "ERROR: este instalador deve ser executado dentro do Termux." >&2
    exit 2
fi

pkg update -y
pkg install -y python git tmux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$HOME/.config/devorar"
CONFIG_FILE="$CONFIG_DIR/worker.env"
BIN_FILE="$PREFIX/bin/devorar-worker"
LOG_DIR="$HOME/.local/state/devorar"
SERVER="${DEVORAR_SERVER:-}"
TOKEN="${DEVORAR_CLUSTER_TOKEN:-}"
PROCESSES="${DEVORAR_PROCESSES:-1}"
WORKER_NAME="${DEVORAR_WORKER_NAME:-}"

case "$PROCESSES" in
    ''|*[!0-9]*) echo "ERROR: DEVORAR_PROCESSES deve ser um número entre 1 e 16." >&2; exit 2 ;;
esac
if [ "$PROCESSES" -lt 1 ] || [ "$PROCESSES" -gt 16 ]; then
    echo "ERROR: DEVORAR_PROCESSES deve estar entre 1 e 16." >&2
    exit 2
fi

mkdir -p "$CONFIG_DIR" "$LOG_DIR"
chmod 700 "$CONFIG_DIR" "$LOG_DIR"

if [ -z "$WORKER_NAME" ] && [ -f "$CONFIG_FILE" ]; then
    WORKER_NAME="$(bash -c '. "$1"; printf "%s" "${DEVORAR_WORKER_NAME:-}"' _ "$CONFIG_FILE")"
fi

if [ -z "$WORKER_NAME" ]; then
    WORKER_NAME="termux-$(python - <<'PY'
import uuid
print(uuid.uuid4().hex[:8])
PY
)"
fi

if [ -n "$SERVER" ] || [ -n "$TOKEN" ]; then
    if [ -z "$SERVER" ] || [ -z "$TOKEN" ]; then
        echo "ERROR: DEVORAR_SERVER e DEVORAR_CLUSTER_TOKEN devem ser fornecidos juntos." >&2
        exit 2
    fi
    if [ "${#TOKEN}" -lt 24 ]; then
        echo "ERROR: DEVORAR_CLUSTER_TOKEN inválido ou curto demais." >&2
        exit 2
    fi
    {
        printf 'DEVORAR_HOME=%q\n' "$SCRIPT_DIR"
        printf 'DEVORAR_SERVER=%q\n' "$SERVER"
        printf 'DEVORAR_CLUSTER_TOKEN=%q\n' "$TOKEN"
        printf 'DEVORAR_PROCESSES=%q\n' "$PROCESSES"
        printf 'DEVORAR_WORKER_NAME=%q\n' "$WORKER_NAME"
    } > "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
fi

cat > "$BIN_FILE" <<'SH2'
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
CONFIG_FILE="$HOME/.config/devorar/worker.env"
SESSION="devorar-worker"
LOG_DIR="$HOME/.local/state/devorar"
LOG_FILE="$LOG_DIR/worker.log"
ACTION="${1:-status}"
mkdir -p "$LOG_DIR"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuração ausente: $CONFIG_FILE" >&2
    echo "Execute novamente termux_install.sh usando o bloco gerado pelo PC." >&2
    exit 2
fi
set -a
. "$CONFIG_FILE"
set +a
case "$ACTION" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "Worker já está ativo."
            exit 0
        fi
        tmux new-session -d -s "$SESSION" "$PREFIX/bin/devorar-worker foreground"
        echo "Worker iniciado em segundo plano."
        echo "Logs: devorar-worker logs"
        ;;
    stop)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            tmux kill-session -t "$SESSION"
            echo "Worker parado."
        else
            echo "Worker já está parado."
        fi
        ;;
    restart)
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        tmux new-session -d -s "$SESSION" "$PREFIX/bin/devorar-worker foreground"
        echo "Worker reiniciado."
        ;;
    status)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "Worker ativo: $DEVORAR_WORKER_NAME -> $DEVORAR_SERVER"
        else
            echo "Worker parado: $DEVORAR_WORKER_NAME -> $DEVORAR_SERVER"
        fi
        ;;
    logs)
        touch "$LOG_FILE"
        tail -n 100 -f "$LOG_FILE"
        ;;
    foreground)
        cd "$DEVORAR_HOME"
        trap 'exit 0' INT TERM
        while true; do
            printf '[%s] iniciando/conectando %s -> %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$DEVORAR_WORKER_NAME" "$DEVORAR_SERVER" >> "$LOG_FILE"
            set +e
            python distributed_worker.py \
                --server "$DEVORAR_SERVER" \
                --name "$DEVORAR_WORKER_NAME" \
                --processes "$DEVORAR_PROCESSES" >> "$LOG_FILE" 2>&1
            CODE=$?
            set -e
            if [ "$CODE" -eq 130 ]; then
                exit 0
            fi
            printf '[%s] worker encerrou com código %s; nova tentativa em 5s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CODE" >> "$LOG_FILE"
            sleep 5
        done
        ;;
    *)
        echo "Uso: devorar-worker {start|stop|restart|status|logs|foreground}" >&2
        exit 2
        ;;
esac
SH2
chmod 700 "$BIN_FILE"

if [ -f "$CONFIG_FILE" ]; then
    "$BIN_FILE" restart
    "$BIN_FILE" status
else
    echo "Devorar instalado em $SCRIPT_DIR."
    echo "Falta somente a configuração do PC; use o bloco que o coordenador imprime ao iniciar."
fi
