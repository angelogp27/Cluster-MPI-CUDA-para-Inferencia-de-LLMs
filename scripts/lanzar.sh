#!/usr/bin/env bash
# Lo ejecuta SOLO el lanzador (rank 0). Lee hosts.txt, elige cuantos nodos
# usar y lanza mpirun.
#   bash /opt/llm_cluster/lanzar.sh            # ejecuta
#   bash /opt/llm_cluster/lanzar.sh --check    # solo comprueba SSH/GPU/binario
#   DRY_RUN=1 bash /opt/llm_cluster/lanzar.sh  # muestra el comando sin ejecutarlo
#
# Regla del programa: -np debe dividir a d_ff=8 (1, 2, 4 u 8). El cluster
# admite hasta 4 nodos, asi que -np es el mayor valor de {4,2,1} <= nodos
# disponibles (con 3 nodos conectados se usan 2).
set -euo pipefail
DIR=${LLM_DIR:-/opt/llm_cluster}
HOSTS=${HOSTS:-$DIR/hosts.txt}
BIN=$DIR/llm_mpi_cuda_red_trafico
IFACE=${IFACE:-tailscale0}
MAX_NODOS=4

[ -f "$HOSTS" ] || { echo "falta $HOSTS (copia hosts.ejemplo.txt a hosts.txt y editalo)"; exit 1; }
mapfile -t LINEAS < <(grep -vE '^\s*(#|$)' "$HOSTS")
DISP=${#LINEAS[@]}
[ "$DISP" -ge 1 ] || { echo "$HOSTS no lista ningun nodo"; exit 1; }
[ "$DISP" -le "$MAX_NODOS" ] || { echo "AVISO: hay $DISP nodos; se consideran solo los primeros $MAX_NODOS"; DISP=$MAX_NODOS; }

NP=1
for v in 4 2 1; do [ "$DISP" -ge "$v" ] && { NP=$v; break; }; done
[ "$NP" -eq "$DISP" ] || echo "AVISO: $DISP nodos disponibles; el programa solo admite -np en {1,2,4,8}, se usan $NP"

USADO=$(mktemp /tmp/hosts_usado.XXXXXX)
trap 'rm -f "$USADO" "${SALIDA:-}"' EXIT
printf '%s\n' "${LINEAS[@]:0:$NP}" > "$USADO"
echo "== nodos que participan (-np $NP):"; sed 's/^/   /' "$USADO"

PRIMERO=$(awk '{print $1}' "$USADO" | head -1)
case "$PRIMERO" in
  "$(hostname)"|localhost|127.*|"$(tailscale ip -4 2>/dev/null | head -1)") ;;
  *) echo "AVISO: la primera linea ($PRIMERO) deberia ser ESTA maquina (hostname: $(hostname))" ;;
esac

echo "== comprobando cada nodo"
FALLOS=0
while read -r h _; do
  [ "$h" = "$PRIMERO" ] && { test -x "$BIN" && echo "   $h (local): OK" || { echo "   $h (local): falta el binario"; FALLOS=$((FALLOS+1)); }; continue; }
  if SAL=$(ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new "$h" \
        "hostname; nvidia-smi --query-gpu=name --format=csv,noheader | head -1; test -x $BIN && echo binario-ok" 2>&1); then
    echo "   $h: $(echo "$SAL" | tr '\n' ' | ')"
    echo "$SAL" | grep -q binario-ok || { echo "     ^ falta el binario en $h"; FALLOS=$((FALLOS+1)); }
  else
    echo "   $h: NO RESPONDE ($SAL)"; FALLOS=$((FALLOS+1))
  fi
done < "$USADO"
[ "$FALLOS" -eq 0 ] || { echo "Hay $FALLOS problema(s); corrigelos antes de lanzar."; exit 1; }
[ "${1:-}" = "--check" ] && { echo "Todo OK."; exit 0; }

CMD=(mpirun --hostfile "$USADO" -np "$NP" --map-by node
     --mca btl_tcp_if_include "$IFACE" --mca oob_tcp_if_include "$IFACE"
     -x PATH -x LD_LIBRARY_PATH "$BIN")
echo "== ${CMD[*]}"
[ -n "${DRY_RUN:-}" ] && exit 0

SALIDA=$(mktemp /tmp/salida_llm.XXXXXX)
"${CMD[@]}" 2>&1 | tee "$SALIDA"
echo
if grep -q 'paquetes' "$SALIDA" && grep -qE '71[.,]92' "$SALIDA"; then
  echo "VERIFICADO: paquetes con 71.92 % (resultado de referencia)."
else
  echo "ATENCION: no aparece 'paquetes' con 71.92 %; revisa la salida antes de dar la ejecucion por valida."
  exit 1
fi
