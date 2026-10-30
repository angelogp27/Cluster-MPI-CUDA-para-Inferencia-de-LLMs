#!/usr/bin/env bash
# Revisa que ESTE nodo (WSL2 + Ubuntu) este listo para el cluster y muestra
# los datos que hay que enviar al grupo:
#   bash /opt/llm_cluster/nodo.sh
DIR=${LLM_DIR:-/opt/llm_cluster}
FUENTE=llm_mpi_cuda_red_trafico
ok()   { printf '  [ OK ] %s\n' "$1"; }
falta(){ printf '  [FALTA] %s\n' "$1"; FALLOS=$((FALLOS+1)); }
FALLOS=0

echo "== 1/6 GPU"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  ok "$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1)"
else falta "nvidia-smi no ve la GPU (driver NVIDIA en Windows; NO instalar driver dentro de WSL)"; fi

echo "== 2/6 Tailscale"
IP=$(tailscale ip -4 2>/dev/null | head -1)
if [ -n "$IP" ]; then ok "IP de Tailscale: $IP"; else falta "Tailscale sin IP (sudo tailscale up)"; fi
if ip link show tailscale0 >/dev/null 2>&1; then ok "interfaz tailscale0 presente"
else falta "no existe la interfaz tailscale0 (¿systemd activo en WSL2? ver guia, paso 2)"; fi

echo "== 3/6 SSH"
if (systemctl is-active --quiet ssh 2>/dev/null || pgrep -x sshd >/dev/null); then ok "servidor SSH activo"
else falta "sshd no esta corriendo (sudo systemctl enable --now ssh)"; fi

echo "== 4/6 CUDA y OpenMPI"
if command -v nvcc >/dev/null; then ok "$(nvcc --version | tail -1)"; else falta "nvcc no esta en el PATH"; fi
if command -v mpirun >/dev/null; then ok "$(mpirun --version | head -1)"; else falta "OpenMPI no instalado"; fi

echo "== 5/6 Programa"
if [ -f "$DIR/$FUENTE.cu" ]; then ok "fuente $DIR/$FUENTE.cu, huella $(sha256sum "$DIR/$FUENTE.cu" | cut -c1-12)"
else falta "falta $DIR/$FUENTE.cu"; fi
if [ -x "$DIR/$FUENTE" ]; then ok "binario compilado"; else falta "falta compilar (bash $DIR/compilar.sh)"; fi

echo "== 6/6 Nombre del nodo"
ok "hostname: $(hostname)   usuario: $(whoami)"

echo
if [ "$FALLOS" -eq 0 ]; then echo "TODO LISTO. Envia al grupo esta linea:"; else echo "Hay $FALLOS punto(s) por resolver. Cuando todo este OK, envia:"; fi
echo "  host=$(hostname) usuario=$(whoami) ip=${IP:-?} gpu=\"$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)\" huella=$(sha256sum "$DIR/$FUENTE.cu" 2>/dev/null | cut -c1-12)"
exit $((FALLOS>0))
