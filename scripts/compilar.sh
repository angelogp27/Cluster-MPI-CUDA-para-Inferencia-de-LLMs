#!/usr/bin/env bash
# Compila el programa en ESTE nodo (WSL2 + Ubuntu) para la GPU de este nodo.
# Cada nodo compila su propia copia, en la misma ruta (/opt/llm_cluster):
#   bash /opt/llm_cluster/compilar.sh
set -euo pipefail
DIR=${LLM_DIR:-/opt/llm_cluster}
FUENTE=llm_mpi_cuda_red_trafico
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
cd "$DIR"

[ -f "$FUENTE.cu" ] || { echo "falta $DIR/$FUENTE.cu: copialo de la carpeta compartida del grupo"; exit 1; }
for cmd in nvidia-smi nvcc mpicxx; do
  command -v "$cmd" >/dev/null || { echo "falta '$cmd' (revisa la guia del nodo, pasos de CUDA y OpenMPI)"; exit 1; }
done

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
[ -n "$CAP" ] || { echo "nvidia-smi no devolvio la capacidad de computo; ¿se ve la GPU en WSL2?"; exit 1; }
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  (sm_$CAP)"
echo "Huella del .cu: $(sha256sum $FUENTE.cu | cut -c1-12)"

nvcc -O3 -arch="sm_$CAP" -ccbin=mpicxx -c "$FUENTE.cu" -o "$FUENTE.o"
mpicxx "$FUENTE.o" -L"$CUDA_HOME/lib64" -lcudart -o "$FUENTE"
echo "OK: compilado $DIR/$FUENTE"
