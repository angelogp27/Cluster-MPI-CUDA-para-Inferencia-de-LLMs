# Clúster MPI + CUDA por Tailscale: empieza aquí

Vamos a unir de **2 a 4 computadoras con GPU NVIDIA**, cada una en su casa, para que **predigan juntas** la siguiente palabra de la frase *"La red distribuye el tráfico mediante …"*. El resultado que debe salir es **`paquetes` con 71.92 %**.

Todas las computadoras corren **Windows con WSL2 (Ubuntu 22.04)** y se conectan entre sí con **Tailscale**.

## 1. ¿Qué guía me toca?

| Si tú… | Tu rol | Tu guía |
|---|---|---|
| ejecutas el programa y proyectas la demo (rank 0) | **Lanzador** | [guia_lanzador.md](guia_lanzador.md) |
| prestas tu GPU al clúster | **Nodo** | [guia_nodo.md](guia_nodo.md) |

El lanzador también es un nodo (su GPU participa), así que **primero hace la guía del nodo** y después la suya.

## 2. Cómo funciona, en una figura

```
                  Tailscale (red privada, IPs 100.x.x.x)
                                  │
   ┌──────────────┬───────────────┼───────────────┬──────────────┐
┌──┴─────────┐ ┌──┴─────────┐ ┌───┴────────┐ ┌────┴───────┐
│ LANZADOR   │ │ NODO 2     │ │ NODO 3     │ │ NODO 4     │
│ rank 0     │ │ rank 1     │ │ rank 2     │ │ rank 3     │
│ GPU NVIDIA │ │ GPU NVIDIA │ │ GPU NVIDIA │ │ GPU NVIDIA │
│ cols 0-1   │ │ cols 2-3   │ │ cols 4-5   │ │ cols 6-7   │
└────────────┘ └────────────┘ └────────────┘ └────────────┘
   (ejemplo con 4 nodos: cada uno calcula 2 de las 8 columnas de W_up)
```

1. El lanzador ejecuta `mpirun`, que entra por **SSH** a los demás nodos y arranca **un proceso por computadora**.
2. Cada proceso calcula en su GPU la misma entrada (embedding, atención causal) y **solo su parte** de la capa Feed-Forward (`d_ff / nodos` columnas).
3. Con `MPI_Allgather`, los nodos se intercambian sus pedazos **por Tailscale** y cada uno reconstruye la matriz completa.
4. El rank 0 imprime las matrices y la predicción.

A diferencia de un esquema maestro-trabajador, **aquí todos los nodos se hablan entre sí**. Por eso todos deben estar en la **misma red Tailscale** (ver el paso 2 de la guía del nodo).

## 3. La regla del número de nodos

El programa exige que el número de procesos **divida a `d_ff = 8`**: solo sirven **1, 2, 4 u 8**. Con un máximo de 4 computadoras:

| Nodos conectados | Se usan (`-np`) |
|---|---|
| 1 | 1 |
| 2 | 2 |
| **3** | **2** (el tercero queda de reserva) |
| 4 | 4 |

`lanzar.sh` lo decide solo, tomando las primeras líneas de `hosts.txt`. El resultado es **idéntico** con 1, 2, 4 u 8 procesos (validado en CPU, ver el manual, sección 6).

## 4. Palabras que vas a ver

| Palabra | Qué significa |
|---|---|
| **Rank** | Número de cada proceso MPI (0, 1, 2, 3). El rank 0 imprime los resultados |
| **Lanzador** | La PC que ejecuta `mpirun`. Hay una sola |
| **Nodo** | Cada PC con GPU que aporta su cálculo |
| **Hostfile** (`hosts.txt`) | Lista de los nodos que participan |
| **WSL2** | Ubuntu que corre dentro de Windows. Todos los comandos de estas guías se pegan ahí, no en PowerShell |
| **Tailscale** | Programa gratuito que conecta las PCs como si estuvieran en la misma red |
| **Huella** | Los primeros 12 caracteres del SHA-256 del archivo `.cu`. Si la tuya no coincide, tienes otra versión del código |
| **`sm_XX`** | La arquitectura de tu GPU (la detecta `compilar.sh` sola) |

## 5. Calendario

| Momento | Lanzador | Cada nodo |
|---|---|---|
| **1. Preparación** (cada uno en su casa, 1-2 h) | Hace la guía del nodo y prepara `hosts.txt` y el SSH | Hace la guía del nodo y envía sus datos al grupo |
| **2. Ensayo individual** (5 min) | Corre `-np 1` y `-np 2` en su sola PC | Corre `-np 1` en su PC |
| **3. Ensayo general** (videollamada, 30 min) | `lanzar.sh --check` y luego `lanzar.sh` con todos | Mantiene su terminal WSL abierta |
| **4. Presentación** | Comparte pantalla y lanza | Mantiene su terminal WSL abierta |

## 6. Inventario del grupo

Cada uno completa su fila con la línea que imprime `nodo.sh`.

| Integrante | Rol | Host (WSL) | Usuario (WSL) | IP de Tailscale | GPU | Huella del `.cu` | Ensayo individual |
|---|---|---|---|---|---|---|---|
| | Lanzador | | | | | `7d86f3647062` | |
| | Nodo | | | | | `7d86f3647062` | |
| | Nodo | | | | | `7d86f3647062` | |
| | Nodo | | | | | `7d86f3647062` | |

## 7. Archivos

El grupo comparte una carpeta con estos archivos del proyecto:

| Archivo | Para qué |
|---|---|
| `llm_mpi_cuda_red_trafico.cu` | El programa (huella `7d86f3647062`) |
| `scripts/compilar.sh` | Compila en tu nodo, detectando solo la arquitectura de tu GPU |
| `scripts/nodo.sh` | Revisa que tu nodo esté listo |
| `scripts/lanzar.sh` | Solo el lanzador: comprueba y ejecuta |
| `scripts/hosts.ejemplo.txt` | Solo el lanzador: plantilla de `hosts.txt` |

Todos los archivos van en **`/opt/llm_cluster`**, la misma ruta absoluta en todas las computadoras (los usuarios de WSL pueden llamarse distinto, por eso no se usa `~`).

## 8. Para quien quiera más detalle

El material inicial del profesor (manual técnico `.docx`, presentación `.pptx` y la validación en CPU `validate_cpu_mpi.cpp.txt`) vive en `.context/`, que no se sube al repositorio. Pídelo al lanzador si lo necesitas.

> **Estado:** los scripts y la selección de nodos se probaron localmente; la ejecución real sobre WSL2 + Tailscale + GPU NVIDIA **aún no se ha probado** y se confirma en el ensayo individual y en el general.
