# Guía del lanzador (quien ejecuta `mpirun` y proyecta la demo)

**Esta guía es solo para quien ejecuta el programa**: la PC que es el **rank 0**, arranca los procesos de los demás por SSH, imprime los resultados y comparte pantalla. Los demás integrantes siguen [guia_nodo.md](guia_nodo.md).

Como lanzador tienes dos tareas:

1. **Preparar** la red y el acceso para que los nodos puedan seguir su guía: Tailscale del grupo, tu llave SSH, la carpeta compartida y el `hosts.txt`.
2. **Conducir** el ensayo general y la demo.

Tiempo: unos 30 minutos de preparación (una vez) y 30 minutos de ensayo general.

## Antes de empezar

- [ ] **Haz primero la [guía del nodo](guia_nodo.md), pasos 1 a 7.** Tu GPU también es parte del clúster y necesitas el programa compilado y probado en tu PC.
- [ ] Una carpeta de Google Drive del grupo.
- [ ] Ideal: PC al cargador y conectada por cable.

---

## Paso 1. Compartir los archivos

**Qué vas a hacer:** dejar en Drive los archivos que todos los nodos van a descargar, para que todos tengan **exactamente la misma versión** del código.

Sube a la carpeta compartida: `llm_mpi_cuda_red_trafico.cu`, `compilar.sh` y `nodo.sh`.

Comprueba la huella y publícala en el grupo:

```bash
sha256sum /opt/llm_cluster/llm_mpi_cuda_red_trafico.cu | cut -c1-12
```

**Debes ver** `7d86f3647062`. Si cambias el `.cu`, la huella cambia: avisa a todos que lo vuelvan a descargar y a compilar.

- [ ] Archivos en Drive, enlace y huella publicados.

---

## Paso 2. Tailscale: la red del grupo

**Qué vas a hacer:** conectar a todos en **una misma red**. En un clúster MPI los nodos se hablan entre sí, por eso **no basta con compartir solo tu PC**: todos tienen que verse con todos.

La forma más simple: **una sola cuenta de Tailscale para el grupo** (por ejemplo, una cuenta de Google creada para el proyecto) con la que cada nodo inicia sesión en su Ubuntu. Todas las PCs quedan en la misma red y no hay límite de usuarios que cuidar.

> Alternativa: invitar a cada integrante con su propia cuenta desde https://login.tailscale.com/admin/users. Antes, revisa el límite de usuarios del plan gratuito en https://tailscale.com/pricing: con 4 personas puede quedarse corto.

1. En tu Ubuntu, tras el paso 2 de la guía del nodo, anota tu IP: `tailscale ip -4`.
2. En https://login.tailscale.com/admin/machines, para **cada** PC: menú **`…`** → **Disable key expiry** (si no, una PC se desconecta sola antes de la demo).
3. Envía al grupo las instrucciones de acceso a la cuenta (por un canal privado, **no** en un chat público).
4. Cuando todos se conecten, `tailscale status` debe listar todas las PCs.

- [ ] Todas las PCs aparecen en `tailscale status`.

---

## Paso 3. Crear tu llave SSH y repartir la pública

**Qué vas a hacer:** que tu PC pueda entrar por SSH a las demás sin contraseña. `mpirun` lo necesita para arrancar los procesos remotos.

```bash
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

**Debes ver** una línea que empieza con `ssh-ed25519 ...`. **Copia esa línea completa** y envíala al grupo (es la llave **pública**: no es secreta). Cada nodo la pega en su `authorized_keys` (paso 8 de su guía).

> Nunca envíes el archivo `~/.ssh/id_ed25519` (sin `.pub`): esa es la llave privada.

- [ ] Llave pública enviada.

---

## Paso 4. Alias SSH de cada nodo

**Qué vas a hacer:** darle a cada nodo un nombre corto con su IP y su usuario. Los usuarios de WSL suelen llamarse distinto en cada PC.

Cuando tengas las líneas `host=... usuario=... ip=...` de todos (paso 9 de la guía del nodo), edita:

```bash
nano ~/.ssh/config
```

Una entrada por nodo (cambia nombre, IP y usuario por los que enviaron):

```
Host nodo2
    HostName 100.101.102.2
    User usuario_del_nodo2
    StrictHostKeyChecking accept-new

Host nodo3
    HostName 100.101.102.3
    User usuario_del_nodo3
    StrictHostKeyChecking accept-new

Host nodo4
    HostName 100.101.102.4
    User usuario_del_nodo4
    StrictHostKeyChecking accept-new
```

Prueba cada nodo (debe entrar **sin** pedir contraseña y mostrar su nombre):

```bash
ssh nodo2 hostname
ssh nodo3 hostname
ssh nodo4 hostname
```

- [ ] Entro por SSH a cada nodo sin contraseña.

---

## Paso 5. Crear `hosts.txt`

```bash
cp scripts/hosts.ejemplo.txt /opt/llm_cluster/hosts.txt   # o copia el archivo desde donde lo tengas
cp scripts/lanzar.sh /opt/llm_cluster/
nano /opt/llm_cluster/hosts.txt
```

Deja **la primera línea con el nombre de tu propia PC** (`hostname`) y las demás con los alias del paso 4. Ejemplo para 4 nodos (borra las líneas que no uses):

```
miequipo slots=1
nodo2 slots=1
nodo3 slots=1
nodo4 slots=1
```

`slots=1` significa un proceso MPI por PC (una GPU por PC). No pongas más de 4 líneas.

- [ ] `hosts.txt` listo.

---

## Paso 6. Comprobar el clúster (sin ejecutar)

```bash
bash /opt/llm_cluster/lanzar.sh --check
```

El script lee `hosts.txt`, decide cuántos nodos usar (ver la regla en el [README](README.md), sección 3) y entra por SSH a cada uno para ver que responde, que tiene GPU y que tiene el binario.

**Debes ver**, por cada nodo, su nombre y el modelo de su GPU, y al final `Todo OK.`

- Si dice **`NO RESPONDE`**: ese nodo tiene Ubuntu cerrado, SSH apagado, o tu llave no está en su `authorized_keys`.
- Si dice **`falta el binario`**: ese nodo no ejecutó `compilar.sh`.

- [ ] `--check` termina con `Todo OK.`

---

## Paso 7. Ejecutar

```bash
bash /opt/llm_cluster/lanzar.sh
```

Internamente ejecuta:

```
mpirun --hostfile <nodos usados> -np N --map-by node \
       --mca btl_tcp_if_include tailscale0 --mca oob_tcp_if_include tailscale0 \
       -x PATH -x LD_LIBRARY_PATH /opt/llm_cluster/llm_mpi_cuda_red_trafico
```

Las opciones `--mca ...tailscale0` obligan a OpenMPI a hablar **por Tailscale**. Sin ellas intenta salir por otra interfaz y se queda colgado.

**Debes ver:**

1. La cabecera con `Procesos MPI (nodos del cluster): N`.
2. **Una línea `[Rank i/N] nodo=... GPU_asignada=0` por cada PC, con nombres distintos.** Eso prueba que se usaron máquinas distintas y no varios procesos en la tuya.
3. Las matrices intermedias (rank 0).
4. La predicción: **`paquetes` con 71.92 %.**
5. Al final de todo, `VERIFICADO: paquetes con 71.92 % ...`.

El resultado debe ser el mismo con 2 o con 4 nodos.

- [ ] Resultado correcto con todos los nodos.

---

## Guion corto para la demo

1. Muestra `tailscale status` (las PCs conectadas).
2. Muestra `hosts.txt` y explica "un proceso por PC, una GPU por PC".
3. Ejecuta `lanzar.sh --check` (se ven nombres de nodo y GPU de cada uno).
4. Ejecuta `lanzar.sh` y señala:
   - las líneas `[Rank i/N] nodo=...` (cada PC con su GPU),
   - las columnas de `W_up` que le toca a cada una (`d_ff / N`),
   - el resultado, `paquetes` 71.92 %.
5. Si tienes tiempo, repite con un nodo menos (por ejemplo `-np 2`) y muestra que el resultado **no cambia**.

## Plan B (si falla la red durante la presentación)

Ejecuta todo en tu PC, con 4 procesos sobre tu GPU:

```bash
cd /opt/llm_cluster && mpirun -np 4 --oversubscribe ./llm_mpi_cuda_red_trafico | tail -20
```

Muestra lo mismo (4 ranks, `paquetes` 71.92 %), pero aclara que son procesos simulados en una sola PC.

---

## Problemas frecuentes

| Síntoma | Qué revisar |
|---|---|
| `mpirun` se queda colgado sin imprimir nada | ¿`tailscale ping` entre todos funciona? ¿Todos tienen `tailscale0`? Prueba con `IFACE=tailscale0` y sin firewall extra |
| `Permission denied (publickey)` | Ese nodo no tiene tu llave en `~/.ssh/authorized_keys` |
| `Host key verification failed` | Entra una vez a mano: `ssh nodoN hostname` y acepta |
| `orted: command not found` / error de versión | Falta OpenMPI en ese nodo o es otra versión. Todos deben tener `mpirun (Open MPI) 4.1.2` |
| `Open MPI ... no hay suficientes slots` | `hosts.txt` con menos líneas que `-np`, o falta `slots=1` |
| Aparece el mismo `nodo=` dos veces | Dos PCs con el mismo nombre. Cambia uno con `sudo hostnamectl set-hostname` |
| Un nodo desaparece en plena demo | Cerró su terminal de Ubuntu o su PC se suspendió |
| Sale `-np 2` aunque hay 3 nodos | Es lo correcto: el programa solo admite 1, 2, 4 u 8 procesos |
