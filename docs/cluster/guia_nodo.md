# Guía del nodo (tu PC con GPU NVIDIA presta su GPU al clúster)

Sigue los pasos en orden. Tiempo: entre 1 y 2 horas la primera vez (casi todo es esperar descargas).

**Todos los comandos con `$` se pegan en la terminal de Ubuntu (WSL2)**, salvo los marcados como *PowerShell*. Cada paso termina con un **"Debes ver"**: si no lo ves, no sigas; mira *Problemas frecuentes* al final.

## Antes de empezar: qué necesitas

- [ ] Windows 10 (22H2) o Windows 11 con una GPU **NVIDIA** (GeForce, RTX, GTX, Quadro).
- [ ] Unos 15 GB libres en disco.
- [ ] La carpeta compartida del grupo, con los archivos de la sección 7 del [README](README.md).
- [ ] La cuenta de Tailscale del grupo (te la indica el lanzador).

---

## Paso 1. Driver NVIDIA en Windows y Ubuntu en WSL2 *(PowerShell)*

**Qué vas a hacer:** que Ubuntu dentro de Windows pueda usar tu tarjeta.

1. Instala o actualiza el driver NVIDIA **en Windows** (no dentro de Ubuntu): https://www.nvidia.com/es-la/drivers/ o la app NVIDIA.
2. Abre **PowerShell como administrador** y pega:

```powershell
wsl --install -d Ubuntu-22.04
wsl -l -v
```

Reinicia si te lo pide, abre **Ubuntu 22.04** desde el menú Inicio y crea tu usuario y contraseña.

**Debes ver** (en `wsl -l -v`) `Ubuntu-22.04` con `VERSION 2`.

> Usamos **Ubuntu 22.04 en todos los nodos** para que OpenMPI tenga exactamente la misma versión.

- [ ] Ubuntu 22.04 abierto con WSL versión 2.

---

## Paso 2. Activar systemd y conectarte a Tailscale

**Qué vas a hacer:** instalar Tailscale **dentro de Ubuntu**, para que tu Ubuntu tenga su propia IP en la red del grupo. Así no hay que tocar el firewall de Windows ni configurar redes de WSL.

**2.1 Activar systemd** (hace falta para Tailscale y SSH):

```bash
printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf
```

Cierra Ubuntu y, en *PowerShell*, ejecuta `wsl --shutdown`. Vuelve a abrir Ubuntu.

**2.2 Instalar y conectar Tailscale:**

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

`tailscale up` muestra un enlace: ábrelo en el navegador de Windows e inicia sesión con **la cuenta del grupo** (o acepta la invitación que te mandó el lanzador).

**2.3 Comprobar:**

```bash
tailscale ip -4
ip link show tailscale0
tailscale status
```

**Debes ver** una IP `100.x.x.x`, la interfaz `tailscale0` y, en `tailscale status`, a los otros integrantes del grupo.

- [ ] Tengo IP de Tailscale y veo a los demás nodos.

---

## Paso 3. Comprobar la GPU dentro de Ubuntu

```bash
nvidia-smi
```

**Debes ver** una tabla con tu tarjeta y `CUDA Version`. Si no aparece, **no instales ningún driver dentro de Ubuntu** (puede romper el soporte): actualiza el driver de Windows (paso 1) y repite `wsl --shutdown`.

- [ ] `nvidia-smi` muestra mi GPU.

---

## Paso 4. Instalar CUDA Toolkit, OpenMPI y SSH

**Qué vas a hacer:** instalar el compilador `nvcc`, OpenMPI y el servidor SSH. Usa el repositorio **WSL-Ubuntu** de NVIDIA (el de Ubuntu estándar instalaría un driver de Linux que choca con el de Windows).

```bash
sudo apt-get update
sudo apt-get install -y build-essential wget openssh-server libopenmpi-dev openmpi-bin

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

> Si algún enlace de NVIDIA no existe, elige tu versión en https://developer.nvidia.com/cuda-downloads (Linux → x86_64 → **WSL-Ubuntu** → 2.0 → deb network). **Todos deben instalar la misma versión** (aquí, 12.6).

Dejar `nvcc` y las bibliotecas CUDA disponibles también para sesiones SSH sin terminal (las usa `mpirun`):

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo /usr/local/cuda/lib64 | sudo tee /etc/ld.so.conf.d/cuda.conf
sudo ldconfig
source ~/.bashrc
nvcc --version && mpirun --version | head -1
```

**Debes ver** la versión de `nvcc` (release 12.6) y `mpirun (Open MPI) 4.1.2`.

- [ ] `nvcc` y `mpirun` responden.

---

## Paso 5. Copiar el programa a `/opt/llm_cluster`

**Qué vas a hacer:** dejar los archivos en la **misma ruta absoluta** que todos los demás. `mpirun` busca el programa en esa ruta en cada PC.

```bash
sudo mkdir -p /opt/llm_cluster && sudo chown "$USER" /opt/llm_cluster
```

Copia a esa carpeta, desde la carpeta compartida del grupo, estos tres archivos: `llm_mpi_cuda_red_trafico.cu`, `compilar.sh` y `nodo.sh`. Desde Windows, tu disco `C:` está en `/mnt/c/`. Por ejemplo, si los descargaste a Descargas:

```bash
cp /mnt/c/Users/<TU_USUARIO_WINDOWS>/Downloads/{llm_mpi_cuda_red_trafico.cu,compilar.sh,nodo.sh} /opt/llm_cluster/
sha256sum /opt/llm_cluster/llm_mpi_cuda_red_trafico.cu | cut -c1-12
```

**Debes ver** `7d86f3647062`. Si ves otra huella, tienes otra versión del código: vuelve a descargarlo.

- [ ] Los tres archivos están en `/opt/llm_cluster` y la huella es `7d86f3647062`.

---

## Paso 6. Compilar

```bash
bash /opt/llm_cluster/compilar.sh
```

El script detecta solo la arquitectura de tu GPU (`sm_86`, `sm_89`, etc.) y compila.

**Debes ver**, al final, `OK: compilado /opt/llm_cluster/llm_mpi_cuda_red_trafico`.

- [ ] Compilado sin errores.

---

## Paso 7. Ensayo individual (solo tu PC)

**Qué vas a hacer:** comprobar que el programa funciona en tu GPU antes de involucrar a nadie más.

```bash
cd /opt/llm_cluster
mpirun -np 1 ./llm_mpi_cuda_red_trafico | tail -15
mpirun -np 2 --oversubscribe ./llm_mpi_cuda_red_trafico | tail -15
```

El segundo comando simula 2 nodos con 2 procesos en tu misma GPU. Para cada caso, **debes ver** una línea `[Rank ...] nodo=... GPU_asignada=0` por proceso, y al final la predicción **`paquetes` con 71.92 % de confianza**.

Si ves `ERROR CUDA` o `no se detecto ninguna GPU`, vuelve al paso 3.

- [ ] Con `-np 1` y `-np 2` sale `paquetes` 71.92 %.

---

## Paso 8. Permitir que el lanzador entre por SSH

**Qué vas a hacer:** que `mpirun`, desde la PC del lanzador, pueda arrancar tu proceso **sin pedir contraseña**.

1. Arranca el servidor SSH (queda activo cada vez que abras Ubuntu):

```bash
sudo systemctl enable --now ssh
systemctl is-active ssh
```

**Debes ver** `active`.

2. El lanzador te enviará su **llave pública** (una línea que empieza con `ssh-ed25519`). Pégala entre comillas simples:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'PEGA_AQUI_LA_LLAVE_PUBLICA_DEL_LANZADOR' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

- [ ] SSH activo y llave del lanzador agregada.

---

## Paso 9. Revisar todo y enviar tus datos al grupo

```bash
bash /opt/llm_cluster/nodo.sh
```

Revisa 6 cosas (GPU, Tailscale, SSH, CUDA/OpenMPI, programa, nombre). Cuando todo diga `[ OK ]`, **copia la última línea** (empieza con `host=...`) y envíala al grupo. El lanzador la usa para armar `hosts.txt` y para llenar el inventario del [README](README.md).

- [ ] `nodo.sh` sin `[FALTA]` y mi línea enviada al grupo.

---

## Paso 10. Durante el ensayo general y la presentación

- **Mantén abierta una terminal de Ubuntu (WSL)**. Windows puede apagar Ubuntu cuando no hay ninguna abierta, y tu nodo desaparecería del clúster. Basta con dejarla abierta y minimizada.
- Conecta la PC al cargador y desactiva la suspensión.
- No cierres Tailscale ni ejecutes `wsl --shutdown`.
- No necesitas ejecutar nada más: el lanzador arranca tu proceso. Verás su salida solo en la pantalla del lanzador.

## Al terminar

```bash
sudo tailscale down      # si quieres salir de la red del grupo
```

---

## Problemas frecuentes

| Síntoma | Qué revisar |
|---|---|
| `nvidia-smi: command not found` o no ve la GPU | Driver NVIDIA **de Windows** actualizado, `wsl --shutdown` y reabrir. No instales driver en Ubuntu |
| No existe `tailscale0` o `tailscale up` falla | ¿systemd activo? `systemctl is-system-running` debe responder `running` o `degraded`. Repite el paso 2.1 |
| `nvcc: command not found` | `source ~/.bashrc` o abre una terminal nueva |
| `compilar.sh`: `Unsupported gpu architecture` | Tu CUDA Toolkit es muy viejo para tu GPU. Instala una versión más nueva (y avisa, porque todos deben tener la misma) |
| `mpirun` reporta `orted: command not found` en otro nodo | OpenMPI no está instalado en ese nodo, o no es la misma versión. Compara `mpirun --version` en todos |
| `error while loading shared libraries: libcudart.so` | Falta el `ldconfig` del paso 4: `echo /usr/local/cuda/lib64 \| sudo tee /etc/ld.so.conf.d/cuda.conf && sudo ldconfig` |
| El nodo "no responde" para el lanzador | ¿Terminal de Ubuntu abierta? ¿`systemctl is-active ssh`? ¿`tailscale ping <IP del lanzador>` funciona? |
| Salen nodos con el mismo `nodo=` en la salida | Dos PCs tienen el mismo nombre. Cambia el tuyo: `sudo hostnamectl set-hostname nodoN` y reabre Ubuntu |
| `ERROR DE CONFIGURACION: d_ff (8) no es divisible...` | El `-np` no es 1, 2, 4 u 8. Usa `lanzar.sh`, que lo elige solo |
