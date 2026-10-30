/*
 * llm_mpi_cuda_red_trafico.cu
 *
 * PROGRAMA HIBRIDO MPI + CUDA -- PARALELISMO DE TENSORES DISTRIBUIDO
 * (version corregida y verificada; disenado para un CLUSTER DE AL MENOS
 *  2 COMPUTADORAS EN RED, CADA UNA CON SU PROPIA GPU, ejecutado sobre
 *  WSL2 + Ubuntu en cada nodo. Ver el manual adjunto, seccion 5, para la
 *  guia completa de configuracion del cluster WSL2 y los comandos de
 *  compilacion y ejecucion.)
 *
 * Caso de estudio (nuevo, independiente de ejercicios previos):
 *   Prompt:  "La red distribuye el trafico mediante ....."
 *   Vocabulario candidato: {"energia", "senales", "datos", "paquetes"}
 *   Prediccion esperada:   "paquetes" (ver manual, seccion 2)
 *
 * Arquitectura de ejecucion hibrida:
 *   1. MPI distribuye el trabajo entre procesos -- en este programa,
 *      UN proceso MPI por NODO/COMPUTADORA del cluster, cada uno
 *      vinculado a la GPU local de su propio nodo.
 *   2. CUDA acelera, en la GPU de cada nodo, el computo matricial local
 *      (GEMM), el Positional Encoding, la Atencion Causal y las
 *      activaciones.
 *   3. La colectiva MPI_Allgather sincroniza entre nodos el tensor
 *      intermedio de la capa Feed-Forward, fragmentado por columnas
 *      (Tensor Parallelism). La reconstruccion de la matriz global a
 *      partir del buffer que entrega Allgather se hace de forma
 *      EXPLICITA (no por reinterpretacion directa), ver PASO 5 abajo --
 *      esta es la parte del algoritmo mas propensa a errores y la que
 *      se valida exhaustivamente en el manual (seccion 4.9 y Anexo B).
 *
 * Requisito de ejecucion: el numero de procesos MPI (-np) DEBE dividir
 * exactamente a d_ff (=8 en este caso: valores validos de -np son
 * 1, 2, 4 u 8). El programa verifica esta condicion explicitamente y
 * aborta con un mensaje claro si no se cumple, en vez de continuar con
 * una configuracion invalida.
 *
 * Compilacion (ver manual, seccion 5.4, para la guia completa en WSL2):
 *   nvcc -O3 -arch=sm_80 -ccbin=mpicxx -c llm_mpi_cuda_red_trafico.cu -o llm_mpi_cuda_red_trafico.o
 *   mpicxx llm_mpi_cuda_red_trafico.o -L${CUDA_HOME}/lib64 -lcudart -o llm_mpi_cuda_red_trafico
 *
 * Ejecucion en un cluster de 2 computadoras (1 proceso MPI por nodo,
 * cada uno con su propia GPU) -- requiere el hostfile descrito en el
 * manual, seccion 5.3:
 *   mpirun --hostfile hosts.txt -np 2 ./llm_mpi_cuda_red_trafico
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <cuda_runtime.h>

#define TILE_SIZE 4
#define MASK_VALUE (-1e20f)

// Macro de verificacion de errores CUDA: aborta TODOS los procesos MPI
// (no solo el que detecto el error) para evitar deadlocks en las
// colectivas MPI posteriores si un proceso muere silenciosamente.
#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "[Rank %d] Error CUDA %s en %s:%d\n", g_rank, cudaGetErrorString(e), __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

static int g_rank = -1; // usado solo para mensajes de diagnostico en CUDA_CHECK

// ============================================================================
// KERNELS CUDA (ejecutados localmente en la GPU de cada nodo del cluster)
// ============================================================================

// Kernel 0: Suma de Positional Encoding: X = E + PE
__global__ void add_positional_encoding_kernel(const float* d_E, const float* d_PE, float* d_X, int N, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < d) {
        int idx = row * d + col;
        d_X[idx] = d_E[idx] + d_PE[idx];
    }
}

// Kernel 1: Multiplicacion de matrices GEMM: C (M x N) = A (M x K) * B (K x N)
__global__ void matmul_cuda_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel 2: Puntajes de atencion escalados S = Q*K^T/sqrt(d_head), con mascara causal aplicada in-place
__global__ void attention_scores_causal_kernel(const float* Q, const float* K, float* S, int N_seq, int d_head) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N_seq && col < N_seq) {
        if (col > row) {
            S[row * N_seq + col] = MASK_VALUE; // un token no puede atender a posiciones futuras
        } else {
            float dot = 0.0f;
            for (int k = 0; k < d_head; ++k) {
                dot += Q[row * d_head + k] * K[col * d_head + k];
            }
            S[row * N_seq + col] = dot / sqrtf((float)d_head);
        }
    }
}

// Kernel 3: Softmax causal por fila (numericamente estable, 1 hilo por fila)
__global__ void softmax_causal_kernel(const float* S, float* P, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        const float* S_row = &S[row * N];
        float* P_row = &P[row * N];

        float max_val = MASK_VALUE;
        for (int c = 0; c <= row; ++c) {
            if (S_row[c] > max_val) max_val = S_row[c];
        }
        float sum_exp = 0.0f;
        for (int c = 0; c <= row; ++c) {
            sum_exp += expf(S_row[c] - max_val);
        }
        for (int c = 0; c < N; ++c) {
            P_row[c] = (c <= row) ? (expf(S_row[c] - max_val) / sum_exp) : 0.0f;
        }
    }
}

// Kernel 4: Activacion ReLU in-place
__global__ void relu_kernel(float* Z, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        Z[idx] = fmaxf(0.0f, Z[idx]);
    }
}

// ============================================================================
// Utilidad de impresion por consola (solo se invoca desde el rank 0)
// ============================================================================
void print_matrix(const char* title, const float* mat, int rows, int cols, const char** row_labels = nullptr, int precision = 4) {
    std::cout << "\n=== " << title << " (" << rows << "x" << cols << ") ===\n";
    for (int r = 0; r < rows; ++r) {
        if (row_labels) std::cout << std::setw(12) << std::left << row_labels[r] << " | ";
        else            std::cout << "Fila " << std::setw(2) << r << " | ";
        for (int c = 0; c < cols; ++c) {
            float v = mat[r * cols + c];
            if (v <= -1e19f) std::cout << std::setw(9) << std::right << "-inf" << " ";
            else std::cout << std::setw(9) << std::right << std::fixed << std::setprecision(precision) << v << " ";
        }
        std::cout << "\n";
    }
}

// ============================================================================
// PROGRAMA PRINCIPAL MPI + CUDA
// ============================================================================
int main(int argc, char** argv) {
    // --------------------------------------------------------------
    // 1. Inicializar el entorno MPI y obtener identidad del proceso
    // --------------------------------------------------------------
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    g_rank = rank;

    char host_name[MPI_MAX_PROCESSOR_NAME];
    int host_name_len = 0;
    MPI_Get_processor_name(host_name, &host_name_len);

    // --------------------------------------------------------------
    // 1.b Asignacion de GPU LOCAL AL NODO (clave para un cluster
    // multi-nodo: "rank % num_gpus" seria incorrecto en varios nodos,
    // porque "rank" es global al cluster, no local al nodo. Se usa
    // MPI_Comm_split_type para obtener un comunicador restringido a
    // los procesos que comparten el mismo nodo fisico (memoria
    // compartida), y dentro de ese comunicador se obtiene el rango
    // LOCAL AL NODO, que es el que realmente debe usarse para elegir
    // la GPU de ESE nodo.
    // --------------------------------------------------------------
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        CUDA_CHECK(cudaSetDevice(local_rank % num_gpus));
    } else {
        std::fprintf(stderr, "[Rank %d, %s] ERROR: no se detecto ninguna GPU CUDA en este nodo.\n", rank, host_name);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int my_device = -1;
    cudaGetDevice(&my_device);

    // --------------------------------------------------------------
    // 2. Hiperparametros del caso de estudio "La red distribuye el
    //    trafico mediante ....."
    // --------------------------------------------------------------
    const int N_seq = 6;       // Tokens en el prompt
    const int d_model = 4;     // Dimension oculta (embedding)
    const int d_ff = 8;        // Dimension intermedia de la FFN
    const int vocab_size = 4;  // Vocabulario candidato final

    const char* tokens[N_seq] = {"La", "red", "distribuye", "el", "trafico", "mediante"};
    const char* vocab[vocab_size] = {"energia", "senales", "datos", "paquetes"};

    // --------------------------------------------------------------
    // 2.b Verificacion OBLIGATORIA de configuracion valida: d_ff debe
    // ser divisible exactamente entre el numero de procesos MPI. Si no
    // lo es, el programa se detiene de forma segura y explicita en vez
    // de continuar con una fragmentacion de columnas invalida.
    // --------------------------------------------------------------
    if (d_ff % num_ranks != 0) {
        if (rank == 0) {
            std::fprintf(stderr,
                "\nERROR DE CONFIGURACION: d_ff (%d) no es divisible entre el numero de\n"
                "procesos MPI solicitado (-np %d).\n"
                "Valores validos de -np para este caso de estudio (d_ff=%d): 1, 2, 4 u 8.\n"
                "Ejemplo para un cluster de 2 computadoras (1 proceso por nodo):\n"
                "  mpirun --hostfile hosts.txt -np 2 ./llm_mpi_cuda_red_trafico\n\n",
                d_ff, num_ranks, d_ff);
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    if (rank == 0) {
        std::cout << "=======================================================================\n";
        std::cout << " INFERENCIA DE LLM HIBRIDA (MPI + CUDA) -- PARALELISMO DE TENSORES\n";
        std::cout << " Caso: \"La red distribuye el trafico mediante .....\"\n";
        std::cout << "=======================================================================\n";
        std::cout << " Procesos MPI (nodos del cluster): " << num_ranks << "\n";
        std::cout << " Configuracion de tensores: N=" << N_seq << ", d_model=" << d_model
                  << ", d_ff=" << d_ff << ", Vocabulario=" << vocab_size << "\n";
        std::cout << "-----------------------------------------------------------------------\n";
    }
    // Cada rank reporta su nodo y GPU asignada (orden no determinista entre nodos,
    // es normal: cada proceso imprime de forma independiente).
    std::fprintf(stdout, "[Rank %d/%d] nodo=%s  rank_local_al_nodo=%d/%d  GPU_asignada=%d\n",
                 rank, num_ranks, host_name, local_rank, local_size, my_device);
    std::fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // --------------------------------------------------------------
    // 3. MATRICES NUMERICAS DEL CASO DE ESTUDIO (identicas en todos
    //    los procesos -- cada nodo las tiene en su propia memoria,
    //    no hay necesidad de transmitirlas por MPI porque no dependen
    //    de ningun dato aleatorio ni de entrada externa).
    // --------------------------------------------------------------
    // a) Embedding semantico E (6x4)
    const float h_E[N_seq * d_model] = {
         0.55f, -0.12f,  0.72f,  0.39f,  // "La"
        -0.81f,  0.95f,  0.52f,  0.57f,  // "red"
        -0.74f, -0.10f, -0.26f,  0.85f,  // "distribuye"
         0.29f,  0.65f, -0.11f, -0.55f,  // "el"
         0.11f, -0.87f,  0.66f,  0.26f,  // "trafico"
         0.52f, -0.29f,  0.94f,  0.79f   // "mediante"
    };

    // b) Positional Encoding PE (6x4) -- formula sinusoidal real de
    //    Vaswani et al.: PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...)
    const float h_PE[N_seq * d_model] = {
         0.000f,  1.000f,  0.000f,  1.000f,  // Pos 0
         0.841f,  0.540f,  0.010f,  1.000f,  // Pos 1
         0.909f, -0.416f,  0.020f,  1.000f,  // Pos 2
         0.141f, -0.990f,  0.030f,  1.000f,  // Pos 3
        -0.757f, -0.654f,  0.040f,  0.999f,  // Pos 4
        -0.959f,  0.284f,  0.050f,  0.999f   // Pos 5
    };

    // c) Pesos de proyeccion Q, K, V (4x4)
    const float h_WQ[d_model * d_model] = {
         0.56f, -0.61f, -0.07f, -0.91f,
        -0.69f,  0.37f,  0.49f,  0.94f,
        -0.35f, -0.26f, -0.06f, -0.62f,
        -0.74f, -0.05f, -0.55f,  0.34f
    };
    const float h_WK[d_model * d_model] = {
        -0.13f,  0.67f,  0.40f, -0.38f,
         0.66f,  0.61f, -0.23f, -0.42f,
         0.36f, -0.72f, -0.60f, -0.99f,
         0.57f,  0.33f,  0.41f,  0.56f
    };
    const float h_WV[d_model * d_model] = {
        -0.08f,  0.14f, -0.72f, -0.77f,
         0.34f, -0.06f,  0.13f,  0.53f,
         0.27f,  0.11f,  0.12f, -0.39f,
        -0.94f, -0.13f, -0.57f, -0.18f
    };

    // d) Pesos globales de la FFN: W_up (4x8) y W_down (8x4)
    const float h_W_up_global[d_model * d_ff] = {
         0.71f, -0.53f, -0.88f, -0.44f, -0.41f,  0.32f,  0.11f,  0.57f,
         0.33f, -0.19f,  0.63f, -0.67f, -0.95f, -0.82f,  0.44f, -0.08f,
        -0.68f,  0.00f, -0.70f,  0.39f, -0.11f, -0.24f, -0.40f,  0.26f,
        -0.28f, -0.82f, -0.76f,  0.92f,  0.82f,  0.40f, -0.47f,  0.94f
    };
    const float h_W_down_global[d_ff * d_model] = {
         0.56f,  0.43f, -0.10f, -0.46f,
        -0.81f,  0.81f, -0.09f, -0.60f,
        -0.39f,  0.16f, -0.65f,  0.71f,
         0.52f,  0.44f, -0.14f,  0.25f,
         0.17f,  0.30f, -0.83f, -0.17f,
        -0.92f, -0.01f, -0.34f, -0.71f,
        -0.79f,  0.18f, -0.66f,  0.85f,
         0.16f, -0.31f,  0.18f, -0.95f
    };

    // e) Un-embedding W_vocab (4x4)
    const float h_W_vocab[d_model * vocab_size] = {
         0.92f, -0.04f,  0.57f, -0.83f,
        -0.03f, -0.02f,  0.88f,  0.14f,
        -0.05f, -0.47f, -0.34f,  0.04f,
        -0.12f, -0.96f,  0.65f,  0.79f
    };

    // --------------------------------------------------------------
    // 4. PARALELISMO DE TENSORES: fragmentacion de columnas de W_up
    //    (ya validado el divisor en el paso 2.b, por lo que esta
    //    division es exacta para CUALQUIER num_ranks aceptado).
    // --------------------------------------------------------------
    const int d_ff_local = d_ff / num_ranks;
    std::vector<float> h_W_up_local(d_model * d_ff_local);
    for (int r = 0; r < d_model; ++r) {
        for (int c = 0; c < d_ff_local; ++c) {
            int global_col = rank * d_ff_local + c;
            h_W_up_local[r * d_ff_local + c] = h_W_up_global[r * d_ff + global_col];
        }
    }

    // --------------------------------------------------------------
    // 5. RESERVA DE MEMORIA EN GPU Y COPIA HOST -> DEVICE
    // --------------------------------------------------------------
    float *d_E, *d_PE, *d_X, *d_WQ, *d_WK, *d_WV;
    float *d_Q, *d_K, *d_V, *d_S, *d_P, *d_O;
    float *d_W_up_local, *d_Z_local;
    float *d_H_ffn_global, *d_W_down_global, *d_FFN_out, *d_W_vocab, *d_Logits;

    CUDA_CHECK(cudaMalloc(&d_E,  N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_PE, N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X,  N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WQ, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WK, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WV, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q,  N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K,  N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V,  N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S,  N_seq * N_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P,  N_seq * N_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O,  N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_up_local, d_model * d_ff_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z_local,    N_seq * d_ff_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H_ffn_global,  N_seq * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_down_global, d_ff * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FFN_out, N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_vocab, d_model * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Logits,  vocab_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_E,  h_E,  N_seq * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_PE, h_PE, N_seq * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_WQ, h_WQ, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_WK, h_WK, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_WV, h_WV, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_up_local, h_W_up_local.data(), d_model * d_ff_local * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_down_global, h_W_down_global, d_ff * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_vocab, h_W_vocab, d_model * vocab_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);

    // --------------------------------------------------------------
    // PASO 0: X = E + PE (en GPU)
    // --------------------------------------------------------------
    dim3 grid_pe((d_model + TILE_SIZE - 1) / TILE_SIZE, (N_seq + TILE_SIZE - 1) / TILE_SIZE);
    add_positional_encoding_kernel<<<grid_pe, block>>>(d_E, d_PE, d_X, N_seq, d_model);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) {
        float h_X[N_seq * d_model];
        CUDA_CHECK(cudaMemcpy(h_X, d_X, N_seq * d_model * sizeof(float), cudaMemcpyDeviceToHost));
        print_matrix("Matriz de entrada X = E + PE", h_X, N_seq, d_model, tokens);
    }

    // --------------------------------------------------------------
    // PASO 1: Proyecciones Q, K, V
    // --------------------------------------------------------------
    dim3 grid_proj((d_model + TILE_SIZE - 1) / TILE_SIZE, (N_seq + TILE_SIZE - 1) / TILE_SIZE);
    matmul_cuda_kernel<<<grid_proj, block>>>(d_X, d_WQ, d_Q, N_seq, d_model, d_model);
    matmul_cuda_kernel<<<grid_proj, block>>>(d_X, d_WK, d_K, N_seq, d_model, d_model);
    matmul_cuda_kernel<<<grid_proj, block>>>(d_X, d_WV, d_V, N_seq, d_model, d_model);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------------
    // PASO 2-3: Atencion causal S y Softmax P
    // --------------------------------------------------------------
    dim3 grid_att((N_seq + TILE_SIZE - 1) / TILE_SIZE, (N_seq + TILE_SIZE - 1) / TILE_SIZE);
    attention_scores_causal_kernel<<<grid_att, block>>>(d_Q, d_K, d_S, N_seq, d_model);
    CUDA_CHECK(cudaDeviceSynchronize());

    softmax_causal_kernel<<<N_seq, 1>>>(d_S, d_P, N_seq);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------------
    // PASO 4: Contextualizacion O = P * V
    // --------------------------------------------------------------
    matmul_cuda_kernel<<<grid_proj, block>>>(d_P, d_V, d_O, N_seq, d_model, N_seq);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) {
        float h_P[N_seq * N_seq], h_O[N_seq * d_model];
        CUDA_CHECK(cudaMemcpy(h_P, d_P, N_seq * N_seq * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_O, d_O, N_seq * d_model * sizeof(float), cudaMemcpyDeviceToHost));
        print_matrix("Matriz de atencion causal P", h_P, N_seq, N_seq, tokens, 3);
        print_matrix("Matriz contextualizada O = P.V", h_O, N_seq, d_model, tokens);
    }

    // --------------------------------------------------------------
    // PASO 5: FFN DISTRIBUIDA CON MPI + CUDA (PARALELISMO DE TENSORES)
    // --------------------------------------------------------------
    // a) Expansion local: Z_local = O (N_seq x d_model) * W_up_local (d_model x d_ff_local)
    dim3 grid_ff_local((d_ff_local + TILE_SIZE - 1) / TILE_SIZE, (N_seq + TILE_SIZE - 1) / TILE_SIZE);
    matmul_cuda_kernel<<<grid_ff_local, block>>>(d_O, d_W_up_local, d_Z_local, N_seq, d_ff_local, d_model);
    CUDA_CHECK(cudaDeviceSynchronize());

    // b) ReLU local in-place
    relu_kernel<<<(N_seq * d_ff_local + 31) / 32, 32>>>(d_Z_local, N_seq * d_ff_local);
    CUDA_CHECK(cudaDeviceSynchronize());

    // c) Copiar resultado local de GPU a host para la colectiva MPI
    std::vector<float> h_H_local(N_seq * d_ff_local);
    CUDA_CHECK(cudaMemcpy(h_H_local.data(), d_Z_local, N_seq * d_ff_local * sizeof(float), cudaMemcpyDeviceToHost));

    // d) MPI_Allgather: cada nodo envia su bloque local (N_seq x d_ff_local) y
    //    recibe la CONCATENACION, en orden de rank, de los bloques de TODOS
    //    los nodos. El buffer de recepcion se dimensiona exactamente a
    //    num_ranks * N_seq * d_ff_local elementos -- su tamano real
    //    garantizado por la semantica de Allgather -- evitando cualquier
    //    desborde de memoria.
    std::vector<float> h_H_flat(num_ranks * N_seq * d_ff_local);
    if (num_ranks > 1) {
        MPI_Allgather(h_H_local.data(), N_seq * d_ff_local, MPI_FLOAT,
                      h_H_flat.data(), N_seq * d_ff_local, MPI_FLOAT,
                      MPI_COMM_WORLD);
    } else {
        h_H_flat = h_H_local;
    }

    // e) RECONSTRUCCION EXPLICITA de la matriz global (N_seq x d_ff) a
    //    partir del buffer plano. El buffer de Allgather esta organizado
    //    como [bloque del rank 0][bloque del rank 1]...[bloque del rank
    //    num_ranks-1], donde cada bloque es (N_seq x d_ff_local) en orden
    //    de fila (row-major). NO es valido reinterpretar ese buffer
    //    directamente como una matriz (N_seq x d_ff) row-major -- eso
    //    mezclaria los limites de fila de cada bloque. En su lugar, se
    //    recorre explicitamente cada bloque y se ubica cada elemento en
    //    su posicion [fila][columna_global] correcta dentro de la matriz
    //    global:
    //       flat[r][i][j]  (indice = r*(N_seq*d_ff_local) + i*d_ff_local + j)
    //       corresponde a  H_global[i][r*d_ff_local + j]
    std::vector<float> h_H_ffn_global(N_seq * d_ff);
    for (int r = 0; r < num_ranks; ++r) {
        for (int i = 0; i < N_seq; ++i) {
            for (int j = 0; j < d_ff_local; ++j) {
                int flat_idx    = r * (N_seq * d_ff_local) + i * d_ff_local + j;
                int global_col  = r * d_ff_local + j;
                h_H_ffn_global[i * d_ff + global_col] = h_H_flat[flat_idx];
            }
        }
    }

    // f) Contraccion: H_out = H_ffn_global (N_seq x d_ff) * W_down (d_ff x d_model)
    CUDA_CHECK(cudaMemcpy(d_H_ffn_global, h_H_ffn_global.data(), N_seq * d_ff * sizeof(float), cudaMemcpyHostToDevice));
    matmul_cuda_kernel<<<grid_proj, block>>>(d_H_ffn_global, d_W_down_global, d_FFN_out, N_seq, d_model, d_ff);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) {
        float h_FFN_out[N_seq * d_model];
        CUDA_CHECK(cudaMemcpy(h_FFN_out, d_FFN_out, N_seq * d_model * sizeof(float), cudaMemcpyDeviceToHost));
        print_matrix("Matriz final oculta FFN H_out (computo distribuido MPI+CUDA)", h_FFN_out, N_seq, d_model, tokens);
    }

    // --------------------------------------------------------------
    // PASO 6: Extraer h_ultimo (ultima fila, token "mediante") y proyectar al vocabulario
    // --------------------------------------------------------------
    const float* d_h_ultimo = d_FFN_out + (N_seq - 1) * d_model;
    matmul_cuda_kernel<<<dim3(1, 1), dim3(vocab_size, 1)>>>(d_h_ultimo, d_W_vocab, d_Logits, 1, vocab_size, d_model);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------------
    // PASO 7: Softmax final y prediccion (solo en rank 0)
    // --------------------------------------------------------------
    if (rank == 0) {
        float h_Logits[vocab_size];
        CUDA_CHECK(cudaMemcpy(h_Logits, d_Logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

        float h_Probs[vocab_size];
        float max_l = h_Logits[0];
        for (int i = 1; i < vocab_size; ++i) if (h_Logits[i] > max_l) max_l = h_Logits[i];
        float sum_e = 0.0f;
        for (int i = 0; i < vocab_size; ++i) { h_Probs[i] = expf(h_Logits[i] - max_l); sum_e += h_Probs[i]; }
        for (int i = 0; i < vocab_size; ++i) h_Probs[i] /= sum_e;

        std::cout << "\n=======================================================================\n";
        std::cout << " RESULTADO PREDICCION MPI+CUDA (LOGITS Y PROBABILIDADES)\n";
        std::cout << "=======================================================================\n";
        int best_idx = 0; float max_p = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            std::cout << " Token candidato [" << v << "]: " << std::setw(10) << std::left << vocab[v]
                      << " | Logit: " << std::setw(8) << std::fixed << std::setprecision(4) << h_Logits[v]
                      << " | Probabilidad: " << std::setw(6) << std::setprecision(2) << (h_Probs[v] * 100.0f) << "%\n";
            if (h_Probs[v] > max_p) { max_p = h_Probs[v]; best_idx = v; }
        }
        std::cout << "\n-----------------------------------------------------------------------\n";
        std::cout << " -> TOKEN SELECCIONADO: \"" << vocab[best_idx] << "\" ("
                  << std::setprecision(2) << (max_p * 100.0f) << "% de confianza)\n";
        std::cout << " -> Frase completa: \"La red distribuye el trafico mediante " << vocab[best_idx] << "\"\n";
        std::cout << " -> Ejecutado con " << num_ranks << " proceso(s) MPI (nodos del cluster)\n";
        std::cout << "=======================================================================\n";
    }

    // --------------------------------------------------------------
    // Liberacion de recursos GPU y MPI
    // --------------------------------------------------------------
    cudaFree(d_E); cudaFree(d_PE); cudaFree(d_X);
    cudaFree(d_WQ); cudaFree(d_WK); cudaFree(d_WV);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_S); cudaFree(d_P); cudaFree(d_O);
    cudaFree(d_W_up_local); cudaFree(d_Z_local);
    cudaFree(d_H_ffn_global); cudaFree(d_W_down_global); cudaFree(d_FFN_out);
    cudaFree(d_W_vocab); cudaFree(d_Logits);

    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return 0;
}
