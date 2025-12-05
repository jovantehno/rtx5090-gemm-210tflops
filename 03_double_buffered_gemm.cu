/**
 * Example 03: Double-Buffered GEMM (Software Pipelining)
 *
 * Key optimization: Overlap memory loads with computation.
 *
 * Concepts demonstrated:
 * - Double buffering (2-stage pipeline)
 * - Prefetching next tile while computing current tile
 * - Hiding memory latency
 *
 * This is a simplified version of CUDA-L2's multi-stage pipelining.
 *
 * Compile: nvcc -O3 -arch=sm_80 03_double_buffered_gemm.cu -o double_buffered_gemm
 * Run: ./double_buffered_gemm
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#define M 1024
#define N 1024
#define K 1024

#define BM 64   // Block tile M
#define BN 64   // Block tile N
#define BK 32   // Block tile K

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/**
 * Double-buffered GEMM kernel
 *
 * Pipeline stages:
 *   Buffer 0: Load tile[i+1] from global memory
 *   Buffer 1: Compute on tile[i]
 *
 * While computing on one buffer, we prefetch into the other.
 */
__global__ void double_buffered_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    // Double buffer - 2 copies of shared memory tiles
    __shared__ float As[2][BM][BK + 1];  // +1 to avoid bank conflicts
    __shared__ float Bs[2][BK][BN + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread position within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    // Each thread computes a 4x4 sub-tile of C
    int thread_row = (tid / 16) * 4;
    int thread_col = (tid % 16) * 4;

    // Global row/col for this block's output tile
    int row_start = by * BM;
    int col_start = bx * BN;

    // Accumulator registers (4x4 per thread)
    float acc[4][4] = {0.0f};

    int num_tiles = (k + BK - 1) / BK;

    // --- Load first tile into buffer 0 ---
    int buf_idx = 0;

    // Cooperative loading: all threads help load the tile
    // Each thread loads multiple elements
    for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
        int local_row = i / BK;
        int local_col = i % BK;
        int global_row = row_start + local_row;
        int global_col = local_col;

        if (global_row < m && global_col < k) {
            As[buf_idx][local_row][local_col] = __half2float(A[global_row * k + global_col]);
        } else {
            As[buf_idx][local_row][local_col] = 0.0f;
        }
    }

    for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y) {
        int local_row = i / BN;
        int local_col = i % BN;
        int global_row = local_row;
        int global_col = col_start + local_col;

        if (global_row < k && global_col < n) {
            Bs[buf_idx][local_row][local_col] = __half2float(B[global_row * n + global_col]);
        } else {
            Bs[buf_idx][local_row][local_col] = 0.0f;
        }
    }

    __syncthreads();

    // --- Main loop: process tiles with double buffering ---
    for (int tile = 0; tile < num_tiles; tile++) {
        int next_buf = 1 - buf_idx;

        // Prefetch next tile into other buffer (if not last tile)
        if (tile + 1 < num_tiles) {
            int next_k_offset = (tile + 1) * BK;

            for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
                int local_row = i / BK;
                int local_col = i % BK;
                int global_row = row_start + local_row;
                int global_col = next_k_offset + local_col;

                if (global_row < m && global_col < k) {
                    As[next_buf][local_row][local_col] = __half2float(A[global_row * k + global_col]);
                } else {
                    As[next_buf][local_row][local_col] = 0.0f;
                }
            }

            for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y) {
                int local_row = i / BN;
                int local_col = i % BN;
                int global_row = next_k_offset + local_row;
                int global_col = col_start + local_col;

                if (global_row < k && global_col < n) {
                    Bs[next_buf][local_row][local_col] = __half2float(B[global_row * n + global_col]);
                } else {
                    Bs[next_buf][local_row][local_col] = 0.0f;
                }
            }
        }

        // Compute on current buffer while prefetch happens
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Load values from shared memory
            float a_frag[4];
            float b_frag[4];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_frag[i] = As[buf_idx][thread_row + i][kk];
                b_frag[i] = Bs[buf_idx][kk][thread_col + i];
            }

            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();

        // Swap buffers
        buf_idx = next_buf;
    }

    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int out_row = row_start + thread_row + i;
            int out_col = col_start + thread_col + j;
            if (out_row < m && out_col < n) {
                C[out_row * n + out_col] = __float2half(acc[i][j]);
            }
        }
    }
}

void init_matrix(half* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = __float2half((float)(rand() % 10) / 10.0f);
    }
}

float benchmark_kernel(
    void (*kernel_launch)(const half*, const half*, half*, int, int, int),
    const half* d_A, const half* d_B, half* d_C,
    int m, int n, int k,
    int warmup_iters, int bench_iters
) {
    for (int i = 0; i < warmup_iters; i++) {
        kernel_launch(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        kernel_launch(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / bench_iters;
}

void launch_double_buffered_gemm(const half* A, const half* B, half* C, int m, int n, int k) {
    // 256 threads per block (16x16 layout)
    dim3 block(16, 16);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    double_buffered_gemm_kernel<<<grid, block>>>(A, B, C, m, n, k);
}

int main() {
    printf("=== Example 03: Double-Buffered GEMM ===\n");
    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("Block tile: %d x %d x %d\n", BM, BN, BK);
    printf("Pipeline stages: 2 (double buffering)\n");

    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    half *h_C = (half*)malloc(M * N * sizeof(half));

    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    float avg_ms = benchmark_kernel(launch_double_buffered_gemm, d_A, d_B, d_C, M, N, K, 10, 100);

    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    printf("Average time: %.3f ms\n", avg_ms);
    printf("Performance: %.2f TFLOPS\n", tflops);
    printf("\nKey improvements:\n");
    printf("- Overlaps memory loads with computation\n");
    printf("- Each thread computes 4x4 sub-tile (more work per thread)\n");
    printf("- Hides memory latency through prefetching\n");

    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
