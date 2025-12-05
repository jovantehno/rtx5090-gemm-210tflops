/**
 * Example 02: Tiled GEMM with Shared Memory
 *
 * Key optimization: Use shared memory to reduce global memory access.
 *
 * Concepts demonstrated:
 * - Tiled matrix multiplication
 * - Shared memory usage
 * - Coalesced memory access
 * - Data reuse within thread block
 *
 * Compile: nvcc -O3 -arch=sm_80 02_tiled_gemm.cu -o tiled_gemm
 * Run: ./tiled_gemm
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#define M 1024
#define N 1024
#define K 1024

// Tile size - must match block dimensions
#define TILE_SIZE 32

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
 * Tiled GEMM kernel
 *
 * Each thread block computes a TILE_SIZE x TILE_SIZE tile of C.
 * We iterate over K dimension in tiles, loading tiles of A and B
 * into shared memory.
 *
 * Memory access pattern:
 * - Global → Shared: Each thread loads one element (coalesced)
 * - Shared → Registers: Each thread reads full row/column (no bank conflicts with padding)
 */
__global__ void tiled_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    // Shared memory for tiles of A and B
    // +1 padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column of C element to compute
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles of K dimension
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        int a_col = t * TILE_SIZE + tx;
        if (row < m && a_col < k) {
            As[ty][tx] = __half2float(A[row * k + a_col]);
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + ty;
        if (b_row < k && col < n) {
            Bs[ty][tx] = __half2float(B[b_row * n + col]);
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Wait for all threads to finish loading
        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        // Wait before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < m && col < n) {
        C[row * n + col] = __float2half(sum);
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

void launch_tiled_gemm(const half* A, const half* B, half* C, int m, int n, int k) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    tiled_gemm_kernel<<<grid, block>>>(A, B, C, m, n, k);
}

int main() {
    printf("=== Example 02: Tiled GEMM with Shared Memory ===\n");
    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("Tile size: %d x %d\n", TILE_SIZE, TILE_SIZE);

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

    float avg_ms = benchmark_kernel(launch_tiled_gemm, d_A, d_B, d_C, M, N, K, 10, 100);

    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    printf("Average time: %.3f ms\n", avg_ms);
    printf("Performance: %.2f TFLOPS\n", tflops);
    printf("\nKey improvements over naive:\n");
    printf("- Shared memory reduces global memory traffic by %dx\n", TILE_SIZE);
    printf("- Coalesced memory access pattern\n");
    printf("- Data reuse within thread block\n");

    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
