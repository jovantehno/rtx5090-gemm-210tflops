/**
 * Example 01: Naive GEMM Implementation
 *
 * This is the baseline - simple but slow.
 * Each thread computes one element of C.
 *
 * Problems:
 * - Terrible memory access pattern (not coalesced)
 * - No data reuse (same data loaded many times)
 * - No use of shared memory or tensor cores
 *
 * Compile: nvcc -O3 -arch=sm_80 01_naive_gemm.cu -o naive_gemm
 * Run: ./naive_gemm
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

// Matrix dimensions
#define M 1024
#define N 1024
#define K 1024

// Error checking macro
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
 * Naive GEMM kernel
 * C = A * B
 * A: M x K
 * B: K x N
 * C: M x N
 */
__global__ void naive_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;

        // Dot product of row of A and column of B
        for (int i = 0; i < k; i++) {
            sum += __half2float(A[row * k + i]) * __half2float(B[i * n + col]);
        }

        C[row * n + col] = __float2half(sum);
    }
}

// Initialize matrix with random values
void init_matrix(half* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = __float2half((float)(rand() % 10) / 10.0f);
    }
}

// Benchmark kernel
float benchmark_kernel(
    void (*kernel_launch)(const half*, const half*, half*, int, int, int),
    const half* d_A, const half* d_B, half* d_C,
    int m, int n, int k,
    int warmup_iters, int bench_iters
) {
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        kernel_launch(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
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

void launch_naive_gemm(const half* A, const half* B, half* C, int m, int n, int k) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    naive_gemm_kernel<<<grid, block>>>(A, B, C, m, n, k);
}

int main() {
    printf("=== Example 01: Naive GEMM ===\n");
    printf("Matrix size: %d x %d x %d\n", M, N, K);

    // Host memory
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    half *h_C = (half*)malloc(M * N * sizeof(half));

    // Initialize
    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Device memory
    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // Benchmark
    float avg_ms = benchmark_kernel(launch_naive_gemm, d_A, d_B, d_C, M, N, K, 10, 100);

    // Calculate TFLOPS
    double flops = 2.0 * M * N * K;  // multiply-add = 2 ops
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    printf("Average time: %.3f ms\n", avg_ms);
    printf("Performance: %.2f TFLOPS\n", tflops);
    printf("\nThis is our baseline. Let's improve it!\n");

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
