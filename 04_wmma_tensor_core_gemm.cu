/**
 * Example 04: WMMA Tensor Core GEMM
 *
 * Key optimization: Use Tensor Cores via WMMA API.
 *
 * Concepts demonstrated:
 * - Warp Matrix Multiply Accumulate (WMMA)
 * - Tensor Core operations (16x16x16 tiles)
 * - Fragment-based programming model
 *
 * WMMA is the "high-level" tensor core API (vs PTX mma instructions).
 * CUDA-L2 uses the lower-level CuTe/CUTLASS abstractions.
 *
 * Compile: nvcc -O3 -arch=sm_80 04_wmma_tensor_core_gemm.cu -o wmma_gemm
 * Run: ./wmma_gemm
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

#define M 1024
#define N 1024
#define K 1024

// WMMA tile dimensions (fixed by hardware)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block tile dimensions (must be multiples of WMMA tiles)
#define BM 64   // 4 WMMA tiles in M
#define BN 64   // 4 WMMA tiles in N
#define BK 16   // 1 WMMA tile in K

// Warps per block
#define WARPS_M 2
#define WARPS_N 2
#define WARP_SIZE 32

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
 * WMMA Tensor Core GEMM kernel
 *
 * Each warp computes a 32x32 tile of C using 2x2 WMMA operations.
 * Block computes 64x64 tile using 2x2 warps.
 */
__global__ void wmma_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    // Shared memory for tiles
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    // Warp and lane IDs
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    // Warp position in block (2x2 grid of warps)
    int warp_row = warpId / WARPS_N;
    int warp_col = warpId % WARPS_N;

    // Block position
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Each warp computes 2x2 WMMA tiles = 32x32 output
    // Declare fragments for 2x2 grid of WMMA operations per warp
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[2][2];

    // Initialize accumulators to zero
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], __float2half(0.0f));
        }
    }

    // Loop over K dimension
    for (int tile_k = 0; tile_k < k; tile_k += BK) {
        // Cooperative loading into shared memory
        // Each thread loads multiple elements
        int tid = threadIdx.x;

        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int global_row = block_row + row;
            int global_col = tile_k + col;

            if (global_row < m && global_col < k) {
                As[row][col] = A[global_row * k + global_col];
            } else {
                As[row][col] = __float2half(0.0f);
            }
        }

        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int global_row = tile_k + row;
            int global_col = block_col + col;

            if (global_row < k && global_col < n) {
                Bs[row][col] = B[global_row * n + global_col];
            } else {
                Bs[row][col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Each warp loads its fragments and computes
        // Warp's position in shared memory
        int warp_smem_row = warp_row * (WMMA_M * 2);  // 32 rows per warp
        int warp_smem_col = warp_col * (WMMA_N * 2);  // 32 cols per warp

        // Load A fragments (2 tiles in M direction)
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(
                a_frag[i],
                &As[warp_smem_row + i * WMMA_M][0],
                BK  // leading dimension
            );
        }

        // Load B fragments (2 tiles in N direction)
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::load_matrix_sync(
                b_frag[j],
                &Bs[0][warp_smem_col + j * WMMA_N],
                BN  // leading dimension
            );
        }

        // Compute 2x2 WMMA operations
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }

        __syncthreads();
    }

    // Store results to global memory
    int warp_out_row = block_row + warp_row * (WMMA_M * 2);
    int warp_out_col = block_col + warp_col * (WMMA_N * 2);

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int out_row = warp_out_row + i * WMMA_M;
            int out_col = warp_out_col + j * WMMA_N;

            if (out_row < m && out_col < n) {
                wmma::store_matrix_sync(
                    &C[out_row * n + out_col],
                    c_frag[i][j],
                    n,  // leading dimension
                    wmma::mem_row_major
                );
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

void launch_wmma_gemm(const half* A, const half* B, half* C, int m, int n, int k) {
    // 4 warps per block = 128 threads
    dim3 block(WARPS_M * WARPS_N * WARP_SIZE);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    wmma_gemm_kernel<<<grid, block>>>(A, B, C, m, n, k);
}

int main() {
    printf("=== Example 04: WMMA Tensor Core GEMM ===\n");
    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("WMMA tile: %d x %d x %d\n", WMMA_M, WMMA_N, WMMA_K);
    printf("Block tile: %d x %d\n", BM, BN);
    printf("Warps per block: %d x %d\n", WARPS_M, WARPS_N);

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

    float avg_ms = benchmark_kernel(launch_wmma_gemm, d_A, d_B, d_C, M, N, K, 10, 100);

    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    printf("Average time: %.3f ms\n", avg_ms);
    printf("Performance: %.2f TFLOPS\n", tflops);
    printf("\nKey improvements:\n");
    printf("- Uses Tensor Cores (massive throughput increase)\n");
    printf("- WMMA API handles warp-level synchronization\n");
    printf("- Each warp computes 32x32 output tile\n");
    printf("\nNote: CUDA-L2 uses lower-level CuTe MMA for more control.\n");

    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
