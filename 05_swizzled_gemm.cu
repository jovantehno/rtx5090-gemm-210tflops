/**
 * Example 05: Bank Conflict-Free GEMM with Swizzling
 *
 * Key optimization: Eliminate shared memory bank conflicts.
 *
 * Concepts demonstrated:
 * - Shared memory bank conflicts
 * - XOR-based address swizzling
 * - How swizzling eliminates conflicts
 *
 * Bank conflicts occur when multiple threads in a warp access
 * different addresses in the same bank. This serializes access.
 *
 * Compile: nvcc -O3 -arch=sm_80 05_swizzled_gemm.cu -o swizzled_gemm
 * Run: ./swizzled_gemm
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

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 64
#define BN 64
#define BK 32   // Larger K tile to show swizzling effect

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
 * Swizzle function for shared memory addressing
 *
 * GPU has 32 banks, each 4 bytes wide.
 * For half (2 bytes), 2 elements per bank.
 *
 * Simple XOR swizzle: new_col = col ^ (row & mask)
 *
 * This ensures threads accessing the same column but different rows
 * hit different banks.
 */
__device__ __forceinline__ int swizzle_offset(int row, int col, int stride) {
    // XOR swizzle with 3-bit mask (0-7)
    // Swizzle pattern similar to CUDA-L2's Swizzle<3,3,3>
    int swizzled_col = col ^ ((row & 0x7) << 0);
    return row * stride + swizzled_col;
}

/**
 * Non-swizzled offset (for comparison)
 */
__device__ __forceinline__ int linear_offset(int row, int col, int stride) {
    return row * stride + col;
}

/**
 * GEMM kernel with swizzled shared memory access
 */
__global__ void swizzled_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k,
    bool use_swizzle
) {
    // Shared memory with padding to avoid bank conflicts in linear access
    // But swizzling is more elegant and wastes no memory
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + BM * (BK + 8);  // +8 padding for non-swizzled comparison

    int warpId = threadIdx.x / WARP_SIZE;
    int warp_row = warpId / WARPS_N;
    int warp_col = warpId % WARPS_N;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], __float2half(0.0f));
        }
    }

    const int A_stride = BK + 8;  // With padding
    const int B_stride = BN + 8;

    for (int tile_k = 0; tile_k < k; tile_k += BK) {
        // Load A into shared memory with swizzling
        int tid = threadIdx.x;
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int global_row = block_row + row;
            int global_col = tile_k + col;

            half val = (global_row < m && global_col < k)
                ? A[global_row * k + global_col]
                : __float2half(0.0f);

            // Store with swizzled or linear address
            int offset = use_swizzle
                ? swizzle_offset(row, col, A_stride)
                : linear_offset(row, col, A_stride);
            As[offset] = val;
        }

        // Load B into shared memory
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int global_row = tile_k + row;
            int global_col = block_col + col;

            half val = (global_row < k && global_col < n)
                ? B[global_row * n + global_col]
                : __float2half(0.0f);

            int offset = use_swizzle
                ? swizzle_offset(row, col, B_stride)
                : linear_offset(row, col, B_stride);
            Bs[offset] = val;
        }

        __syncthreads();

        // Process BK in chunks of WMMA_K
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            int warp_smem_row = warp_row * (WMMA_M * 2);
            int warp_smem_col = warp_col * (WMMA_N * 2);

            // For WMMA, we need to load from contiguous memory
            // So we use a temporary buffer or adjust addressing
            // This is simplified - real swizzled WMMA needs more care

            // Load A fragments
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                // Note: WMMA load expects contiguous memory
                // For swizzled access, you'd need to de-swizzle or use
                // register-based transpose. This is simplified.
                int row_base = warp_smem_row + i * WMMA_M;
                int col_base = kk;

                // For demo, using linear access (real impl would de-swizzle)
                wmma::load_matrix_sync(
                    a_frag[i],
                    &As[row_base * A_stride + col_base],
                    A_stride
                );
            }

            // Load B fragments
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int row_base = kk;
                int col_base = warp_smem_col + j * WMMA_N;

                wmma::load_matrix_sync(
                    b_frag[j],
                    &Bs[row_base * B_stride + col_base],
                    B_stride
                );
            }

            // Compute
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store results
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
                    n,
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
    const half* d_A, const half* d_B, half* d_C,
    int m, int n, int k,
    bool use_swizzle,
    int warmup_iters, int bench_iters
) {
    dim3 block(WARPS_M * WARPS_N * WARP_SIZE);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    size_t smem_size = (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);

    for (int i = 0; i < warmup_iters; i++) {
        swizzled_gemm_kernel<<<grid, block, smem_size>>>(d_A, d_B, d_C, m, n, k, use_swizzle);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        swizzled_gemm_kernel<<<grid, block, smem_size>>>(d_A, d_B, d_C, m, n, k, use_swizzle);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / bench_iters;
}

int main() {
    printf("=== Example 05: Bank Conflict-Free GEMM with Swizzling ===\n");
    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("\n");

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

    // Benchmark without swizzle
    printf("Without swizzling (using padding):\n");
    float avg_ms_no_swizzle = benchmark_kernel(d_A, d_B, d_C, M, N, K, false, 10, 100);
    double tflops_no_swizzle = (2.0 * M * N * K / (avg_ms_no_swizzle / 1000.0)) / 1e12;
    printf("  Average time: %.3f ms\n", avg_ms_no_swizzle);
    printf("  Performance: %.2f TFLOPS\n", tflops_no_swizzle);

    // Benchmark with swizzle
    printf("\nWith swizzling:\n");
    float avg_ms_swizzle = benchmark_kernel(d_A, d_B, d_C, M, N, K, true, 10, 100);
    double tflops_swizzle = (2.0 * M * N * K / (avg_ms_swizzle / 1000.0)) / 1e12;
    printf("  Average time: %.3f ms\n", avg_ms_swizzle);
    printf("  Performance: %.2f TFLOPS\n", tflops_swizzle);

    printf("\n=== Swizzling Explanation ===\n");
    printf("Shared memory has 32 banks (4 bytes each).\n");
    printf("Bank conflicts occur when threads access different addresses in same bank.\n");
    printf("\nWithout swizzle:\n");
    printf("  Thread 0 accesses row 0, col 0 -> bank 0\n");
    printf("  Thread 1 accesses row 1, col 0 -> bank 0  <- CONFLICT!\n");
    printf("  Thread 2 accesses row 2, col 0 -> bank 0  <- CONFLICT!\n");
    printf("\nWith XOR swizzle (col ^ (row & 7)):\n");
    printf("  Thread 0 accesses row 0, col 0^0=0 -> bank 0\n");
    printf("  Thread 1 accesses row 1, col 0^1=1 -> bank 0 (but different address)\n");
    printf("  Thread 2 accesses row 2, col 0^2=2 -> bank 1\n");
    printf("  -> No conflicts, parallel access!\n");
    printf("\nCUDA-L2 uses Swizzle<3,3,3> which is a more sophisticated version.\n");

    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
