/**
 * Example 06: Block Swizzling for L2 Cache Optimization
 *
 * Key optimization: Reorder thread block execution for better L2 cache reuse.
 *
 * Concepts demonstrated:
 * - L2 cache behavior in GEMM
 * - Block scheduling patterns
 * - Swizzle stride tuning
 *
 * This is one of CUDA-L2's key optimizations that contributes to
 * beating cuBLAS.
 *
 * Compile: nvcc -O3 -arch=sm_80 06_block_swizzle_l2_gemm.cu -o block_swizzle_gemm
 * Run: ./block_swizzle_gemm
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

#define M 4096
#define N 4096
#define K 4096

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 64
#define BN 64
#define BK 16

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
 * GEMM kernel with block swizzling for L2 cache optimization
 *
 * Without swizzling: blocks execute in row-major order
 *   Block (0,0), (1,0), (2,0), ... (0,1), (1,1), ...
 *
 * This causes poor L2 cache utilization because:
 * - Blocks in same row of grid load same rows of A
 * - By the time we get to next row, A data is evicted
 *
 * With swizzling: blocks execute in a pattern that maximizes reuse
 *   Execute blocks in "swizzle groups" that share data
 */
template<bool UseBlockSwizzle>
__global__ void block_swizzle_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k,
    int swizzle_stride,
    int grid_dim_x
) {
    // Shared memory
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    // Calculate actual block coordinates with swizzling
    int bx, by;
    if constexpr (UseBlockSwizzle) {
        // Block swizzle pattern from CUDA-L2
        // Divide grid into "swizzle groups" along N dimension
        // Within each group, execute in a pattern that maximizes B reuse

        int group_id = blockIdx.z;           // Which swizzle group
        int local_bx = blockIdx.x;           // X within group
        int by_raw = blockIdx.y;

        // Map to actual block coordinates
        bx = group_id * (swizzle_stride / BN) + local_bx;
        by = by_raw;

        // Early exit if out of bounds (can happen with swizzling)
        if (bx * BN >= n || by * BM >= m) return;
    } else {
        bx = blockIdx.x;
        by = blockIdx.y;
    }

    int warpId = threadIdx.x / WARP_SIZE;
    int warp_row = warpId / WARPS_N;
    int warp_col = warpId % WARPS_N;

    int block_row = by * BM;
    int block_col = bx * BN;

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

    // Main loop
    for (int tile_k = 0; tile_k < k; tile_k += BK) {
        int tid = threadIdx.x;

        // Load A
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int global_row = block_row + row;
            int global_col = tile_k + col;

            As[row][col] = (global_row < m && global_col < k)
                ? A[global_row * k + global_col]
                : __float2half(0.0f);
        }

        // Load B
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int global_row = tile_k + row;
            int global_col = block_col + col;

            Bs[row][col] = (global_row < k && global_col < n)
                ? B[global_row * n + global_col]
                : __float2half(0.0f);
        }

        __syncthreads();

        int warp_smem_row = warp_row * (WMMA_M * 2);
        int warp_smem_col = warp_col * (WMMA_N * 2);

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(a_frag[i], &As[warp_smem_row + i * WMMA_M][0], BK);
        }

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::load_matrix_sync(b_frag[j], &Bs[0][warp_smem_col + j * WMMA_N], BN);
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
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
                wmma::store_matrix_sync(&C[out_row * n + out_col], c_frag[i][j], n, wmma::mem_row_major);
            }
        }
    }
}

void init_matrix(half* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = __float2half((float)(rand() % 10) / 10.0f);
    }
}

template<bool UseBlockSwizzle>
float benchmark_kernel(
    const half* d_A, const half* d_B, half* d_C,
    int m, int n, int k,
    int swizzle_stride,
    int warmup_iters, int bench_iters
) {
    int blocks_m = (m + BM - 1) / BM;
    int blocks_n = (n + BN - 1) / BN;

    dim3 block(WARPS_M * WARPS_N * WARP_SIZE);
    dim3 grid;

    if constexpr (UseBlockSwizzle) {
        // With swizzling: use 3D grid
        int blocks_per_group = swizzle_stride / BN;
        int num_groups = (blocks_n + blocks_per_group - 1) / blocks_per_group;
        int local_blocks_x = (blocks_n + num_groups - 1) / num_groups;

        grid = dim3(local_blocks_x, blocks_m, num_groups);
    } else {
        grid = dim3(blocks_n, blocks_m);
    }

    for (int i = 0; i < warmup_iters; i++) {
        block_swizzle_gemm_kernel<UseBlockSwizzle><<<grid, block>>>(
            d_A, d_B, d_C, m, n, k, swizzle_stride, blocks_n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        block_swizzle_gemm_kernel<UseBlockSwizzle><<<grid, block>>>(
            d_A, d_B, d_C, m, n, k, swizzle_stride, blocks_n);
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
    printf("=== Example 06: Block Swizzling for L2 Cache Optimization ===\n");
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

    // Test different swizzle strides
    // RTX 5090 has ~96MB L2 (vs A100's 40MB), so larger strides may be optimal
    int swizzle_strides[] = {1024, 1792, 2048, 3072, 4096, 6144, 8192};

    printf("Without block swizzling:\n");
    float avg_ms_no_swizzle = benchmark_kernel<false>(d_A, d_B, d_C, M, N, K, 0, 5, 20);
    double tflops_no_swizzle = (2.0 * M * N * K / (avg_ms_no_swizzle / 1000.0)) / 1e12;
    printf("  Average time: %.3f ms\n", avg_ms_no_swizzle);
    printf("  Performance: %.2f TFLOPS\n\n", tflops_no_swizzle);

    printf("With block swizzling (varying stride):\n");
    for (int stride : swizzle_strides) {
        float avg_ms = benchmark_kernel<true>(d_A, d_B, d_C, M, N, K, stride, 5, 20);
        double tflops = (2.0 * M * N * K / (avg_ms / 1000.0)) / 1e12;
        float speedup = avg_ms_no_swizzle / avg_ms;
        printf("  Stride %4d: %.3f ms, %.2f TFLOPS (%.2fx)\n",
               stride, avg_ms, tflops, speedup);
    }

    printf("\n=== Block Swizzling Explanation ===\n");
    printf("\nProblem with naive block scheduling:\n");
    printf("  Grid: (Bx, By) blocks\n");
    printf("  Execution order: (0,0), (1,0), (2,0), ..., (0,1), (1,1), ...\n");
    printf("\n");
    printf("  For C[i,j] = A[i,:] * B[:,j]:\n");
    printf("  - Blocks (0,0), (1,0), (2,0) all load A[0:BM, :] (same rows!)\n");
    printf("  - By the time (0,1) runs, A[0:BM,:] is evicted from L2\n");
    printf("  - Must reload same A data -> poor cache utilization\n");
    printf("\n");
    printf("Solution with block swizzling:\n");
    printf("  Group blocks that share data and execute them together.\n");
    printf("  \n");
    printf("  Example with swizzle_stride=1024 (16 BN=64 blocks):\n");
    printf("    Group 0: blocks (0-15, 0), (0-15, 1), ... all run ~together\n");
    printf("    These share columns of B -> better B reuse\n");
    printf("    Group 1: blocks (16-31, 0), (16-31, 1), ...\n");
    printf("\n");
    printf("  CUDA-L2 uses swizzle_stride=1792 for A100 (tuned value).\n");
    printf("  Optimal stride depends on L2 cache size and matrix dimensions.\n");
    printf("\n");
    printf("For RTX 5090 (96MB L2 vs A100's 40MB):\n");
    printf("  -> Larger swizzle_stride may be optimal\n");
    printf("  -> More data fits in cache\n");

    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
