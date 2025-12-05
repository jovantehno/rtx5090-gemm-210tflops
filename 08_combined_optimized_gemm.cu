/**
 * Example 08: Combined Optimized GEMM
 *
 * Combines ALL optimizations from previous examples:
 * - Async copy pipeline (cp.async)
 * - WMMA tensor cores
 * - Block swizzling for L2 cache
 * - Larger tiles (128x128)
 *
 * Compile: nvcc -O3 -arch=sm_80 08_combined_optimized_gemm.cu -o combined_gemm
 * Run: ./combined_gemm
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

// Matrix dimensions
#define M 4096
#define N 4096
#define K 4096

// WMMA tile dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block tile dimensions
#define BM 128
#define BN 128
#define BK 16

// Pipeline stages
#define NUM_STAGES 3

// Warp configuration: 4x4 warps = 16 warps = 512 threads
#define WARPS_M 4
#define WARPS_N 4
#define WARP_SIZE 32
#define NUM_THREADS (WARPS_M * WARPS_N * WARP_SIZE)

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
 * Combined optimized GEMM kernel
 */
template<bool UseBlockSwizzle>
__global__ void combined_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k,
    int swizzle_stride
) {
    // Shared memory: 3 stages * (128*16 + 16*128) * 2 bytes = 24KB
    __shared__ half As[NUM_STAGES][BM][BK];
    __shared__ half Bs[NUM_STAGES][BK][BN];

    // Block coordinates with optional swizzling
    int bx, by;
    if constexpr (UseBlockSwizzle) {
        int group_id = blockIdx.z;
        int blocks_per_group = swizzle_stride / BN;
        bx = group_id * blocks_per_group + blockIdx.x;
        by = blockIdx.y;
        if (bx * BN >= n || by * BM >= m) return;
    } else {
        bx = blockIdx.x;
        by = blockIdx.y;
    }

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;

    int block_row = by * BM;
    int block_col = bx * BN;

    // Each warp computes 2x2 WMMA tiles = 32x32 output
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

    int num_k_tiles = k / BK;

    // Loading pattern:
    // A: 128x16 = 2048 halfs, 512 threads
    // Load as float4 (16 bytes = 8 halfs) - cp.async only supports 4 or 16 bytes
    // 2048/8 = 256 loads needed, 512 threads -> some threads load, some don't
    // Actually: 128 rows * 16 cols = 2048 halfs
    // Each row has 16 halfs = 2 float4s
    // 128 rows * 2 float4s = 256 float4s total
    // 512 threads -> first 256 threads load, rest idle for A

    // B: 16x128 = 2048 halfs
    // Each row has 128 halfs = 16 float4s
    // 16 rows * 16 float4s = 256 float4s total

    // A tile loading: 256 float4 loads, 512 threads
    int a_thread_row = tid / 2;    // 0-255 maps to row 0-127 (2 float4 per row)
    int a_thread_col = (tid % 2) * 8;  // 0 or 8

    // B tile loading: 256 float4 loads
    int b_thread_row = tid / 16;   // 0-31 maps to row 0-15
    int b_thread_col = (tid % 16) * 8;  // 0, 8, 16, ... 120

    // === Prologue: Fill pipeline ===
    #pragma unroll
    for (int stage = 0; stage < NUM_STAGES - 1 && stage < num_k_tiles; stage++) {
        int tile_k = stage * BK;

        // Load A: float4 (8 halfs) per active thread (first 256 threads)
        if (tid < 256) {
            int row = a_thread_row;
            int col = a_thread_col;
            int global_row = block_row + row;
            int global_col = tile_k + col;

            const half* src = &A[global_row * k + global_col];
            half* dst = &As[stage][row][col];

            // Load 16 bytes = 8 halfs
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                  "l"(src)
            );
        }

        // Load B: float4 (8 halfs) per active thread (first 256 threads)
        if (tid < 256) {
            int row = b_thread_row;
            int col = b_thread_col;
            int global_row = tile_k + row;
            int global_col = block_col + col;

            const half* src = &B[global_row * n + global_col];
            half* dst = &Bs[stage][row][col];

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                  "l"(src)
            );
        }

        asm volatile("cp.async.commit_group;\n" ::);
    }

    // === Main loop ===
    for (int tile = 0; tile < num_k_tiles; tile++) {
        int compute_stage = tile % NUM_STAGES;
        int load_tile_idx = tile + NUM_STAGES - 1;
        int load_stage = load_tile_idx % NUM_STAGES;

        // Wait for compute stage
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 2));
        __syncthreads();

        // Issue loads for future tile
        if (load_tile_idx < num_k_tiles) {
            int tile_k = load_tile_idx * BK;

            // Load A
            if (tid < 256) {
                int row = a_thread_row;
                int col = a_thread_col;
                int global_row = block_row + row;
                int global_col = tile_k + col;

                const half* src = &A[global_row * k + global_col];
                half* dst = &As[load_stage][row][col];

                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                      "l"(src)
                );
            }

            // Load B
            if (tid < 256) {
                int row = b_thread_row;
                int col = b_thread_col;
                int global_row = tile_k + row;
                int global_col = block_col + col;

                const half* src = &B[global_row * n + global_col];
                half* dst = &Bs[load_stage][row][col];

                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                      "l"(src)
                );
            }

            asm volatile("cp.async.commit_group;\n" ::);
        }

        // === Compute with WMMA ===
        int warp_smem_row = warp_row * (WMMA_M * 2);
        int warp_smem_col = warp_col * (WMMA_N * 2);

        // BK=16 = WMMA_K, so single iteration
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(
                a_frag[i],
                &As[compute_stage][warp_smem_row + i * WMMA_M][0],
                BK
            );
        }

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::load_matrix_sync(
                b_frag[j],
                &Bs[compute_stage][0][warp_smem_col + j * WMMA_N],
                BN
            );
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

    // === Store results ===
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

template<bool UseBlockSwizzle>
float benchmark_kernel(
    const half* d_A, const half* d_B, half* d_C,
    int m, int n, int k,
    int swizzle_stride,
    int warmup_iters, int bench_iters
) {
    int blocks_m = m / BM;
    int blocks_n = n / BN;

    dim3 block(NUM_THREADS);
    dim3 grid;

    if constexpr (UseBlockSwizzle) {
        int blocks_per_group = swizzle_stride / BN;
        int num_groups = (blocks_n + blocks_per_group - 1) / blocks_per_group;
        int local_blocks_x = (blocks_n + num_groups - 1) / num_groups;
        grid = dim3(local_blocks_x, blocks_m, num_groups);
    } else {
        grid = dim3(blocks_n, blocks_m, 1);
    }

    for (int i = 0; i < warmup_iters; i++) {
        combined_gemm_kernel<UseBlockSwizzle><<<grid, block>>>(
            d_A, d_B, d_C, m, n, k, swizzle_stride
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        combined_gemm_kernel<UseBlockSwizzle><<<grid, block>>>(
            d_A, d_B, d_C, m, n, k, swizzle_stride
        );
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
    printf("=== Example 08: Combined Optimized GEMM ===\n");
    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("Block tile: %d x %d x %d\n", BM, BN, BK);
    printf("Pipeline stages: %d\n", NUM_STAGES);
    printf("Threads per block: %d (%dx%d warps)\n", NUM_THREADS, WARPS_M, WARPS_N);
    printf("\n");

    static_assert(M % BM == 0, "M must be multiple of BM");
    static_assert(N % BN == 0, "N must be multiple of BN");
    static_assert(K % BK == 0, "K must be multiple of BK");

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

    // Test without block swizzle
    printf("Without block swizzling:\n");
    float no_swizzle_ms = benchmark_kernel<false>(d_A, d_B, d_C, M, N, K, 0, 5, 20);
    double no_swizzle_tflops = (2.0 * M * N * K / (no_swizzle_ms / 1000.0)) / 1e12;
    printf("  Time: %.3f ms, Performance: %.2f TFLOPS\n\n", no_swizzle_ms, no_swizzle_tflops);

    // Test with different swizzle strides
    printf("With block swizzling (testing strides):\n");
    int strides[] = {2048, 3072, 4096, 6144, 8192};
    float best_ms = no_swizzle_ms;
    int best_stride = 0;

    for (int stride : strides) {
        float ms = benchmark_kernel<true>(d_A, d_B, d_C, M, N, K, stride, 5, 20);
        double tflops = (2.0 * M * N * K / (ms / 1000.0)) / 1e12;
        float speedup = no_swizzle_ms / ms;

        printf("  Stride %4d: %.3f ms, %.2f TFLOPS (%.2fx)\n", stride, ms, tflops, speedup);

        if (ms < best_ms) {
            best_ms = ms;
            best_stride = stride;
        }
    }

    printf("\n=== Summary ===\n");
    double best_tflops = (2.0 * M * N * K / (best_ms / 1000.0)) / 1e12;
    printf("Best configuration: stride=%d\n", best_stride);
    printf("Best performance: %.2f TFLOPS\n", best_tflops);
    printf("Speedup over no-swizzle: %.2fx\n", no_swizzle_ms / best_ms);

    printf("\n=== Optimizations Applied ===\n");
    printf("1. Async copy pipeline (cp.async) - %d stages\n", NUM_STAGES);
    printf("2. WMMA tensor cores - 16x16x16 tiles\n");
    printf("3. Large output tiles - %dx%d per block\n", BM, BN);
    printf("4. Block swizzling for L2 cache\n");
    printf("5. 16 warps per block (512 threads)\n");
    printf("\nCompare with: python3 compare_cublas.py --size 4096\n");

    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
