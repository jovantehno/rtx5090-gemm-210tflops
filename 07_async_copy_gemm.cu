/**
 * Example 07: Async Copy Pipeline (cp.async)
 *
 * Key optimization: Use hardware-accelerated async memory copies.
 *
 * Concepts demonstrated:
 * - cp.async instruction (SM80+)
 * - Multi-stage pipelining with async copies
 * - Commit/wait groups for pipeline control
 *
 * This is the Ampere (SM80) way. Blackwell uses TMA instead.
 *
 * Compile: nvcc -O3 -arch=sm_80 07_async_copy_gemm.cu -o async_copy_gemm
 * Run: ./async_copy_gemm
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

// Matrix dimensions - must be multiples of tile sizes for this simplified example
#define M 1024
#define N 1024
#define K 1024

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 64
#define BN 64
#define BK 16

// Number of pipeline stages
#define NUM_STAGES 3

#define WARPS_M 2
#define WARPS_N 2
#define WARP_SIZE 32
#define NUM_THREADS (WARPS_M * WARPS_N * WARP_SIZE)  // 128

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
 * GEMM kernel with async copy pipeline
 *
 * This version uses vectorized loads (float4 = 16 bytes = 8 halfs)
 * which guarantees alignment when matrix dimensions are multiples of 8.
 *
 * For production code, you'd handle edge cases separately.
 */
__global__ void async_copy_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    // Multi-buffer shared memory for pipelining
    __shared__ __align__(16) half As[NUM_STAGES][BM][BK];
    __shared__ __align__(16) half Bs[NUM_STAGES][BK][BN];

    int warpId = threadIdx.x / WARP_SIZE;
    int warp_row = warpId / WARPS_N;
    int warp_col = warpId % WARPS_N;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tid = threadIdx.x;

    // Fragments for WMMA
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

    int num_tiles = k / BK;

    // For A: Each thread loads (BM * BK) / NUM_THREADS elements
    // BM=64, BK=16, NUM_THREADS=128 -> 8 elements per thread
    // We load as float4 (8 halfs) per thread
    constexpr int A_ELEMENTS_PER_THREAD = (BM * BK) / NUM_THREADS;  // 8
    static_assert(A_ELEMENTS_PER_THREAD == 8, "Adjust loading logic");

    // For B: BK=16, BN=64 -> 8 elements per thread
    constexpr int B_ELEMENTS_PER_THREAD = (BK * BN) / NUM_THREADS;  // 8
    static_assert(B_ELEMENTS_PER_THREAD == 8, "Adjust loading logic");

    // Calculate thread's load position for A (loading as float4 = 8 halfs)
    // We have 64*16 = 1024 halfs, 128 float4s, 128 threads -> 1 float4 per thread
    int a_load_row = tid / (BK / 8);   // tid / 2 (BK=16, 2 float4s per row)
    int a_load_col = (tid % (BK / 8)) * 8;  // 0 or 8

    // Calculate thread's load position for B
    // 16*64 = 1024 halfs, 128 float4s
    int b_load_row = tid / (BN / 8);   // tid / 8
    int b_load_col = (tid % (BN / 8)) * 8;

    // === Prologue: Fill pipeline ===
    #pragma unroll
    for (int stage = 0; stage < NUM_STAGES - 1; stage++) {
        int tile_k = stage * BK;

        // Async copy A tile using cp.async with 16-byte copies
        {
            int global_row = block_row + a_load_row;
            int global_col = tile_k + a_load_col;

            const half* src = &A[global_row * k + global_col];
            half* dst = &As[stage][a_load_row][a_load_col];

            // cp.async for 16 bytes (8 halfs)
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :
                : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                  "l"(src)
            );
        }

        // Async copy B tile
        {
            int global_row = tile_k + b_load_row;
            int global_col = block_col + b_load_col;

            const half* src = &B[global_row * n + global_col];
            half* dst = &Bs[stage][b_load_row][b_load_col];

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
    for (int tile = 0; tile < num_tiles; tile++) {
        int compute_stage = tile % NUM_STAGES;
        int load_tile_idx = tile + NUM_STAGES - 1;
        int load_stage = load_tile_idx % NUM_STAGES;

        // Wait for compute stage data
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 2));
        __syncthreads();

        // Issue loads for future tile
        if (load_tile_idx < num_tiles) {
            int tile_k = load_tile_idx * BK;

            // Load A
            {
                int global_row = block_row + a_load_row;
                int global_col = tile_k + a_load_col;

                const half* src = &A[global_row * k + global_col];
                half* dst = &As[load_stage][a_load_row][a_load_col];

                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                      "l"(src)
                );
            }

            // Load B
            {
                int global_row = tile_k + b_load_row;
                int global_col = block_col + b_load_col;

                const half* src = &B[global_row * n + global_col];
                half* dst = &Bs[load_stage][b_load_row][b_load_col];

                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                      "l"(src)
                );
            }

            asm volatile("cp.async.commit_group;\n" ::);
        }

        // === Compute using WMMA ===
        int warp_smem_row = warp_row * (WMMA_M * 2);
        int warp_smem_col = warp_col * (WMMA_N * 2);

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

            wmma::store_matrix_sync(
                &C[out_row * n + out_col],
                c_frag[i][j],
                n,
                wmma::mem_row_major
            );
        }
    }
}

/**
 * Reference kernel without async (for comparison)
 */
__global__ void sync_copy_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    int warpId = threadIdx.x / WARP_SIZE;
    int warp_row = warpId / WARPS_N;
    int warp_col = warpId % WARPS_N;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tid = threadIdx.x;

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

    // Same vectorized loading pattern but synchronous
    int a_load_row = tid / (BK / 8);
    int a_load_col = (tid % (BK / 8)) * 8;
    int b_load_row = tid / (BN / 8);
    int b_load_col = (tid % (BN / 8)) * 8;

    for (int tile_k = 0; tile_k < k; tile_k += BK) {
        // Synchronous vectorized load A
        {
            int global_row = block_row + a_load_row;
            int global_col = tile_k + a_load_col;

            float4 tmp = *reinterpret_cast<const float4*>(&A[global_row * k + global_col]);
            *reinterpret_cast<float4*>(&As[a_load_row][a_load_col]) = tmp;
        }

        // Synchronous vectorized load B
        {
            int global_row = tile_k + b_load_row;
            int global_col = block_col + b_load_col;

            float4 tmp = *reinterpret_cast<const float4*>(&B[global_row * n + global_col]);
            *reinterpret_cast<float4*>(&Bs[b_load_row][b_load_col]) = tmp;
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

    int warp_out_row = block_row + warp_row * (WMMA_M * 2);
    int warp_out_col = block_col + warp_col * (WMMA_N * 2);

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int out_row = warp_out_row + i * WMMA_M;
            int out_col = warp_out_col + j * WMMA_N;

            wmma::store_matrix_sync(&C[out_row * n + out_col], c_frag[i][j], n, wmma::mem_row_major);
        }
    }
}

void init_matrix(half* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = __float2half((float)(rand() % 10) / 10.0f);
    }
}

template<typename KernelFunc>
float benchmark_kernel(
    KernelFunc kernel,
    const half* d_A, const half* d_B, half* d_C,
    int m, int n, int k,
    int warmup_iters, int bench_iters
) {
    dim3 block(NUM_THREADS);
    dim3 grid(n / BN, m / BM);

    for (int i = 0; i < warmup_iters; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
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
    printf("=== Example 07: Async Copy Pipeline (cp.async) ===\n");
    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("Pipeline stages: %d\n", NUM_STAGES);
    printf("Block tile: %dx%dx%d\n", BM, BN, BK);
    printf("\n");

    // Verify dimensions are compatible
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

    // Benchmark synchronous version
    printf("Synchronous memory copy (vectorized float4):\n");
    float sync_ms = benchmark_kernel(sync_copy_gemm_kernel, d_A, d_B, d_C, M, N, K, 10, 100);
    double sync_tflops = (2.0 * M * N * K / (sync_ms / 1000.0)) / 1e12;
    printf("  Average time: %.3f ms\n", sync_ms);
    printf("  Performance: %.2f TFLOPS\n", sync_tflops);

    // Benchmark async version
    printf("\nAsync copy pipeline (cp.async.cg 16-byte):\n");
    float async_ms = benchmark_kernel(async_copy_gemm_kernel, d_A, d_B, d_C, M, N, K, 10, 100);
    double async_tflops = (2.0 * M * N * K / (async_ms / 1000.0)) / 1e12;
    printf("  Average time: %.3f ms\n", async_ms);
    printf("  Performance: %.2f TFLOPS\n", async_tflops);

    if (async_ms < sync_ms) {
        printf("  Speedup: %.2fx faster\n", sync_ms / async_ms);
    } else {
        printf("  Note: Sync is faster here (%.2fx) - async shines on larger problems\n", async_ms / sync_ms);
    }

    printf("\n=== Async Copy Explanation ===\n");
    printf("\ncp.async.cg.shared.global [dst], [src], size:\n");
    printf("  - 'cg' = cache global (L2 cache hint)\n");
    printf("  - size: 4, 8, or 16 bytes\n");
    printf("  - Both src and dst must be aligned to 'size'\n");
    printf("\nPipeline control:\n");
    printf("  cp.async.commit_group  - Mark batch of copies as a group\n");
    printf("  cp.async.wait_group N  - Wait until <= N groups pending\n");
    printf("\nMemory timeline:\n");
    printf("  Sync:  [Load][Wait][Compute][Load][Wait][Compute]...\n");
    printf("  Async: [Load0][Load1][Compute0+Load2][Compute1+Load3]...\n");
    printf("         ^-- memory latency hidden by pipelining\n");
    printf("\nCUDA-L2 uses CUTLASS CuTe equivalent:\n");
    printf("  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;\n");
    printf("  cp_async_fence();  // same as commit_group\n");
    printf("  cp_async_wait<N>(); // same as wait_group\n");
    printf("\nFor Blackwell (SM100):\n");
    printf("  TMA (Tensor Memory Accelerator) replaces cp.async\n");
    printf("  - Hardware handles 2D/3D tensor addressing\n");
    printf("  - Supports swizzled layouts directly\n");
    printf("  - Enables warp specialization patterns\n");

    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
